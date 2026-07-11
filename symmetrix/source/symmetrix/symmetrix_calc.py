"""ASE Calculator for symmetrix implementation of equivariant graph neural
network library

This file was written and publicly released by Dr. Noam Bernstein as part of his
work for the U. S. Government, and is not subject to copyright.
"""
import json
import logging
import copy
from tempfile import NamedTemporaryFile
import numpy as np

try:
    from matscipy.neighbours import neighbour_list as neighbor_list
except:
    logging.warning("Symmetrix using slow ase.neighborlist.neighbor_list")
    from ase.neighborlist import neighbor_list

from ase.calculators.calculator import Calculator, PropertyNotImplementedError, all_changes, equal
from ase.stress import full_3x3_to_voigt_6_stress

from . import symmetrix

class Symmetrix(Calculator):
    """ASE Calculator using symmetrix library to evaluate equivariant graph neural network 
    potential energy functions

    Parameters
    ----------
    model_file: str
        JSON-format model file used for potential energy

    Notes
    -----
    Wraps symmetrix library from https://github.com/wcwitt/symmetrix via python interface at https://pypi.org/project/symmetrix/
    """
    implemented_properties = ['energy', 'free_energy', 'energies', 'forces', 'stress']
    _macefield_response_properties = ['polarization', 'becs', 'polarizability']
    _macefield_eps0 = 8.8541878128e-12 / 1.602176634e-19 / 1e10


    def __init__(self, model_file, dtype="float64", use_kokkos=True, **kwargs):
        Calculator.__init__(self, **kwargs)
        if dtype not in ["float32", "float64"]:
            raise ValueError(f"Unsupported dtype '{dtype}'. Supported dtypes are 'float64' and 'float32'.")
        self._macefield_info = None
        self._electric_field = kwargs.get("electric_field", None)
        self._model_has_field_coupling = self._json_has_field_coupling(model_file)

        if use_kokkos and self._model_has_field_coupling:
            if dtype == "float32":
                raise ValueError("MACEField JSON models require dtype 'float64' in the native serial path.")
            use_kokkos = False

        if use_kokkos and not hasattr(symmetrix, "MACEKokkos"):
            if not str(model_file).endswith(".json"):
                self._raise_if_macefield_checkpoint(model_file)
            raise RuntimeError("Symmetrix was built without Kokkos support.")
        self.use_kokkos = use_kokkos
        if self.use_kokkos:
            if not symmetrix._kokkos_is_initialized():
                symmetrix._init_kokkos()
            MACE = symmetrix.MACEKokkos if dtype == "float64" else symmetrix.MACEKokkosFloat
        else:
            if dtype == "float32":
                raise ValueError(f"dtype '{dtype}' requires `use_kokkos = True`")
            MACE = symmetrix.MACE
        try:
            self.evaluator = MACE(str(model_file))
        except RuntimeError: # expecting json.exception.parse_error.101
            self._raise_if_macefield_checkpoint(model_file)

            # import this here so that torch/mace support isn't needed if file is already symmetrix json
            from .extract_mace_data import extract_mace_data
            kwargs_extract = {k: v for k, v in kwargs.items()
                if k in ['species',
                         'head',
                         'num_spline_points']}
            logging.warning(f"Converting model from pytorch model to symmetrix dict with {kwargs_extract}")
            data = extract_mace_data(model_file, **kwargs_extract)
            with NamedTemporaryFile("w") as fout:
                logging.warning(f"Converting via NamedTemporaryFile {fout.name}")
                fout.write(json.dumps(data))
                self.evaluator = MACE(fout.name)

        self.cutoff = self.evaluator.r_cut
        self.implemented_properties = list(type(self).implemented_properties)
        if self._has_native_field_coupling():
            self.implemented_properties.extend(self._macefield_response_properties)

    def _json_has_field_coupling(self, model_file):
        if not str(model_file).endswith(".json"):
            return False
        try:
            with open(model_file) as fin:
                return bool(json.load(fin).get("has_field_coupling", False))
        except (OSError, json.JSONDecodeError, AttributeError):
            return False

    def _raise_if_macefield_checkpoint(self, model_file):
        try:
            import torch
        except ImportError:
            return

        model = torch.load(
            model_file,
            map_location=torch.device("cpu"),
            weights_only=False,
        )

        is_macefield = (
            type(model).__name__ == "MACEField"
            or (hasattr(model, "field_feats") and hasattr(model, "field_linear"))
        )
        if is_macefield:
            raise RuntimeError(
                "MACEField PyTorch checkpoints cannot be used directly with Symmetrix. "
                "Convert/extract the model to Symmetrix JSON first, then pass the JSON file.")

    def check_state(self, atoms, tol=1e-15):
        state = super().check_state(atoms, tol=tol)
        if (
            self._has_native_field_coupling()
            and not state
            and (
                not hasattr(self, "atoms")
                or not equal(self._macefield_info, getattr(atoms, "info", {}), atol=tol)
            )
        ):
            state.append("info")
        return state

    def _has_native_field_coupling(self):
        return (
            hasattr(self, "evaluator")
            and getattr(self.evaluator, "has_field_coupling", False)
        )

    def _resolve_electric_field(self, atoms=None, require_graph=False):
        if atoms is None:
            atoms = self.atoms

        if self._electric_field is not None:
            field = self._electric_field
        elif "electric_field" in atoms.info:
            field = atoms.info["electric_field"]
        elif "REF_electric_field" in atoms.info:
            field = atoms.info["REF_electric_field"]
        else:
            field = np.zeros(3)

        field = np.asarray(field, dtype=float)
        if field.shape == (3,):
            return field
        if field.shape != (len(atoms), 3):
            raise ValueError("electric_field must have shape (3,) or (natoms, 3).")
        if require_graph:
            raise PropertyNotImplementedError(
                "MACEField response properties require a graph-level electric_field with shape (3,).")
        return field

    def _mace_inputs(self, atoms):
        ase_atomic_numbers = atoms.get_atomic_numbers().tolist()
        mace_atomic_numbers = self.evaluator.atomic_numbers
        i_list, j_list, r, xyz = neighbor_list('ijdD', atoms, self.cutoff)
        num_nodes = len(atoms)
        node_types = [mace_atomic_numbers.index(ase_atomic_numbers[i]) for i in range(num_nodes)]
        num_neigh = np.bincount(j_list, minlength=num_nodes)
        neigh_types = [mace_atomic_numbers.index(ase_atomic_numbers[j]) for j in j_list]
        return num_nodes, node_types, num_neigh, j_list, neigh_types, xyz, r, i_list

    def _compute_macefield(self, atoms, electric_field):
        num_nodes, node_types, num_neigh, j_list, neigh_types, xyz, r, i_list = self._mace_inputs(atoms)
        self.evaluator.compute_node_energies_forces_field(
            num_nodes,
            node_types,
            num_neigh,
            j_list,
            neigh_types,
            xyz.flatten(),
            r,
            np.asarray(electric_field, dtype=float).flatten(),
        )
        return num_nodes, i_list, j_list, xyz

    def _calculate_macefield_responses(self, electric_field, properties):
        volume = self.atoms.get_volume()
        raw_polarization = -np.asarray(self.evaluator.electric_field_adj, dtype=float)
        if raw_polarization.shape != (3,):
            raise PropertyNotImplementedError(
                "MACEField response properties require a graph-level electric_field with shape (3,).")

        self.results['polarization'] = raw_polarization / volume

        field_derivatives_computed = False
        force_derivatives_computed = False

        def compute_field_derivatives(include_forces=False):
            nonlocal field_derivatives_computed, force_derivatives_computed
            if include_forces and force_derivatives_computed:
                return
            if field_derivatives_computed and not include_forces:
                return
            num_nodes, node_types, num_neigh, j_list, neigh_types, xyz, r, _ = self._mace_inputs(self.atoms)
            if include_forces:
                self.evaluator.compute_electric_field_force_derivative(
                    num_nodes,
                    node_types,
                    num_neigh,
                    j_list,
                    neigh_types,
                    xyz.flatten(),
                    r,
                    np.asarray(electric_field, dtype=float).flatten(),
                )
                field_derivatives_computed = True
                force_derivatives_computed = True
            else:
                self.evaluator.compute_electric_field_hessian(
                    num_nodes,
                    node_types,
                    num_neigh,
                    j_list,
                    neigh_types,
                    xyz.flatten(),
                    r,
                    np.asarray(electric_field, dtype=float).flatten(),
                )
                field_derivatives_computed = True

        if 'polarizability' in properties:
            compute_field_derivatives(include_forces='becs' in properties)
            polarizability = -np.asarray(self.evaluator.electric_field_hessian, dtype=float).reshape(3, 3)
            self.results['polarizability'] = (polarizability / volume / self._macefield_eps0).reshape(9)

        if 'becs' in properties:
            compute_field_derivatives(include_forces=True)
            num_nodes, _, _, j_list, _, xyz, _, i_list = self._mace_inputs(self.atoms)
            pair_derivatives = np.asarray(
                self.evaluator.electric_field_force_derivative,
                dtype=float,
            ).reshape(3, -1, 3)[:, :len(i_list), :]
            becs = np.zeros((len(self.atoms), 3, 3))
            for field_component in range(3):
                for cartesian in range(3):
                    becs[:, field_component, cartesian] = (
                        np.bincount(
                            j_list,
                            weights=pair_derivatives[field_component, :, cartesian],
                            minlength=num_nodes,
                        )
                        - np.bincount(
                            i_list,
                            weights=pair_derivatives[field_component, :, cartesian],
                            minlength=num_nodes,
                        )
                    )
            self.results['becs'] = becs.reshape(len(self.atoms), 9)

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        Calculator.calculate(self, atoms, properties, system_changes)

        num_nodes, node_types, num_neigh, j_list, neigh_types, xyz, r, i_list = self._mace_inputs(self.atoms)
        if self._has_native_field_coupling():
            electric_field = self._resolve_electric_field(
                require_graph=any(prop in properties for prop in self._macefield_response_properties))
            self._compute_macefield(self.atoms, electric_field)
            self._macefield_info = copy.deepcopy(getattr(self.atoms, "info", {}))
        else:
            self.evaluator.compute_node_energies_forces(
                num_nodes, node_types, num_neigh, j_list, neigh_types, xyz.flatten(), r)

        self.results['energy'] = self.results['free_energy'] = np.sum(self.evaluator.node_energies)
        self.results['energies'] = np.asarray(self.evaluator.node_energies)

        pair_forces = np.asarray(self.evaluator.node_forces).reshape((-1, 3))
        pair_forces = pair_forces[:len(i_list), :]  # currently, `evaluator.node_forces` is a container
                                                    # which can grow larger than the actual number of pairs

        # atom forces from pair_forces
        N_atoms = len(self.atoms)
        atom_forces = np.zeros((N_atoms, 3))
        atom_forces[:, 0] = np.bincount(j_list, weights=pair_forces[:, 0], minlength=N_atoms) - np.bincount(i_list, weights=pair_forces[:, 0], minlength=N_atoms)
        atom_forces[:, 1] = np.bincount(j_list, weights=pair_forces[:, 1], minlength=N_atoms) - np.bincount(i_list, weights=pair_forces[:, 1], minlength=N_atoms)
        atom_forces[:, 2] = np.bincount(j_list, weights=pair_forces[:, 2], minlength=N_atoms) - np.bincount(i_list, weights=pair_forces[:, 2], minlength=N_atoms)

        self.results['forces'] = atom_forces

        # stress from pair_forces
        self.results['stress'] = full_3x3_to_voigt_6_stress((-pair_forces.T @ xyz)  / self.atoms.get_volume())

        if (
            self._has_native_field_coupling()
            and any(prop in properties for prop in self._macefield_response_properties)
        ):
            self._calculate_macefield_responses(electric_field, properties)
