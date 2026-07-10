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


    def __init__(self, model_file, dtype="float64", use_kokkos=True, **kwargs):
        Calculator.__init__(self, **kwargs)
        if dtype not in ["float32", "float64"]:
            raise ValueError(f"Unsupported dtype '{dtype}'. Supported dtypes are 'float64' and 'float32'.")
        self._macefield_calculator = None
        self._macefield_info = None
        self._electric_field = kwargs.get("electric_field", None)

        if use_kokkos and not hasattr(symmetrix, "MACEKokkos"):
            if not str(model_file).endswith(".json"):
                self._macefield_calculator = self._make_macefield_calculator(
                    model_file,
                    dtype=dtype,
                    **kwargs,
                )
                if self._macefield_calculator is not None:
                    self.implemented_properties = list(self._macefield_calculator.implemented_properties)
                    self.cutoff = self._macefield_calculator.r_max
                    return
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
            self._macefield_calculator = self._make_macefield_calculator(
                model_file,
                dtype=dtype,
                **kwargs,
            )
            if self._macefield_calculator is not None:
                self.implemented_properties = list(self._macefield_calculator.implemented_properties)
                self.cutoff = self._macefield_calculator.r_max
                return

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

    def _make_macefield_calculator(self, model_file, dtype, **kwargs):
        try:
            import torch
            from mace.calculators import MACECalculator
        except ImportError:
            return None

        model = torch.load(
            model_file,
            map_location=torch.device("cpu"),
            weights_only=False,
        )

        is_macefield = (
            type(model).__name__ == "MACEField"
            or (hasattr(model, "field_feats") and hasattr(model, "field_linear"))
        )
        if not is_macefield:
            return None

        calculator_kwargs = {
            "model_paths": [str(model_file)],
            "model_type": "MACEField",
            "device": kwargs.get("device", "cpu"),
            "default_dtype": dtype,
            "head": kwargs.get("head", None),
            "electric_field": kwargs.get("electric_field", None),
        }
        if kwargs.get("compute_atomic_stresses", False):
            calculator_kwargs["compute_atomic_stresses"] = True

        return MACECalculator(**calculator_kwargs)

    def check_state(self, atoms, tol=1e-15):
        state = super().check_state(atoms, tol=tol)
        if (
            (self._macefield_calculator is not None or self._has_native_field_coupling())
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
            self._macefield_calculator is None
            and hasattr(self, "evaluator")
            and getattr(self.evaluator, "has_field_coupling", False)
        )

    def _resolve_electric_field(self):
        if self._electric_field is not None:
            field = self._electric_field
        elif "electric_field" in self.atoms.info:
            field = self.atoms.info["electric_field"]
        elif "REF_electric_field" in self.atoms.info:
            field = self.atoms.info["REF_electric_field"]
        else:
            field = np.zeros(3)

        field = np.asarray(field, dtype=float)
        if field.shape == (3,):
            field = np.tile(field, (len(self.atoms), 1))
        elif field.shape != (len(self.atoms), 3):
            raise ValueError("electric_field must have shape (3,) or (natoms, 3).")
        return field

    def calculate(self, atoms=None, properties=['energy'], system_changes=all_changes):
        if self._macefield_calculator is not None:
            Calculator.calculate(self, atoms, properties, system_changes)
            self._macefield_calculator.calculate(self.atoms, properties, system_changes)
            self.results = {
                key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
                for key, value in self._macefield_calculator.results.items()
            }
            self._macefield_info = copy.deepcopy(getattr(self.atoms, "info", {}))
            return

        Calculator.calculate(self, atoms, properties, system_changes)

        ase_atomic_numbers = self.atoms.get_atomic_numbers().tolist()
        mace_atomic_numbers = self.evaluator.atomic_numbers
        i_list, j_list, r, xyz = neighbor_list('ijdD', self.atoms, self.cutoff)
        num_nodes = np.max(i_list) + 1
        node_types = [mace_atomic_numbers.index(ase_atomic_numbers[i]) for i in range(num_nodes)]
        num_neigh = np.bincount(j_list, minlength=num_nodes)
        neigh_types = [mace_atomic_numbers.index(ase_atomic_numbers[j]) for j in j_list]
        if self._has_native_field_coupling():
            self.evaluator.compute_node_energies_forces_field(
                num_nodes,
                node_types,
                num_neigh,
                j_list,
                neigh_types,
                xyz.flatten(),
                r,
                self._resolve_electric_field().flatten(),
            )
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
