import json
from pathlib import Path

import numpy as np
import pytest

try:
    from lammps import lammps
except ImportError as exc:
    pytest.skip(f"LAMMPS Python module is not available: {exc}", allow_module_level=True)

try:
    from ase.build import bulk
    from ase.neighborlist import neighbor_list
    from symmetrix import symmetrix as native_symmetrix
    from symmetrix.extract_mace_data import extract_mace_data
except ImportError as exc:
    pytest.skip(f"MACEField test dependencies are not available: {exc}", allow_module_level=True)


MODEL_PATH = Path("/home/bonan/appdir/mace-field/MACEField-MH-0-omat-dielectric.model")


@pytest.fixture(scope="module")
def macefield_json_path(tmp_path_factory):
    if not MODEL_PATH.exists():
        pytest.skip(f"MACEField example model is not available: {MODEL_PATH}")

    output_path = tmp_path_factory.mktemp("lammps-macefield-json") / "macefield.json"
    data = extract_mace_data(
        MODEL_PATH,
        species=[7, 13],
        head="mp-dielectric",
    )
    output_path.write_text(json.dumps(data))
    return output_path


@pytest.mark.parametrize(
    "pair_style",
    [
        "symmetrix/mace electric_field 0.01 0.0 0.0 no_domain_decomposition",
        "symmetrix/mace electric_field 0.01 0.0 0.0 no_mpi_message_passing",
    ],
)
def test_lammps_kokkos_macefield_energy_forces_match_native(macefield_json_path, pair_style):
    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    atoms.set_pbc(False)
    atoms.center(vacuum=6.0)
    electric_field = np.array([0.01, 0.0, 0.0], dtype=np.float64)

    native = native_symmetrix.MACE(str(macefield_json_path))
    atomic_numbers = atoms.get_atomic_numbers().tolist()
    mace_atomic_numbers = native.atomic_numbers
    i_list, j_list, r, xyz = neighbor_list("ijdD", atoms, native.r_cut)
    num_nodes = len(atoms)
    node_types = np.asarray([mace_atomic_numbers.index(atomic_numbers[i]) for i in range(num_nodes)], dtype=np.int32)
    num_neigh = np.asarray(np.bincount(j_list, minlength=num_nodes), dtype=np.int32)
    neigh_types = np.asarray([mace_atomic_numbers.index(atomic_numbers[j]) for j in j_list], dtype=np.int32)
    neigh_indices = np.asarray(j_list, dtype=np.int32)
    native.compute_node_energies_forces_field(
        num_nodes,
        node_types,
        num_neigh,
        neigh_indices,
        neigh_types,
        xyz.reshape(-1),
        r,
        electric_field,
    )

    expected_energy = np.sum(native.node_energies)
    pair_forces = np.asarray(native.node_forces).reshape((-1, 3))[: len(i_list), :]
    expected_forces = np.zeros((num_nodes, 3))
    for component in range(3):
        expected_forces[:, component] = (
            np.bincount(j_list, weights=pair_forces[:, component], minlength=num_nodes)
            - np.bincount(i_list, weights=pair_forces[:, component], minlength=num_nodes)
        )

    cell_lengths = atoms.cell.lengths()
    create_atoms = "\n".join(
        "            create_atoms    {} single {:.12f} {:.12f} {:.12f} units box".format(
            1 if symbol == "Al" else 2,
            *position,
        )
        for symbol, position in zip(atoms.get_chemical_symbols(), atoms.positions)
    )

    lmp = lammps(cmdargs=["-screen", "none", "-k", "on", "-sf", "kk"])
    try:
        lmp.commands_string(
            f"""
            clear
            units           metal
            boundary        f f f
            atom_style      atomic
            atom_modify     map yes sort 0 0
            newton          on

            region          box block 0.0 {cell_lengths[0]} 0.0 {cell_lengths[1]} 0.0 {cell_lengths[2]}
            create_box      2 box
{create_atoms}
            mass            1 26.9815385
            mass            2 14.0067

            pair_style      {pair_style}
            pair_coeff      * * {macefield_json_path} Al N

            run 0
            """
        )

        actual_energy = lmp.get_thermo("pe")
        actual_forces = lmp.numpy.extract_atom("f", nelem=num_nodes, dim=3).copy()
    finally:
        lmp.close()

    assert actual_energy == pytest.approx(expected_energy, abs=1e-6)
    assert np.allclose(actual_forces, expected_forces, atol=1e-5, rtol=1e-5)
