# This file was written and publicly released by Dr. Noam Bernstein as part of his
# work for the U. S. Government, and is not subject to copyright.

import pytest

import os
import json
import time
from pathlib import Path

import numpy as np

from ase.atoms import Atoms
from ase.build import bulk
from ase.calculators.calculator import PropertyNotImplementedError
from ase.stress import full_3x3_to_voigt_6_stress

try:
    from symmetrix import Symmetrix
except ModuleNotFoundError as exc:
    if "No module named 'symmetrix.symmetrix'" in str(exc):
        raise RuntimeError("Can't import symmetrix.symmetrix, probably need to run pytest in venv "
                "and install version to be tested with "
                "'(cd /path/to/repo && python3 -m pip install -e .)'") from exc
    else:
        raise

try:
    import mace
    from mace.calculators import MACECalculator
    from mace.tools.utils import get_cache_dir
    from mace.calculators.foundations_models import download_mace_mp_checkpoint
except ImportError as exc:
    mace = None

@pytest.fixture(scope="module")
def mace_foundation_model(tmp_path_factory):
    if mace is None:
        return None
    else:
        # download into a temp dir by modifying XDG_CACHE_HOME which
        # mace.calculators.foundations_models uses
        cache_dir = tmp_path_factory.mktemp("mace_cache")
        xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
        os.environ["XDG_CACHE_HOME"] = str(cache_dir)
        downloaded_model = download_mace_mp_checkpoint('small-omat-0')
        if xdg_cache_home is None:
            del os.environ["XDG_CACHE_HOME"]
        else:
            os.environ["XDG_CACHE_HOME"] = xdg_cache_home

        return str(downloaded_model)


@pytest.mark.parametrize("use_kokkos", [True, False])
def test_calc_caching(model_cache, use_kokkos):
    atoms = Atoms('O', cell=[2] * 3, pbc=[True] * 3)
    atoms *= 4
    rng = np.random.default_rng(5)
    atoms.rattle(rng=rng)

    calc = Symmetrix(model_cache["mace-mp-0b3-medium-1-8.json"], use_kokkos=use_kokkos)
    atoms.calc = calc

    t0 = time.time()
    E = atoms.get_potential_energy()
    dt_E = time.time() - t0

    t0 = time.time()
    E = atoms.get_forces()
    dt_F = time.time() - t0

    # without perturbation, forces are from cache
    assert dt_F < dt_E / 100

    atoms.positions[0, 0] += 0.1

    t0 = time.time()
    E = atoms.get_forces()
    dt_F_pert = time.time() - t0

    # with perturbation, forces have to be recomputed
    assert np.abs(dt_F_pert - dt_E) / dt_E < 0.5


@pytest.mark.parametrize("use_kokkos", [True, False])
def test_symmetrix_calc_finite_diff(model_cache, use_kokkos):
    atoms = Atoms('O', cell=[2] * 3, pbc=[True] * 3)
    atoms *= 2
    rng = np.random.default_rng(5)
    atoms.rattle(rng=rng)

    F = np.eye(3) + 0.01 * rng.normal(size=(3,3))
    atoms.set_cell(atoms.cell @ F, True)

    print("pre-converted")
    calc = Symmetrix(model_cache["mace-mp-0b3-medium-1-8.json"], use_kokkos=use_kokkos)
    do_grad_test(atoms, calc, True)


@pytest.mark.skipif(mace is None, reason="mace-torch is not available")
@pytest.mark.parametrize("use_kokkos", [True, False])
def test_mace_onthefly_calc_finite_diff(mace_foundation_model, use_kokkos):
    atoms = Atoms('O', cell=[2] * 3, pbc=[True] * 3)
    atoms *= 2
    rng = np.random.default_rng(5)
    atoms.rattle(rng=rng)

    F = np.eye(3) + 0.01 * rng.normal(size=(3,3))
    atoms.set_cell(atoms.cell @ F, True)

    print("converted on-the-fly")
    calc = Symmetrix(mace_foundation_model, species=[1, 8], use_kokkos=use_kokkos)
    do_grad_test(atoms, calc, True)


@pytest.mark.skipif(mace is None, reason="mace-torch is not available")
@pytest.mark.parametrize("use_kokkos", [True, False])
def test_symmetrix_vs_pytorch(mace_foundation_model, use_kokkos):
    atoms = Atoms('O', cell=[2] * 3, pbc=[True] * 3)
    atoms *= 2
    rng = np.random.default_rng(5)
    atoms.rattle(rng=rng)

    F = np.eye(3) + 0.01 * rng.normal(size=(3,3))
    atoms.set_cell(atoms.cell @ F, True)

    atoms_s = atoms.copy()
    atoms_p = atoms.copy()

    calc_sym = Symmetrix(mace_foundation_model, species=[1, 8], use_kokkos=use_kokkos)
    atoms_s.calc = calc_sym

    calc_torch = MACECalculator(mace_foundation_model)
    atoms_p.calc = calc_torch

    # are these in fact reasonable accuracies?
    assert np.allclose(atoms_s.get_potential_energy(), atoms_p.get_potential_energy(), atol=0.001)
    assert np.allclose(atoms_s.get_forces(), atoms_p.get_forces(), atol=0.002)
    assert np.allclose(atoms_s.get_stress(), atoms_p.get_stress(), atol=0.003)


@pytest.fixture(scope="module")
def macefield_model_path():
    model_path = Path("/home/bonan/appdir/mace-field/MACEField-MH-0-omat-dielectric.model")
    if not model_path.exists():
        pytest.skip(f"MACEField example model is not available: {model_path}")
    return model_path


@pytest.mark.skipif(mace is None, reason="mace-field is not available")
def test_macefield_model_requires_explicit_json_conversion(macefield_model_path):
    with pytest.raises(RuntimeError, match="MACEField.*Convert/extract.*Symmetrix JSON"):
        Symmetrix(
            macefield_model_path,
            species=[7, 13],
            head="mp-dielectric",
            use_kokkos=False,
            dtype="float64",
        )


@pytest.mark.skipif(mace is None, reason="mace-field is not available")
def test_macefield_native_json_ase_energy_forces_match_pytorch(macefield_model_path, tmp_path):
    from symmetrix.extract_mace_data import extract_mace_data

    json_path = tmp_path / "macefield.json"
    json_path.write_text(json.dumps(extract_mace_data(
        macefield_model_path,
        species=[7, 13],
        head="mp-dielectric",
    )))

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    atoms.info["electric_field"] = np.array([0.01, 0.0, 0.0])

    atoms_sym = atoms.copy()
    atoms_torch = atoms.copy()
    atoms_sym.calc = Symmetrix(json_path, use_kokkos=False, dtype="float64")
    atoms_torch.calc = MACECalculator(
        model_paths=[str(macefield_model_path)],
        model_type="MACEField",
        head="mp-dielectric",
        device="cpu",
        default_dtype="float64",
    )

    assert np.allclose(atoms_sym.get_potential_energy(), atoms_torch.get_potential_energy(), atol=1e-3)
    assert np.allclose(atoms_sym.get_forces(), atoms_torch.get_forces(), atol=2e-3)


@pytest.mark.skipif(mace is None, reason="mace-field is not available")
def test_macefield_native_json_ase_response_properties_match_pytorch(macefield_model_path, tmp_path):
    from symmetrix.extract_mace_data import extract_mace_data

    json_path = tmp_path / "macefield.json"
    json_path.write_text(json.dumps(extract_mace_data(
        macefield_model_path,
        species=[7, 13],
        head="mp-dielectric",
    )))

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    atoms.info["electric_field"] = np.array([0.01, -0.02, 0.03])

    atoms_sym = atoms.copy()
    atoms_torch = atoms.copy()
    atoms_sym.calc = Symmetrix(json_path, use_kokkos=False, dtype="float64")
    atoms_torch.calc = MACECalculator(
        model_paths=[str(macefield_model_path)],
        model_type="MACEField",
        head="mp-dielectric",
        device="cpu",
        default_dtype="float64",
    )

    atoms_torch.get_potential_energy()
    expected_polarization = atoms_torch.calc.results["polarization"]
    expected_becs = atoms_torch.calc.results["becs"]
    expected_polarizability = atoms_torch.calc.results["polarizability"]

    assert "polarization" in atoms_sym.calc.implemented_properties
    assert "becs" in atoms_sym.calc.implemented_properties
    assert "polarizability" in atoms_sym.calc.implemented_properties

    actual_polarization = atoms_sym.calc.get_property("polarization", atoms_sym)
    actual_becs = atoms_sym.calc.get_property("becs", atoms_sym)
    actual_polarizability = atoms_sym.calc.get_property("polarizability", atoms_sym)

    assert actual_polarization.shape == (3,)
    assert actual_becs.shape == (len(atoms), 9)
    assert actual_polarizability.shape == (9,)
    assert np.allclose(actual_polarization, expected_polarization, atol=1e-5)
    assert np.allclose(actual_becs, expected_becs, atol=5e-2)
    assert np.allclose(actual_polarizability, expected_polarizability, atol=5e-3)


@pytest.mark.skipif(mace is None, reason="mace-field is not available")
def test_macefield_native_json_response_properties_require_graph_field(macefield_model_path, tmp_path):
    from symmetrix.extract_mace_data import extract_mace_data

    json_path = tmp_path / "macefield.json"
    json_path.write_text(json.dumps(extract_mace_data(
        macefield_model_path,
        species=[7, 13],
        head="mp-dielectric",
    )))

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    atoms.info["electric_field"] = np.zeros((len(atoms), 3))
    atoms.calc = Symmetrix(json_path, use_kokkos=False, dtype="float64")

    assert np.isfinite(atoms.get_potential_energy())
    with pytest.raises(PropertyNotImplementedError, match="graph-level electric_field"):
        atoms.calc.get_property("polarization", atoms)


def test_macefield_native_json_polarization_uses_single_native_field_call(monkeypatch, tmp_path):
    class DummyFieldEvaluator:
        has_field_coupling = True
        r_cut = 3.0
        atomic_numbers = [7, 13]

        def __init__(self):
            self.calls = 0
            self.node_energies = []
            self.node_forces = []
            self.electric_field_adj = []

        def compute_node_energies_forces_field(
            self,
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz,
            r,
            electric_field,
        ):
            self.calls += 1
            self.node_energies = np.zeros(num_nodes)
            self.node_forces = np.zeros_like(np.asarray(xyz, dtype=float))
            self.electric_field_adj = np.array([1.0, 2.0, 3.0])

    evaluator = DummyFieldEvaluator()
    monkeypatch.setattr("symmetrix.symmetrix_calc.symmetrix.MACE", lambda filename: evaluator)

    json_path = tmp_path / "macefield.json"
    json_path.write_text(json.dumps({"has_field_coupling": True}))

    atoms = Atoms(
        "AlN",
        positions=[[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    atoms.info["electric_field"] = np.array([0.01, 0.0, 0.0])
    atoms.calc = Symmetrix(json_path, use_kokkos=False, dtype="float64")

    assert np.allclose(
        atoms.calc.get_property("polarization", atoms),
        -np.array([1.0, 2.0, 3.0]) / atoms.get_volume(),
    )
    assert evaluator.calls == 1


def test_macefield_native_json_polarizability_uses_native_field_hessian(monkeypatch, tmp_path):
    class DummyFieldEvaluator:
        has_field_coupling = True
        r_cut = 3.0
        atomic_numbers = [7, 13]

        def __init__(self):
            self.calls = 0
            self.hessian_calls = 0
            self.node_energies = []
            self.node_forces = []
            self.electric_field_adj = []
            self.electric_field_hessian = []

        def compute_node_energies_forces_field(
            self,
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz,
            r,
            electric_field,
        ):
            self.calls += 1
            self.node_energies = np.zeros(num_nodes)
            self.node_forces = np.zeros_like(np.asarray(xyz, dtype=float))
            self.electric_field_adj = np.array([1.0, 2.0, 3.0])

        def compute_electric_field_hessian(
            self,
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz,
            r,
            electric_field,
        ):
            self.hessian_calls += 1
            self.electric_field_hessian = np.arange(9, dtype=float).reshape(3, 3)

    evaluator = DummyFieldEvaluator()
    monkeypatch.setattr("symmetrix.symmetrix_calc.symmetrix.MACE", lambda filename: evaluator)

    json_path = tmp_path / "macefield.json"
    json_path.write_text(json.dumps({"has_field_coupling": True}))

    atoms = Atoms(
        "AlN",
        positions=[[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    atoms.info["electric_field"] = np.array([0.01, 0.0, 0.0])
    atoms.calc = Symmetrix(json_path, use_kokkos=False, dtype="float64")

    expected = (
        -np.arange(9, dtype=float).reshape(3, 3)
        / atoms.get_volume()
        / atoms.calc._macefield_eps0
    ).reshape(9)
    assert np.allclose(atoms.calc.get_property("polarizability", atoms), expected)
    assert evaluator.calls == 1
    assert evaluator.hessian_calls == 1


def test_macefield_native_json_becs_use_native_force_field_derivative(monkeypatch, tmp_path):
    class DummyFieldEvaluator:
        has_field_coupling = True
        r_cut = 3.0
        atomic_numbers = [7, 13]

        def __init__(self):
            self.calls = 0
            self.derivative_calls = 0
            self.node_energies = []
            self.node_forces = []
            self.electric_field_adj = []
            self.electric_field_force_derivative = []

        def compute_node_energies_forces_field(
            self,
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz,
            r,
            electric_field,
        ):
            self.calls += 1
            self.node_energies = np.zeros(num_nodes)
            self.node_forces = np.zeros_like(np.asarray(xyz, dtype=float))
            self.electric_field_adj = np.array([1.0, 2.0, 3.0])

        def compute_electric_field_force_derivative(
            self,
            num_nodes,
            node_types,
            num_neigh,
            neigh_indices,
            neigh_types,
            xyz,
            r,
            electric_field,
        ):
            self.derivative_calls += 1
            self.electric_field_force_derivative = np.arange(3*len(xyz), dtype=float).reshape(3, -1, 3)

    evaluator = DummyFieldEvaluator()
    monkeypatch.setattr("symmetrix.symmetrix_calc.symmetrix.MACE", lambda filename: evaluator)

    json_path = tmp_path / "macefield.json"
    json_path.write_text(json.dumps({"has_field_coupling": True}))

    atoms = Atoms(
        "AlN",
        positions=[[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
    )
    atoms.info["electric_field"] = np.array([0.01, 0.0, 0.0])
    atoms.calc = Symmetrix(json_path, use_kokkos=False, dtype="float64")

    num_nodes, _, _, j_list, _, xyz, _, i_list = atoms.calc._mace_inputs(atoms)
    pair_derivative = np.arange(3*xyz.size, dtype=float).reshape(3, -1, 3)[:, :len(i_list)]
    expected = np.zeros((num_nodes, 3, 3))
    for field_component in range(3):
        for cartesian in range(3):
            expected[:, field_component, cartesian] = (
                np.bincount(j_list, weights=pair_derivative[field_component, :, cartesian], minlength=num_nodes)
                - np.bincount(i_list, weights=pair_derivative[field_component, :, cartesian], minlength=num_nodes)
            )

    assert np.allclose(atoms.calc.get_property("becs", atoms), expected.reshape(num_nodes, 9))
    assert evaluator.calls == 1
    assert evaluator.derivative_calls == 1


@pytest.mark.skipif(mace is None, reason="mace-field is not available")
def test_macefield_native_json_uses_serial_field_path_when_kokkos_requested(macefield_model_path, tmp_path):
    from symmetrix.extract_mace_data import extract_mace_data

    json_path = tmp_path / "macefield.json"
    json_path.write_text(json.dumps(extract_mace_data(
        macefield_model_path,
        species=[7, 13],
        head="mp-dielectric",
    )))

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    atoms.info["electric_field"] = np.array([0.01, 0.0, 0.0])
    atoms.calc = Symmetrix(json_path, use_kokkos=True, dtype="float64")

    assert atoms.calc.use_kokkos is False
    assert "polarization" in atoms.calc.implemented_properties
    assert np.isfinite(atoms.get_potential_energy())
    assert atoms.calc.get_property("polarization", atoms).shape == (3,)


@pytest.mark.skipif(mace is None, reason="mace-field is not available")
def test_macefield_native_json_electric_field_changes_cached_results(macefield_model_path, tmp_path):
    from symmetrix.extract_mace_data import extract_mace_data

    json_path = tmp_path / "macefield.json"
    json_path.write_text(json.dumps(extract_mace_data(
        macefield_model_path,
        species=[7, 13],
        head="mp-dielectric",
    )))

    atoms = bulk("AlN", "wurtzite", a=3.112, c=4.982)
    atoms.calc = Symmetrix(json_path, use_kokkos=False, dtype="float64")

    atoms.info["electric_field"] = np.array([0.0, 0.0, 0.0])
    energy_zero = atoms.get_potential_energy()
    forces_zero = atoms.get_forces()

    atoms.info["electric_field"][0] = 0.01
    energy_field = atoms.get_potential_energy()
    forces_field = atoms.get_forces()

    assert not np.isclose(energy_zero, energy_field, rtol=0.0, atol=1e-8)
    assert not np.allclose(forces_zero, forces_field)


def test_plain_mace_model_uses_native_symmetrix_path(monkeypatch, tmp_path):
    class PlainTorchModel:
        pass

    class DummyEvaluator:
        r_cut = 3.0

    def fake_torch_load(*args, **kwargs):
        return PlainTorchModel()

    monkeypatch.setattr("torch.load", fake_torch_load)
    monkeypatch.setattr("symmetrix.symmetrix_calc.symmetrix.MACE", lambda filename: DummyEvaluator())

    calc = Symmetrix(tmp_path / "plain.model", use_kokkos=False)

    assert calc.evaluator.r_cut == 3.0
    assert calc.implemented_properties == ["energy", "free_energy", "energies", "forces", "stress"]


def test_unloadable_torch_model_preserves_load_error(monkeypatch, tmp_path):
    def fake_native_loader(filename):
        raise RuntimeError("not native json")

    def fake_torch_load(*args, **kwargs):
        raise ValueError("checkpoint cannot be unpickled")

    monkeypatch.setattr("symmetrix.symmetrix_calc.symmetrix.MACE", fake_native_loader)
    monkeypatch.setattr("torch.load", fake_torch_load)

    with pytest.raises(ValueError, match="checkpoint cannot be unpickled"):
        Symmetrix(tmp_path / "broken.model", use_kokkos=False)


def do_grad_test(atoms, calc, check, ax=None, label=None, plot_factor=1.0):
    atoms = atoms.copy()
    atoms.calc = calc

    F0 = atoms.get_forces()
    S0 = atoms.get_stress()
    F0_norm = np.linalg.norm(F0)
    S0_norm = np.linalg.norm(S0)
    p0 = atoms.positions.copy()
    c0 = atoms.cell.copy()
    V0 = atoms.get_volume()

    f_data = []
    passed_f = True
    F_scaling = None
    for dx_exp in np.arange(1.0, 5.1, 0.5):
        dx = 0.1 ** dx_exp

        #### forces ####
        atoms.positions = p0
        atoms.cell = c0
        F_fd = np.zeros((len(atoms), 3))
        for i_a in range(len(atoms)):
            for j_a in range(3):
                p = p0.copy()
                p[i_a, j_a] = p0[i_a, j_a] + dx
                atoms.positions = p
                E_p = atoms.get_potential_energy()
                p[i_a, j_a] = p0[i_a, j_a] - dx
                atoms.positions = p
                E_m = atoms.get_potential_energy()
                F_fd[i_a, j_a] = -(E_p - E_m) / (2 * dx)
        F_err = np.linalg.norm(F0 - F_fd)
        print(f"F {dx:6f} {F0_norm:10.6e} {F_err:10.6e} {F_err / F0_norm:10.6e} {F_err / F0_norm / (dx ** 2):10.6e}")

        f_data.append([dx, F_err])

        # force error only shows expected 2nd order scaling for dx = 0.1 ** 1, 0.1 ** 1.5
        if F_scaling is None and dx_exp >= 1.99:
            # F_err / F0_norm < F_scaling * dx ** 2
            F_scaling = 2.5 * F_err / F0_norm / (dx ** 2)
        if F_scaling is not None and dx_exp < 4.01:
            print("test forces", dx_exp, dx, F_err / F0_norm, "<?", F_scaling * dx ** 2)
            passed_f = passed_f and (F_err / F0_norm < F_scaling * dx ** 2)

    if ax is not None:
        f_data = np.asarray(f_data)
        ax.loglog(f_data[:, 0], f_data[:, 1] * plot_factor, "-", label=label)

    passed_s = True
    S_scaling = None
    for dx_exp in np.arange(1.0, 5.1, 0.5):
        dx = 0.1 ** dx_exp

        #### stress ####
        atoms.positions = p0
        atoms.cell = c0
        S_fd = np.zeros((3,3))
        for i0 in range(3):
            for i1 in range(3):
                F = np.eye(3)
                F[i0, i1] += dx / 2
                F[i1, i0] += dx / 2
                atoms.positions = p0
                atoms.cell = c0
                atoms.set_cell(c0 @ F, True)
                E_p = atoms.get_potential_energy()

                F = np.eye(3)
                F[i0, i1] -= dx / 2
                F[i1, i0] -= dx / 2
                atoms.positions = p0
                atoms.cell = c0
                atoms.set_cell(c0 @ F, True)
                E_m = atoms.get_potential_energy()

                S_fd[i0, i1] = (E_p - E_m) / (2 * dx) / V0

        S_err = np.linalg.norm(S0 - full_3x3_to_voigt_6_stress(S_fd))
        print(f"S {dx:6f} {S0_norm:10.6e} {S_err:10.6e} {S_err / S0_norm:10.6e} {S_err / S0_norm / dx ** 2:10.6e}")

        if S_scaling is None and dx_exp >= 1.99:
            # S_err / S0_norm < S_scaling * dx ** 2
            S_scaling = 1.5 * S_err / S0_norm / (dx ** 2)
        if S_scaling is not None and dx_exp < 4.01:
            print("test stress", dx_exp, dx, S_err / S0_norm, "<?", S_scaling * dx ** 2)
            passed_f = passed_f and (S_err / S0_norm < S_scaling * dx ** 2)

    if check:
        assert passed_f and passed_s
