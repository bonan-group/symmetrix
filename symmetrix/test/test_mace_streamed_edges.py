import copy
import json

import numpy as np
import pytest
from ase import Atoms

from symmetrix import Symmetrix
from symmetrix import symmetrix as native_symmetrix

try:
    from symmetrix.extract_mace_data import extract_mace_data
except ImportError as exc:
    pytest.skip(
        f"Compact MACE extraction dependencies are not available: {exc}",
        allow_module_level=True,
    )


@pytest.fixture(scope="module")
def streamed_model_paths(tmp_path_factory, macefield_model_path):
    output_dir = tmp_path_factory.mktemp("mace-streamed-edges")
    field_data = extract_mace_data(
        macefield_model_path,
        species=[7, 13],
        head="mp-dielectric",
        num_spline_points=16,
    )
    field_path = output_dir / "compact-field.json"
    field_path.write_text(json.dumps(field_data, separators=(",", ":")))

    standard_data = copy.deepcopy(field_data)
    standard_data["model_type"] = "MACE"
    standard_data["has_field_coupling"] = False
    standard_data.pop("field_couplings", None)
    standard_path = output_dir / "compact-standard.json"
    standard_path.write_text(json.dumps(standard_data, separators=(",", ":")))
    return standard_path, field_path


@pytest.fixture(scope="module")
def legacy_standard_model_path(tmp_path_factory, macefield_model_path):
    legacy_data = extract_mace_data(
        macefield_model_path,
        species=[7, 13],
        head="mp-dielectric",
        num_spline_points=16,
        radial_format="pair-splines",
    )
    legacy_data["model_type"] = "MACE"
    legacy_data["has_field_coupling"] = False
    legacy_data.pop("field_couplings", None)
    legacy_path = tmp_path_factory.mktemp("mace-legacy-edges") / "legacy.json"
    legacy_path.write_text(json.dumps(legacy_data))
    return legacy_path


def _small_structure():
    return Atoms(
        ["Al", "N", "Al", "N"],
        positions=[
            [0.0, 0.0, 0.0],
            [1.8, 0.2, 0.1],
            [0.3, 2.0, 0.4],
            [1.9, 1.8, 0.7],
        ],
        cell=[8.0, 8.0, 8.0],
        pbc=False,
    )


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_compact_v2_defaults_to_fully_streamed_execution(
    streamed_model_paths,
    use_kokkos,
):
    standard_path, _ = streamed_model_paths
    calculator = Symmetrix(standard_path, use_kokkos=use_kokkos)

    assert calculator.streamed_edges == "all"
    assert calculator.evaluator.supports_streamed_edges
    assert calculator.evaluator.streamed_edges_mode == "all"


@pytest.mark.parametrize(
    "use_kokkos,dtype,atol",
    [
        (False, "float64", 2e-11),
        (False, "float32", 2e-5),
        (True, "float64", 2e-11),
        (True, "float32", 2e-5),
    ],
)
def test_streamed_modes_match_legacy_and_release_radial_storage(
    streamed_model_paths,
    use_kokkos,
    dtype,
    atol,
):
    standard_path, _ = streamed_model_paths
    outputs = {}
    storage = {}
    for mode in ("legacy", "r1", "all"):
        atoms = _small_structure()
        calculator = Symmetrix(
            standard_path,
            use_kokkos=use_kokkos,
            dtype=dtype,
            streamed_edges=mode,
        )
        assert calculator.evaluator.supports_streamed_edges
        assert calculator.evaluator.streamed_edges_mode == mode
        atoms.calc = calculator
        outputs[mode] = (
            atoms.get_potential_energy(),
            atoms.get_forces(),
            atoms.get_stress(),
        )
        storage[mode] = (
            calculator.evaluator.R0_storage_size,
            calculator.evaluator.R1_storage_size,
        )
        if mode == "legacy":
            legacy_evaluator = calculator.evaluator

    reference_energy, reference_forces, reference_stress = outputs["legacy"]
    for mode in ("r1", "all"):
        energy, forces, stress = outputs[mode]
        assert energy == pytest.approx(reference_energy, rel=0.0, abs=atol)
        np.testing.assert_allclose(forces, reference_forces, rtol=0.0, atol=atol)
        np.testing.assert_allclose(stress, reference_stress, rtol=0.0, atol=atol)

    assert storage["legacy"][0] > 0
    assert storage["legacy"][1] > 0
    assert storage["r1"][0] > 0
    assert storage["r1"][1] == 0
    assert storage["all"] == (0, 0)

    legacy_evaluator.set_streamed_edges("all")
    assert legacy_evaluator.R0_storage_size == 0
    assert legacy_evaluator.R1_storage_size == 0


@pytest.mark.parametrize(
    "use_kokkos,dtype,first_order_atol,response_atol",
    [
        (False, "float64", 2e-9, 2e-6),
        (False, "float32", 5e-5, 5e-4),
        (True, "float64", 2e-9, 2e-6),
        (True, "float32", 5e-5, 5e-4),
    ],
    ids=[
        "native-float64",
        "native-float32",
        "kokkos-float64",
        "kokkos-float32",
    ],
)
def test_field_streamed_modes_match_legacy(
    streamed_model_paths,
    use_kokkos,
    dtype,
    first_order_atol,
    response_atol,
):
    _, field_path = streamed_model_paths
    electric_field = np.array([0.01, -0.02, 0.03])
    outputs = {}
    storage = {}
    for mode in ("legacy", "r1", "all"):
        atoms = _small_structure()
        calculator = Symmetrix(
            field_path,
            use_kokkos=use_kokkos,
            dtype=dtype,
            streamed_edges=mode,
            electric_field=electric_field,
        )
        assert calculator.evaluator.supports_streamed_edges
        atoms.calc = calculator
        calculator.calculate(
            atoms,
            properties=[
                "energy", "forces", "stress", "polarization", "polarizability",
                "becs",
            ],
        )
        results = calculator.results
        field_adjoint = np.asarray(
            calculator.evaluator.electric_field_adj).copy()
        field_hessian = np.asarray(
            calculator.evaluator.electric_field_hessian).copy()
        field_force_derivative = np.asarray(
            calculator.evaluator.electric_field_force_derivative).copy()
        outputs[mode] = (
            results["energy"],
            results["forces"],
            results["stress"],
            results["polarization"],
            results["polarizability"],
            results["becs"],
            field_adjoint,
            field_hessian,
            field_force_derivative,
        )
        storage[mode] = (
            calculator.evaluator.R0_storage_size,
            calculator.evaluator.R1_storage_size,
        )

    (
        reference_energy,
        reference_forces,
        reference_stress,
        reference_polarization,
        reference_polarizability,
        reference_becs,
        reference_field_adj,
        reference_field_hessian,
        reference_field_force_derivative,
    ) = outputs["legacy"]
    for mode in ("r1", "all"):
        (
            energy,
            forces,
            stress,
            polarization,
            polarizability,
            becs,
            field_adj,
            field_hessian,
            field_force_derivative,
        ) = outputs[mode]
        assert energy == pytest.approx(
            reference_energy, rel=0.0, abs=first_order_atol)
        np.testing.assert_allclose(
            forces, reference_forces, rtol=0.0, atol=first_order_atol)
        np.testing.assert_allclose(
            stress, reference_stress, rtol=0.0, atol=first_order_atol)
        np.testing.assert_allclose(
            polarization, reference_polarization,
            rtol=0.0, atol=first_order_atol)
        np.testing.assert_allclose(
            polarizability, reference_polarizability,
            rtol=0.0, atol=response_atol)
        np.testing.assert_allclose(
            becs, reference_becs, rtol=0.0, atol=response_atol)
        np.testing.assert_allclose(
            field_adj, reference_field_adj,
            rtol=0.0, atol=first_order_atol)
        np.testing.assert_allclose(
            field_hessian, reference_field_hessian,
            rtol=0.0, atol=response_atol)
        np.testing.assert_allclose(
            field_force_derivative,
            reference_field_force_derivative,
            rtol=0.0,
            atol=response_atol,
        )

    assert storage["legacy"][0] > 0
    assert storage["legacy"][1] > 0
    assert storage["r1"][0] > 0
    assert storage["r1"][1] == 0
    assert storage["all"] == (0, 0)


def test_native_float32_matches_native_float64(streamed_model_paths):
    standard_path, field_path = streamed_model_paths
    atoms = _small_structure()

    def evaluate(model_path, dtype, electric_field=None):
        calculator = Symmetrix(
            model_path,
            use_kokkos=False,
            dtype=dtype,
            streamed_edges="all",
            electric_field=electric_field,
        )
        assert isinstance(
            calculator.evaluator,
            native_symmetrix.MACEFloat if dtype == "float32" else native_symmetrix.MACE,
        )
        assert calculator.evaluator.scalar_size_bytes == (
            4 if dtype == "float32" else 8)
        calculator.calculate(
            atoms,
            properties=(
                ["energy", "forces"]
                if electric_field is None
                else ["energy", "forces", "polarization", "polarizability", "becs"]
            ),
        )
        return calculator.results

    standard64 = evaluate(standard_path, "float64")
    standard32 = evaluate(standard_path, "float32")
    assert standard32["energy"] == pytest.approx(
        standard64["energy"], rel=0.0, abs=2e-4)
    np.testing.assert_allclose(
        standard32["forces"], standard64["forces"], rtol=0.0, atol=2e-4)

    electric_field = np.array([0.01, -0.02, 0.03])
    field64 = evaluate(field_path, "float64", electric_field)
    field32 = evaluate(field_path, "float32", electric_field)
    assert field32["energy"] == pytest.approx(
        field64["energy"], rel=0.0, abs=2e-4)
    for property_name in ("forces", "polarization"):
        np.testing.assert_allclose(
            field32[property_name], field64[property_name], rtol=0.0, atol=2e-4)
    for property_name in ("polarizability", "becs"):
        np.testing.assert_allclose(
            field32[property_name], field64[property_name], rtol=0.0, atol=2e-3)


def test_kokkos_rejects_out_of_range_phi1_hidden_degree(
    streamed_model_paths,
    tmp_path,
):
    _, field_path = streamed_model_paths
    model = json.loads(field_path.read_text())
    model["Phi1_l2"][0] = model["L_max"] + 1
    invalid_path = tmp_path / "invalid-phi1-l2.json"
    invalid_path.write_text(json.dumps(model))

    with pytest.raises(RuntimeError, match="out-of-range hidden degree"):
        native_symmetrix.MACEKokkos(str(invalid_path))


def test_native_float32_rejects_out_of_range_model_values(
    streamed_model_paths,
    tmp_path,
):
    standard_path, _ = streamed_model_paths
    model = json.loads(standard_path.read_text())
    model["H0_weights"][0] = 1e100
    invalid_path = tmp_path / "float32-overflow.json"
    invalid_path.write_text(json.dumps(model))

    with pytest.raises(ValueError, match="outside the requested precision range"):
        native_symmetrix.MACEFloat(str(invalid_path))


@pytest.mark.parametrize(
    "use_kokkos,dtype",
    [
        (False, "float64"),
        (False, "float32"),
        (True, "float64"),
        (True, "float32"),
    ],
)
def test_legacy_pair_spline_models_warn_and_default_to_legacy(
    legacy_standard_model_path,
    use_kokkos,
    dtype,
):
    with pytest.warns(UserWarning, match="format-v1.*streamed_edges='legacy'"):
        calculator = Symmetrix(
            legacy_standard_model_path,
            use_kokkos=use_kokkos,
            dtype=dtype,
        )

    evaluator = calculator.evaluator
    assert calculator.streamed_edges == "legacy"
    assert not evaluator.supports_streamed_edges
    assert evaluator.streamed_edges_mode == "legacy"
    with pytest.raises(ValueError, match="format-v2 compact MACE or MACEField"):
        evaluator.set_streamed_edges("all")


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_legacy_pair_spline_float32_matches_float64(
    legacy_standard_model_path,
    use_kokkos,
):
    outputs = {}
    for dtype in ("float64", "float32"):
        atoms = _small_structure()
        with pytest.warns(UserWarning, match="format-v1.*streamed_edges='legacy'"):
            atoms.calc = Symmetrix(
                legacy_standard_model_path,
                use_kokkos=use_kokkos,
                dtype=dtype,
            )
        assert atoms.calc.evaluator.scalar_size_bytes == (
            8 if dtype == "float64" else 4)
        outputs[dtype] = (atoms.get_potential_energy(), atoms.get_forces())

    assert outputs["float32"][0] == pytest.approx(
        outputs["float64"][0], rel=0.0, abs=2e-4)
    np.testing.assert_allclose(
        outputs["float32"][1], outputs["float64"][1], rtol=0.0, atol=2e-4)


def test_kokkos_streamed_path_offsets_match_model_layout(streamed_model_paths):
    standard_path, _ = streamed_model_paths
    calculator = Symmetrix(standard_path, use_kokkos=True, streamed_edges="all")
    model = json.loads(standard_path.read_text())

    expected = [0]
    for l1, l2 in zip(model["Phi1_l1"], model["Phi1_l2"]):
        expected.append(expected[-1] + (2 * l1 + 1) * (2 * l2 + 1))

    assert calculator.evaluator.Phi1_path_row_offsets == expected
    assert native_symmetrix._kokkos_default_execution_space()


def test_calculator_rejects_unknown_streamed_mode(streamed_model_paths):
    standard_path, _ = streamed_model_paths
    with pytest.raises(ValueError, match="legacy.*r1.*all"):
        Symmetrix(standard_path, streamed_edges="tiles")
