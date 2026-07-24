import copy
import json

import numpy as np
import pytest
from ase import Atoms

from symmetrix import Symmetrix
from symmetrix import symmetrix as native_symmetrix
from model_downloads import MODEL_URLS, cached_model_path

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
def legacy_standard_model_path():
    try:
        return cached_model_path(
            "MACE-OFF23_small-1-8.json",
            MODEL_URLS["MACE-OFF23_small-1-8.json"],
        )
    except RuntimeError as exc:
        pytest.skip(str(exc))


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


@pytest.mark.parametrize(
    "use_kokkos,dtype,atol",
    [
        (False, "float64", 2e-11),
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
        outputs[mode] = (atoms.get_potential_energy(), atoms.get_forces())
        storage[mode] = (
            calculator.evaluator.R0_storage_size,
            calculator.evaluator.R1_storage_size,
        )
        if mode == "legacy":
            legacy_evaluator = calculator.evaluator

    reference_energy, reference_forces = outputs["legacy"]
    for mode in ("r1", "all"):
        energy, forces = outputs[mode]
        assert energy == pytest.approx(reference_energy, rel=0.0, abs=atol)
        np.testing.assert_allclose(forces, reference_forces, rtol=0.0, atol=atol)

    assert storage["legacy"][0] > 0
    assert storage["legacy"][1] > 0
    assert storage["r1"][0] > 0
    assert storage["r1"][1] == 0
    assert storage["all"] == (0, 0)

    legacy_evaluator.set_streamed_edges("all")
    assert legacy_evaluator.R0_storage_size == 0
    assert legacy_evaluator.R1_storage_size == 0


@pytest.mark.parametrize("evaluator_name", ["MACE", "MACEKokkos"])
def test_streamed_modes_reject_field_coupled_models(
    streamed_model_paths,
    evaluator_name,
):
    _, field_path = streamed_model_paths
    evaluator = getattr(native_symmetrix, evaluator_name)(str(field_path))
    assert not evaluator.supports_streamed_edges
    with pytest.raises(ValueError, match="ordinary format-v2 compact MACE"):
        evaluator.set_streamed_edges("r1")


def test_streamed_modes_reject_legacy_pair_spline_models(legacy_standard_model_path):
    evaluator = native_symmetrix.MACE(str(legacy_standard_model_path))
    assert not evaluator.supports_streamed_edges
    with pytest.raises(ValueError, match="ordinary format-v2 compact MACE"):
        evaluator.set_streamed_edges("all")


def test_calculator_rejects_unknown_streamed_mode(streamed_model_paths):
    standard_path, _ = streamed_model_paths
    with pytest.raises(ValueError, match="legacy.*r1.*all"):
        Symmetrix(standard_path, streamed_edges="tiles")
