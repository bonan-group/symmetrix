import gc
import json
import os
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk

from symmetrix import Symmetrix
from symmetrix import symmetrix as native_symmetrix


try:
    import torch
    from mace.calculators import MACECalculator
    from mace.tools.scripts_utils import remove_pt_head
    from symmetrix.extract_mace_data import extract_mace_data
except ImportError as exc:
    torch = None
    MACECalculator = None
    remove_pt_head = None
    extract_mace_data = None
    mace_import_error = exc
else:
    mace_import_error = None


MH1_HEADS = (
    "matpes_r2scan",
    "mp_pbe_refit_add",
    "spice_wB97M",
    "oc20_usemppbe",
    "omol",
    "omat_pbe",
)


def _mh1_streamed_edge_block_size(use_kokkos):
    if (
        use_kokkos
        and native_symmetrix._kokkos_default_execution_space() == "Cuda"
    ):
        return 16384
    return 1024


def _mh1_model_path():
    if mace_import_error is not None:
        pytest.skip(f"mace-torch is not available: {mace_import_error}")
    path = os.environ.get("SYMMETRIX_MH1_MODEL")
    if not path:
        pytest.skip("set SYMMETRIX_MH1_MODEL to run the MACE-MH-1 integration tests")
    path = Path(path)
    if not path.is_file():
        pytest.skip(f"MACE-MH-1 checkpoint does not exist: {path}")
    return path


@pytest.fixture(scope="module")
def mh1_si_artifact(tmp_path_factory):
    data = extract_mace_data(
        _mh1_model_path(),
        species=[14],
        head="matpes_r2scan",
    )
    path = tmp_path_factory.mktemp("mh1") / "mh1-si.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    return data, path


@pytest.fixture(scope="module")
def mh1_h_si_artifact(tmp_path_factory):
    data = extract_mace_data(
        _mh1_model_path(),
        species=[14, 1],
        head="matpes_r2scan",
    )
    path = tmp_path_factory.mktemp("mh1-h-si") / "mh1-h-si.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    return data, path


def test_mh1_rejects_malformed_nonlinear_json(tmp_path):
    path = tmp_path / "malformed-nonlinear.json"
    path.write_text(json.dumps({
        "symmetrix_format_version": 3,
        "model_type": "MACE_Nonlinear",
    }))
    with pytest.raises((RuntimeError, ValueError), match="node_embedding"):
        native_symmetrix.MACENonlinear(str(path))
    if hasattr(native_symmetrix, "MACENonlinearKokkos"):
        with pytest.raises((RuntimeError, ValueError), match="node_embedding"):
            native_symmetrix.MACENonlinearKokkos(str(path))


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_mh1_rejects_invalid_species_mapping_before_evaluation(tmp_path, use_kokkos):
    linear = {
        "irreps_in": "1x0e",
        "irreps_out": "1x0e",
        "instructions": [{
            "i_in": 0,
            "i_out": 0,
            "path_weight": 1.0,
            "path_shape": [1, 1],
        }],
        "weight": {"shape": [1], "values": [1.0]},
        "bias": {"shape": [0], "values": []},
        "output_mask": {"shape": [1], "values": [1.0]},
    }
    path = tmp_path / "invalid-mapping.json"
    path.write_text(json.dumps({
        "symmetrix_format_version": 3,
        "model_type": "MACE_Nonlinear",
        "node_embedding": linear,
        "atomic_numbers": [1],
        "model_atomic_numbers": [1],
        "model_indices": [2],
        "num_elements": 1,
    }))
    if use_kokkos:
        if not hasattr(native_symmetrix, "MACENonlinearKokkos"):
            pytest.skip("Symmetrix was built without Kokkos support")
        if not native_symmetrix._kokkos_is_initialized():
            native_symmetrix._init_kokkos()
        evaluator = native_symmetrix.MACENonlinearKokkos
    else:
        evaluator = native_symmetrix.MACENonlinear
    with pytest.raises(ValueError, match="model_indices"):
        evaluator(str(path))


def test_mh1_subset_schema_preserves_dynamic_architecture(mh1_si_artifact):
    data, _ = mh1_si_artifact
    assert data["symmetrix_format_version"] == 3
    assert data["model_type"] == "MACE_Nonlinear"
    assert data["head"] == "matpes_r2scan"
    assert data["atomic_numbers"] == [14]
    assert data["model_indices"] == [13]
    assert len(data["model_atomic_numbers"]) == 89
    assert len(data["interactions"]) == data["num_interactions"]
    assert len(data["products"]) == data["num_interactions"]
    assert len(data["readouts"]) == data["num_interactions"]
    assert all(
        interaction["class"] == "RealAgnosticResidualNonLinearInteractionBlock"
        for interaction in data["interactions"]
    )
    assert all(interaction["conv_tp_weights"]["layers"] for interaction in data["interactions"])
    assert all(interaction["density_fn"]["layers"] for interaction in data["interactions"])
    hidden_irreps = data["readouts"][-1]["linear_1"]["irreps_out"]
    assert hidden_irreps == data["readouts"][-1]["linear_2"]["irreps_in"]
    assert hidden_irreps.endswith("x0e")
    assert max(
        contraction["correlation"]
        for product in data["products"]
        for contraction in product["symmetric_contractions"]["contractions"]
    ) >= 3


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_mh1_streamed_edge_modes_agree(mh1_si_artifact, use_kokkos):
    _, model_path = mh1_si_artifact
    if use_kokkos and not hasattr(native_symmetrix, "MACENonlinearKokkos"):
        pytest.skip("Symmetrix was built without Kokkos support")
    atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 1, 1))
    properties = ["energy", "energies", "forces", "stress"]
    results = {}
    workspace_rows = {}
    workspace_bytes = {}
    edge_count = None
    for mode in ("legacy", "r1", "all"):
        calculator = Symmetrix(
            model_path,
            use_kokkos=use_kokkos,
            dtype="float64",
            streamed_edges=mode,
        )
        assert calculator.evaluator.uses_mh1_fast_path
        assert calculator.evaluator.supports_streamed_edges
        assert calculator.evaluator.streamed_edges_mode == mode
        calculator.calculate(atoms.copy(), properties=properties)
        if edge_count is None:
            edge_count = len(calculator._mace_inputs(atoms)[6])
        workspace_rows[mode] = calculator.evaluator.edge_workspace_rows
        if hasattr(calculator.evaluator, "edge_workspace_bytes"):
            workspace_bytes[mode] = calculator.evaluator.edge_workspace_bytes
        results[mode] = {
            name: np.array(calculator.results[name], copy=True)
            for name in properties
        }
    for mode in ("r1", "all"):
        for name in properties:
            assert np.allclose(results[mode][name], results["legacy"][name], atol=2e-12)
    assert workspace_rows["legacy"] == edge_count
    assert workspace_rows["r1"] == edge_count
    assert workspace_rows["all"] <= min(
        edge_count, _mh1_streamed_edge_block_size(use_kokkos)
    )
    if workspace_bytes:
        assert workspace_bytes["all"] < workspace_bytes["legacy"]

    automatic = Symmetrix(model_path, use_kokkos=use_kokkos, dtype="float64")
    assert automatic.streamed_edges == "all"


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_mh1_kokkos_mode_switch_releases_full_edge_workspaces(
    mh1_si_artifact, dtype
):
    evaluator_name = (
        "MACENonlinearKokkos" if dtype == "float64" else "MACENonlinearKokkosFloat"
    )
    if not hasattr(native_symmetrix, evaluator_name):
        pytest.skip("Symmetrix was built without Kokkos support")
    _, model_path = mh1_si_artifact
    atoms = bulk("Si", "diamond", a=5.43).repeat((3, 3, 3))
    calculator = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="legacy",
    )
    calculator.calculate(atoms, properties=["energy", "forces"])
    edge_count = len(calculator._mace_inputs(atoms)[6])
    legacy_bytes = calculator.evaluator.edge_workspace_bytes
    assert calculator.evaluator.edge_workspace_rows == edge_count

    calculator.evaluator.set_streamed_edges("all")
    calculator.calculate(atoms, properties=["energy", "forces"])
    assert calculator.evaluator.edge_workspace_rows <= min(
        edge_count, _mh1_streamed_edge_block_size(True)
    )
    assert calculator.evaluator.edge_workspace_bytes < legacy_bytes


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_mh1_kokkos_mode_switch_fences_pending_evaluation(
    mh1_si_artifact, dtype
):
    evaluator_name = (
        "MACENonlinearKokkos" if dtype == "float64" else "MACENonlinearKokkosFloat"
    )
    if not hasattr(native_symmetrix, evaluator_name):
        pytest.skip("Symmetrix was built without the requested Kokkos precision")
    _, model_path = mh1_si_artifact
    atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 1, 1))
    calculator = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype=dtype,
        streamed_edges="legacy",
    )
    inputs = calculator._mace_inputs(atoms)
    (
        num_nodes,
        node_types,
        num_neigh,
        neighbors,
        neigh_types,
        xyz,
        distances,
        _,
    ) = inputs

    calculator.evaluator.compute_node_energies_forces(
        num_nodes,
        node_types,
        num_neigh,
        neighbors,
        neigh_types,
        xyz.flatten(),
        distances,
    )
    calculator.evaluator.set_streamed_edges("all")
    legacy_energies = np.asarray(calculator.evaluator.node_energies)
    legacy_forces = np.asarray(calculator.evaluator.node_forces)

    calculator.evaluator.compute_node_energies_forces(
        num_nodes,
        node_types,
        num_neigh,
        neighbors,
        neigh_types,
        xyz.flatten(),
        distances,
    )
    calculator.evaluator.set_streamed_edges("legacy")
    all_energies = np.asarray(calculator.evaluator.node_energies)
    all_forces = np.asarray(calculator.evaluator.node_forces)
    tolerance = 2e-12 if dtype == "float64" else 2e-5
    np.testing.assert_allclose(
        all_energies, legacy_energies, rtol=0.0, atol=tolerance
    )
    np.testing.assert_allclose(all_forces, legacy_forces, rtol=0.0, atol=tolerance)


def test_mh1_cuda_streamed_fast_path_matches_native(mh1_si_artifact):
    if not hasattr(native_symmetrix, "MACENonlinearKokkos"):
        pytest.skip("Symmetrix was built without Kokkos support")
    if native_symmetrix._kokkos_default_execution_space() != "Cuda":
        pytest.skip("CUDA-only nonlinear fallback regression")
    _, model_path = mh1_si_artifact
    atoms = Atoms(
        "Si3",
        positions=[[0.0, 0.0, 0.0], [2.2, 0.1, 0.0], [0.4, 2.1, 0.3]],
        cell=[8.0, 8.0, 8.0],
        pbc=False,
    )
    properties = ["energy", "energies", "forces", "stress"]
    reference = Symmetrix(
        model_path,
        use_kokkos=False,
        dtype="float64",
        streamed_edges="legacy",
    )
    cuda = Symmetrix(model_path, use_kokkos=True, dtype="float64")
    assert cuda.evaluator.uses_mh1_fast_path
    assert cuda.evaluator.supports_streamed_edges
    assert cuda.streamed_edges == "all"
    reference.calculate(atoms, properties=properties)
    cuda.calculate(atoms, properties=properties)
    for name in properties:
        np.testing.assert_allclose(
            cuda.results[name], reference.results[name], rtol=0.0, atol=2e-11
        )
    del cuda, reference
    gc.collect()


@pytest.mark.parametrize("head", MH1_HEADS)
def test_mh1_extracts_and_evaluates_each_head(head, tmp_path):
    data = extract_mace_data(_mh1_model_path(), species=[14], head=head)
    assert data["model_type"] == "MACE_Nonlinear"
    assert data["head"] == head
    assert data["atomic_numbers"] == [14]
    assert data["scale_shift"]["scale"]["shape"] == []
    assert data["atomic_energies"]["shape"] == [1, 89]
    path = tmp_path / f"mh1-{head}.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    atoms = Atoms(
        "Si2",
        positions=[[0.0, 0.0, 0.0], [2.2, 0.1, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
    )
    expected = MACECalculator(
        model_paths=str(_mh1_model_path()),
        device="cpu",
        default_dtype="float64",
        head=head,
    )
    expected.calculate(atoms.copy(), properties=["energy", "forces"])
    native_results = {}
    for use_kokkos in (False, True):
        actual = Symmetrix(path, use_kokkos=use_kokkos, dtype="float64")
        assert actual.evaluator.uses_mh1_fast_path
        actual.calculate(atoms.copy(), properties=["energy", "energies", "forces"])
        assert actual.results["energy"] == pytest.approx(
            expected.results["energy"], abs=2e-5
        )
        assert np.allclose(actual.results["forces"], expected.results["forces"], atol=2e-5)
        native_results[use_kokkos] = {
            name: np.array(actual.results[name], copy=True)
            for name in ("energy", "energies", "forces")
        }
    for name in native_results[False]:
        assert np.allclose(native_results[False][name], native_results[True][name], atol=2e-12)


def test_mh1_universal_extraction_preserves_all_species(tmp_path):
    data = extract_mace_data(_mh1_model_path(), head="matpes_r2scan")
    assert data["atomic_numbers"] == data["model_atomic_numbers"]
    assert data["model_indices"] == list(range(len(data["model_atomic_numbers"])))
    assert len(data["atomic_numbers"]) == 89
    path = tmp_path / "mh1-universal.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    serial = Symmetrix(path, use_kokkos=False, dtype="float64")
    kokkos = Symmetrix(path, use_kokkos=True, dtype="float64")
    assert serial.evaluator.uses_mh1_fast_path
    assert kokkos.evaluator.uses_mh1_fast_path
    upstream = MACECalculator(
        model_paths=str(_mh1_model_path()),
        device="cpu",
        default_dtype="float64",
        head="matpes_r2scan",
    )
    model_atomic_numbers = data["model_atomic_numbers"]
    selected_indices = (0, len(model_atomic_numbers) // 2, len(model_atomic_numbers) - 1)
    selected_numbers = [model_atomic_numbers[index] for index in selected_indices]
    for first, second in zip(selected_numbers, selected_numbers[1:] + selected_numbers[:1]):
        atoms = Atoms(
            numbers=[first, second],
            positions=[[0.0, 0.0, 0.0], [1.8, 0.1, 0.0]],
            cell=[12.0, 12.0, 12.0],
            pbc=True,
        )
        properties = ["energy", "energies", "forces", "stress"]
        upstream.calculate(atoms.copy(), properties=properties)
        serial.calculate(atoms.copy(), properties=properties)
        kokkos.calculate(atoms.copy(), properties=properties)
        for name in properties:
            assert np.allclose(serial.results[name], upstream.results[name], atol=2e-5)
            assert np.allclose(kokkos.results[name], serial.results[name], atol=2e-12)


def test_mh1_rejects_legacy_pair_spline_extraction():
    with pytest.raises(ValueError, match="format-version-3 compact schema"):
        extract_mace_data(
            _mh1_model_path(),
            species=[14],
            head="matpes_r2scan",
            radial_format="pair-splines",
        )


def test_mh1_raw_checkpoint_dispatches_to_native_kokkos():
    calculator = Symmetrix(
        _mh1_model_path(),
        head="matpes_r2scan",
        species=[14],
        dtype="float64",
    )
    assert type(calculator.evaluator).__name__ == "MACENonlinearKokkos"


def test_mh1_json_dispatch_does_not_depend_on_filename_suffix(mh1_si_artifact, tmp_path):
    _, model_path = mh1_si_artifact
    suffixless_path = tmp_path / "mh1-model-data"
    suffixless_path.symlink_to(model_path)
    calculator = Symmetrix(suffixless_path, use_kokkos=False, dtype="float64")
    assert type(calculator.evaluator).__name__ == "MACENonlinear"


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_mh1_fast_path_requires_exact_architecture(
    mh1_si_artifact, tmp_path, use_kokkos
):
    data, model_path = mh1_si_artifact
    changed = json.loads(json.dumps(data))
    changed["interactions"][0]["hidden_irreps"] = "512x0e"
    path = tmp_path / "near-mh1.json"
    path.write_text(json.dumps(changed))
    if use_kokkos and not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    evaluator_type = (
        native_symmetrix.MACENonlinearKokkos
        if use_kokkos
        else native_symmetrix.MACENonlinear
    )
    evaluator = evaluator_type(str(path))
    assert not evaluator.uses_mh1_fast_path
    assert not evaluator.supports_streamed_edges
    assert evaluator.streamed_edges_mode == "legacy"
    with pytest.raises(ValueError, match="published MACE-MH-1"):
        evaluator.set_streamed_edges("all")
    if use_kokkos and hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        with pytest.raises(ValueError, match="Float32.*published MACE-MH-1"):
            native_symmetrix.MACENonlinearKokkosFloat(str(path))
    atoms = Atoms(
        "Si2",
        positions=[[0.0, 0.0, 0.0], [2.2, 0.1, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    fast = Symmetrix(model_path, use_kokkos=use_kokkos, dtype="float64")
    generic = Symmetrix(path, use_kokkos=use_kokkos, dtype="float64")
    fast.calculate(atoms.copy(), properties=["energy", "energies", "forces", "stress"])
    generic.calculate(atoms.copy(), properties=["energy", "energies", "forces", "stress"])
    assert fast.evaluator.uses_mh1_fast_path
    assert not generic.evaluator.uses_mh1_fast_path
    for property_name in ("energy", "energies", "forces", "stress"):
        assert np.allclose(
            generic.results[property_name], fast.results[property_name], atol=2e-12
        )


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_mh1_fast_path_accepts_equivalent_irrep_formatting(
    mh1_si_artifact, tmp_path, use_kokkos
):
    data, _ = mh1_si_artifact
    changed = json.loads(json.dumps(data))
    changed["interactions"][1]["node_feats_irreps"] = " 512x0e + 512x1o "
    changed["interactions"][0]["gate"]["irreps_out"] = (
        " 512x0e + 512x1o + 512x2e + 512x3o "
    )
    path = tmp_path / "mh1-equivalent-irreps.json"
    path.write_text(json.dumps(changed, separators=(",", ":")))
    if use_kokkos and not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    evaluator_type = (
        native_symmetrix.MACENonlinearKokkos
        if use_kokkos
        else native_symmetrix.MACENonlinear
    )
    evaluator = evaluator_type(str(path))
    assert evaluator.uses_mh1_fast_path


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_mh1_fast_path_requires_conditionable_edge_mlp(
    mh1_si_artifact, tmp_path, use_kokkos
):
    data, _ = mh1_si_artifact
    changed = json.loads(json.dumps(data))
    width = changed["interactions"][0]["conv_tp_weights"]["layers"][0]["weight"]["shape"][1]
    changed["interactions"][0]["conv_tp_weights"]["layers"].insert(0, {
        "type": "layer_norm",
        "normalized_shape": [width],
        "eps": 1e-5,
        "weight": {"shape": [width], "values": [1.0] * width},
        "bias": {"shape": [width], "values": [0.0] * width},
    })
    path = tmp_path / "near-mh1-unconditionable.json"
    path.write_text(json.dumps(changed, separators=(",", ":")))
    if use_kokkos and not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    evaluator_type = (
        native_symmetrix.MACENonlinearKokkos
        if use_kokkos
        else native_symmetrix.MACENonlinear
    )
    evaluator = evaluator_type(str(path))
    assert not evaluator.uses_mh1_fast_path
    if use_kokkos and hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        with pytest.raises(ValueError, match="Float32.*published MACE-MH-1"):
            native_symmetrix.MACENonlinearKokkosFloat(str(path))


def test_mh1_extraction_rejects_unsupported_architecture_features():
    from symmetrix.extract_mace_nonlinear import extract_mace_nonlinear_data

    model = torch.load(
        _mh1_model_path(), map_location=torch.device("cpu"), weights_only=False
    ).to(torch.float64)
    model = remove_pt_head(model, "matpes_r2scan")

    model.use_last_readout_only = True
    with pytest.raises(RuntimeError, match="use_last_readout_only"):
        extract_mace_nonlinear_data(model, [14])
    model.use_last_readout_only = False

    model.joint_embedding = torch.nn.Identity()
    with pytest.raises(RuntimeError, match="joint_embedding"):
        extract_mace_nonlinear_data(model, [14])
    del model.joint_embedding

    original_readout = model.readouts[-1]
    model.readouts[-1] = torch.nn.Identity()
    with pytest.raises(RuntimeError, match="only supports LinearReadoutBlock"):
        extract_mace_nonlinear_data(model, [14])
    model.readouts[-1] = original_readout


def test_mh1_single_head_wigner_tensors_preserve_float64():
    from e3nn import o3
    from symmetrix.extract_mace_nonlinear import extract_mace_nonlinear_data

    model = torch.load(
        _mh1_model_path(), map_location=torch.device("cpu"), weights_only=False
    ).to(torch.float64)
    model = remove_pt_head(model, "matpes_r2scan")
    previous_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        data = extract_mace_nonlinear_data(model, [14])
    finally:
        torch.set_default_dtype(previous_dtype)
    serialized = data["interactions"][0]["conv_tp"]
    module = model.interactions[0].conv_tp
    for native_instruction, instruction in zip(
        serialized["instructions"], module.instructions, strict=True
    ):
        expected = o3.wigner_3j(
            module.irreps_in1[instruction.i_in1].ir.l,
            module.irreps_in2[instruction.i_in2].ir.l,
            module.irreps_out[instruction.i_out].ir.l,
        ).to(torch.float64)
        actual = np.asarray(native_instruction["wigner_3j"]["values"])
        assert np.allclose(actual, expected.numpy().reshape(-1), atol=1e-14)


def test_mh1_native_serial_and_kokkos_match_upstream(mh1_si_artifact):
    _, model_path = mh1_si_artifact
    atoms = Atoms(
        "Si2",
        positions=[[0.0, 0.0, 0.0], [1.8, 0.1, 0.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    expected = MACECalculator(
        model_paths=str(_mh1_model_path()),
        device="cpu",
        default_dtype="float64",
        head="matpes_r2scan",
    )
    expected.calculate(atoms.copy(), properties=["energy", "energies", "forces", "stress"])

    native_results = {}
    for use_kokkos, evaluator_name in ((False, "MACENonlinear"), (True, "MACENonlinearKokkos")):
        actual = Symmetrix(model_path, use_kokkos=use_kokkos, dtype="float64")
        assert actual.evaluator.uses_mh1_fast_path
        actual.calculate(atoms.copy(), properties=["energy", "energies", "forces", "stress"])
        assert type(actual.evaluator).__name__ == evaluator_name
        assert actual.cutoff == pytest.approx(expected.r_max)
        assert actual.results["energy"] == pytest.approx(expected.results["energy"], abs=2e-5)
        assert np.allclose(actual.results["energies"], expected.results["energies"], atol=2e-5)
        assert np.allclose(actual.results["forces"], expected.results["forces"], atol=2e-5)
        assert np.allclose(actual.results["stress"], expected.results["stress"], atol=2e-5)
        native_results[use_kokkos] = {
            name: np.array(actual.results[name], copy=True)
            for name in ("energy", "energies", "forces", "stress")
        }

    for name in native_results[False]:
        assert np.allclose(native_results[False][name], native_results[True][name], atol=2e-12)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_mh1_kokkos_repeated_calculations(mh1_si_artifact, dtype):
    _, model_path = mh1_si_artifact
    serial = Symmetrix(model_path, use_kokkos=False, dtype="float64")
    kokkos = Symmetrix(model_path, use_kokkos=True, dtype=dtype)
    tolerance = 2e-12 if dtype == "float64" else 2e-5
    atoms = Atoms(
        "Si2",
        positions=[[0.0, 0.0, 0.0], [2.2, 0.1, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
    )
    first_energy = None
    for displacement in (0.0, 0.04, -0.03):
        moved = atoms.copy()
        moved.positions[1] += [displacement, -0.5 * displacement, 0.25 * displacement]
        serial.calculate(moved, properties=["energy", "forces", "stress"])
        kokkos.calculate(moved, properties=["energy", "forces", "stress"])
        assert kokkos.results["energy"] == pytest.approx(
            serial.results["energy"], abs=tolerance
        )
        assert np.allclose(
            kokkos.results["forces"], serial.results["forces"], atol=tolerance
        )
        assert np.allclose(
            kokkos.results["stress"], serial.results["stress"], atol=tolerance
        )
        if first_energy is None:
            first_energy = kokkos.results["energy"]
        elif displacement != 0.0:
            assert kokkos.results["energy"] != pytest.approx(first_energy, abs=1e-8)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_mh1_kokkos_reuses_workspaces_across_graph_sizes(mh1_si_artifact, dtype):
    _, model_path = mh1_si_artifact
    serial = Symmetrix(model_path, use_kokkos=False, dtype="float64")
    kokkos = Symmetrix(model_path, use_kokkos=True, dtype=dtype)
    primitive = bulk("Si", "diamond", a=5.43)
    systems = (primitive, primitive.repeat((2, 2, 2)), primitive.repeat((3, 3, 3)))

    for atoms in (*systems, *reversed(systems), *systems):
        serial.calculate(atoms, properties=["energy", "energies", "forces", "stress"])
        kokkos.calculate(atoms, properties=["energy", "energies", "forces", "stress"])
        for property_name in ("energy", "energies", "forces", "stress"):
            tolerance = 2e-12
            if dtype == "float32":
                tolerance = 2e-4 * len(atoms) if property_name == "energy" else 2e-4
            assert np.allclose(
                kokkos.results[property_name],
                serial.results[property_name],
                atol=tolerance,
            )


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_mh1_kokkos_handles_changes_in_supported_species(
    mh1_h_si_artifact, dtype
):
    _, model_path = mh1_h_si_artifact
    serial = Symmetrix(model_path, use_kokkos=False, dtype="float64")
    kokkos = Symmetrix(model_path, use_kokkos=True, dtype=dtype)
    kokkos_tolerance = 2e-12 if dtype == "float64" else 2e-4
    upstream = MACECalculator(
        model_paths=str(_mh1_model_path()),
        device="cpu",
        default_dtype="float64",
        head="matpes_r2scan",
    )
    for symbols, distance in (
        ("Si2", 2.2), ("H2", 0.8), ("SiH", 1.5), ("HSi", 1.5), ("Si2", 2.4)
    ):
        atoms = Atoms(
            symbols,
            positions=[[0.0, 0.0, 0.0], [distance, 0.1, 0.0]],
            cell=[14.0, 14.0, 14.0],
            pbc=True,
        )
        upstream.calculate(atoms.copy(), properties=["energy", "forces", "stress"])
        serial.calculate(atoms, properties=["energy", "forces", "stress"])
        kokkos.calculate(atoms, properties=["energy", "forces", "stress"])
        assert serial.results["energy"] == pytest.approx(upstream.results["energy"], abs=2e-5)
        assert np.allclose(serial.results["forces"], upstream.results["forces"], atol=2e-5)
        assert np.allclose(serial.results["stress"], upstream.results["stress"], atol=2e-5)
        assert kokkos.results["energy"] == pytest.approx(
            serial.results["energy"], abs=kokkos_tolerance
        )
        assert np.allclose(
            kokkos.results["forces"], serial.results["forces"], atol=kokkos_tolerance
        )
        assert np.allclose(
            kokkos.results["stress"], serial.results["stress"], atol=kokkos_tolerance
        )


def test_mh1_serial_matches_upstream_for_isolated_atom(mh1_si_artifact):
    _, model_path = mh1_si_artifact
    atoms = Atoms("Si", positions=[[0.0, 0.0, 0.0]], cell=[10.0, 10.0, 10.0])
    actual = Symmetrix(model_path, use_kokkos=False, dtype="float64")
    expected = MACECalculator(
        model_paths=str(_mh1_model_path()),
        device="cpu",
        default_dtype="float64",
        head="matpes_r2scan",
    )
    actual.calculate(atoms.copy(), properties=["energy", "energies", "forces"])
    expected.calculate(atoms.copy(), properties=["energy", "energies", "forces"])
    assert actual.evaluator.uses_mh1_fast_path
    assert len(actual.evaluator.node_forces) == 0
    assert actual.results["energy"] == pytest.approx(expected.results["energy"], abs=2e-9)
    assert np.allclose(actual.results["energies"], expected.results["energies"], atol=2e-9)
    assert np.allclose(actual.results["forces"], expected.results["forces"], atol=2e-9)


@pytest.mark.parametrize(
    ("use_kokkos", "dtype"),
    [(False, "float64"), (True, "float64"), (True, "float32")],
)
def test_mh1_native_force_matches_finite_difference(
    mh1_si_artifact, use_kokkos, dtype
):
    _, model_path = mh1_si_artifact
    atoms = Atoms(
        "Si2",
        positions=[[0.0, 0.0, 0.0], [2.2, 0.1, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
    )
    atoms.calc = Symmetrix(model_path, use_kokkos=use_kokkos, dtype=dtype)
    force = atoms.get_forces()[1, 0]
    step = 1e-4 if dtype == "float64" else 2e-3
    displaced = atoms.copy()
    displaced.calc = atoms.calc
    displaced.positions[1, 0] += step
    energy_plus = displaced.get_potential_energy()
    displaced.positions[1, 0] -= 2 * step
    energy_minus = displaced.get_potential_energy()
    tolerance = 2e-4 if dtype == "float64" else 3e-3
    assert force == pytest.approx(
        -(energy_plus - energy_minus) / (2 * step), abs=tolerance
    )


@pytest.mark.parametrize("module_name", ["conv_tp_weights", "density_fn"])
def test_mh1_affine_mlp_forward_and_reverse_match_autograd(mh1_si_artifact, module_name):
    data, _ = mh1_si_artifact
    model = torch.load(
        _mh1_model_path(), map_location=torch.device("cpu"), weights_only=False
    ).to(torch.float64)
    model = remove_pt_head(model, "matpes_r2scan")
    torch_module = getattr(model.interactions[0], module_name)
    native_module = native_symmetrix.AffineMLP(
        json.dumps(data["interactions"][0][module_name])
    )
    rng = np.random.default_rng(123)
    values = rng.normal(size=native_module.input_size)
    seed = rng.normal(size=native_module.output_size)
    torch_values = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    expected = torch_module(torch_values[None, :])[0]
    (expected * torch.tensor(seed, dtype=torch.float64)).sum().backward()
    assert np.allclose(native_module.evaluate(values), expected.detach().numpy(), atol=2e-11)
    assert np.allclose(native_module.reverse(values, seed), torch_values.grad.numpy(), atol=2e-10)


def test_mh1_e3_linear_batch_matches_repeated_scalar_calls():
    definition = {
        "irreps_in": "2x0e+2x0e+2x1o",
        "irreps_out": "2x0e+2x1o",
        "instructions": [
            {
                "i_in": 0,
                "i_out": 0,
                "path_weight": 0.75,
                "path_shape": [2, 2],
            },
            {
                "i_in": 1,
                "i_out": 0,
                "path_weight": 1.25,
                "path_shape": [2, 2],
            },
            {
                "i_in": 2,
                "i_out": 1,
                "path_weight": -0.4,
                "path_shape": [2, 2],
            },
        ],
        "weight": {
            "shape": [12],
            "values": [
                0.2, -0.1, 0.5, 0.3,
                -0.2, 0.4, 0.8, -0.5,
                -0.4, 0.7, 0.1, 0.6,
            ],
        },
        "bias": {"shape": [8], "values": [0.1, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]},
        "output_mask": {
            "shape": [8],
            "values": [1.0, 0.5, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0],
        },
    }
    module = native_symmetrix.E3Linear(json.dumps(definition))
    rng = np.random.default_rng(1024)
    samples = 7
    values = rng.normal(size=(samples, module.input_dimension))
    seeds = rng.normal(size=(samples, module.output_dimension))
    expected = np.concatenate([module.evaluate(row) for row in values])
    expected_adjoint = np.concatenate([module.reverse(row) for row in seeds])
    assert np.allclose(module.evaluate_batch(values.ravel(), samples), expected, atol=2e-14)
    assert np.allclose(
        module.reverse_batch(seeds.ravel(), samples), expected_adjoint, atol=2e-14
    )
    if hasattr(native_symmetrix, "E3LinearKokkos"):
        if not native_symmetrix._kokkos_is_initialized():
            native_symmetrix._init_kokkos()
        kokkos = native_symmetrix.E3LinearKokkos(json.dumps(definition))
        assert np.allclose(
            kokkos.evaluate_batch(values.ravel(), samples), expected, atol=2e-14
        )
        assert np.allclose(
            kokkos.reverse_batch(seeds.ravel(), samples), expected_adjoint, atol=2e-14
        )
    if hasattr(native_symmetrix, "E3LinearKokkosFloat"):
        float_module = native_symmetrix.E3LinearKokkosFloat(
            json.dumps(definition)
        )
        float_module.set_backend("scalar")
        scalar_values = np.asarray(
            float_module.evaluate_batch(values.astype(np.float32).ravel(), samples)
        )
        scalar_adjoints = np.asarray(
            float_module.reverse_batch(seeds.astype(np.float32).ravel(), samples)
        )
        float_module.set_backend("packed_gemm")
        packed_values = np.asarray(
            float_module.evaluate_batch(values.astype(np.float32).ravel(), samples)
        )
        packed_adjoints = np.asarray(
            float_module.reverse_batch(seeds.astype(np.float32).ravel(), samples)
        )
        assert float_module.backend == "packed_gemm"
        assert np.allclose(packed_values, scalar_values, atol=2e-6)
        assert np.allclose(packed_adjoints, scalar_adjoints, atol=2e-6)


@pytest.mark.parametrize("layer", [0, 1])
def test_mh1_float32_linear_backends_match_on_official_shapes(
    mh1_si_artifact, layer
):
    if not hasattr(native_symmetrix, "E3LinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Float32 Kokkos E3 primitives")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    data, _ = mh1_si_artifact
    module = native_symmetrix.E3LinearKokkosFloat(
        json.dumps(data["interactions"][layer]["linear_up"])
    )
    rng = np.random.default_rng(2048 + layer)
    samples = 17
    values = rng.normal(
        scale=0.1, size=(samples, module.input_dimension)
    ).astype(np.float32)
    seeds = rng.normal(
        scale=0.1, size=(samples, module.output_dimension)
    ).astype(np.float32)
    module.set_backend("scalar")
    scalar_values = np.asarray(module.evaluate_batch(values.ravel(), samples))
    scalar_adjoints = np.asarray(module.reverse_batch(seeds.ravel(), samples))
    module.set_backend("packed_gemm")
    packed_values = np.asarray(module.evaluate_batch(values.ravel(), samples))
    packed_adjoints = np.asarray(module.reverse_batch(seeds.ravel(), samples))
    assert module.selected_backend(samples) == "packed_gemm"
    assert module.workspace_bytes > 0
    assert np.allclose(packed_values, scalar_values, atol=3e-6)
    assert np.allclose(packed_adjoints, scalar_adjoints, atol=3e-6)


@pytest.mark.parametrize("layer", [0, 1])
def test_mh1_tensor_product_forward_and_reverse_match_autograd(
    mh1_si_artifact, layer
):
    data, _ = mh1_si_artifact
    model = torch.load(
        _mh1_model_path(), map_location=torch.device("cpu"), weights_only=False
    ).to(torch.float64)
    model = remove_pt_head(model, "matpes_r2scan")
    torch_module = model.interactions[layer].conv_tp
    native_module = native_symmetrix.E3TensorProduct(
        json.dumps(data["interactions"][layer]["conv_tp"])
    )
    rng = np.random.default_rng(321)
    x = rng.normal(scale=0.2, size=native_module.input_1_dimension)
    y = rng.normal(scale=0.2, size=native_module.input_2_dimension)
    w = rng.normal(scale=0.2, size=torch_module.weight_numel)
    seed = rng.normal(scale=0.2, size=native_module.output_dimension)
    tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    ty = torch.tensor(y, dtype=torch.float64, requires_grad=True)
    tw = torch.tensor(w, dtype=torch.float64, requires_grad=True)
    expected = torch_module(tx[None, :], ty[None, :], tw[None, :])[0]
    (expected * torch.tensor(seed, dtype=torch.float64)).sum().backward()
    actual = native_module.evaluate(x, y, w)
    x_adj, y_adj, w_adj = native_module.reverse(x, y, w, seed)
    assert np.allclose(actual, expected.detach().numpy(), atol=2e-6)
    assert np.allclose(x_adj, tx.grad.numpy(), atol=2e-6)
    assert np.allclose(y_adj, ty.grad.numpy(), atol=2e-6)
    assert np.allclose(w_adj, tw.grad.numpy(), atol=2e-6)
    if hasattr(native_symmetrix, "E3TensorProductKokkos"):
        if not native_symmetrix._kokkos_is_initialized():
            native_symmetrix._init_kokkos()
        kokkos = native_symmetrix.E3TensorProductKokkos(
            json.dumps(data["interactions"][layer]["conv_tp"])
        )
        assert kokkos.uses_mh1_fast_path
        kokkos_actual = kokkos.evaluate(x, y, w)
        kokkos_x_adj, kokkos_y_adj, kokkos_w_adj = kokkos.reverse(x, y, w, seed)
        assert np.allclose(kokkos_actual, actual, atol=2e-12)
        assert np.allclose(kokkos_x_adj, x_adj, atol=2e-12)
        assert np.allclose(kokkos_y_adj, y_adj, atol=2e-12)
        assert np.allclose(kokkos_w_adj, w_adj, atol=2e-12)
    if hasattr(native_symmetrix, "E3TensorProductKokkosFloat"):
        float_module = native_symmetrix.E3TensorProductKokkosFloat(
            json.dumps(data["interactions"][layer]["conv_tp"])
        )
        factors = np.array([1.0, 0.7, -0.4], dtype=np.float32)
        first = np.asarray([factor * x for factor in factors], dtype=np.float32)
        second = np.asarray(
            [(1.0 - 0.2 * factor) * y for factor in factors],
            dtype=np.float32,
        )
        batch_weights = np.asarray(
            [(1.0 + 0.1 * factor) * w for factor in factors],
            dtype=np.float32,
        )
        seeds = np.asarray(
            [(1.0 - 0.1 * factor) * seed for factor in factors],
            dtype=np.float32,
        )
        expected_values = np.concatenate([
            native_module.evaluate(node_x, node_y, node_w)
            for node_x, node_y, node_w in zip(first, second, batch_weights)
        ])
        expected_adjoints = [
            native_module.reverse(node_x, node_y, node_w, node_seed)
            for node_x, node_y, node_w, node_seed
            in zip(first, second, batch_weights, seeds)
        ]
        actual_values = float_module.evaluate_batch(
            first.ravel(), second.ravel(), batch_weights.ravel(), len(factors)
        )
        actual_adjoints = float_module.reverse_batch(
            first.ravel(), second.ravel(), batch_weights.ravel(),
            seeds.ravel(), len(factors)
        )
        assert float_module.uses_mh1_fast_path
        assert float_module.backend == "official_kokkos"
        assert np.allclose(actual_values, expected_values, atol=3e-6)
        for actual_component, expected_component in zip(
            actual_adjoints, zip(*expected_adjoints)
        ):
            assert np.allclose(
                actual_component, np.concatenate(expected_component), atol=3e-6
            )


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_mh1_uuu_tensor_product_uses_diagonal_weights(use_kokkos):
    definition = {
        "irreps_in1": "3x0e",
        "irreps_in2": "3x0e",
        "irreps_out": "3x0e",
        "instructions": [{
            "i_in1": 0,
            "i_in2": 0,
            "i_out": 0,
            "connection_mode": "uuu",
            "has_weight": True,
            "path_weight": 1.0,
            "path_shape": [3],
            "wigner_3j": {"shape": [1, 1, 1], "values": [1.0]},
        }],
        "weight": {"shape": [0], "values": []},
        "output_mask": {"shape": [3], "values": [1.0, 1.0, 1.0]},
    }
    if use_kokkos:
        if not hasattr(native_symmetrix, "E3TensorProductKokkos"):
            pytest.skip("Symmetrix was built without Kokkos support")
        if not native_symmetrix._kokkos_is_initialized():
            native_symmetrix._init_kokkos()
        module = native_symmetrix.E3TensorProductKokkos(json.dumps(definition))
    else:
        module = native_symmetrix.E3TensorProduct(json.dumps(definition))
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([4.0, 5.0, 6.0])
    weights = np.array([0.1, 0.2, 0.3])
    seed = np.array([0.7, -0.4, 0.2])
    assert np.allclose(module.evaluate(x, y, weights), weights * x * y)
    x_adj, y_adj, weight_adj = module.reverse(x, y, weights, seed)
    assert np.allclose(x_adj, seed * weights * y)
    assert np.allclose(y_adj, seed * weights * x)
    assert np.allclose(weight_adj, seed * x * y)


def test_mh1_kokkos_tensor_product_validates_binding_dimensions():
    if not hasattr(native_symmetrix, "E3TensorProductKokkos"):
        pytest.skip("Symmetrix was built without Kokkos support")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    definition = {
        "irreps_in1": "1x0e",
        "irreps_in2": "1x0e",
        "irreps_out": "1x0e",
        "instructions": [{
            "i_in1": 0,
            "i_in2": 0,
            "i_out": 0,
            "connection_mode": "uvu",
            "has_weight": True,
            "path_weight": 1.0,
            "path_shape": [1, 1],
            "wigner_3j": {"shape": [1, 1, 1], "values": [1.0]},
        }],
        "weight": {"shape": [1], "values": [2.0]},
        "output_mask": {"shape": [1], "values": [1.0]},
    }
    module = native_symmetrix.E3TensorProductKokkos(json.dumps(definition))
    assert module.has_internal_weights
    assert module.evaluate([3.0], [5.0], []) == pytest.approx([30.0])
    with pytest.raises(ValueError, match="input dimensions"):
        module.evaluate([], [5.0], [])
    with pytest.raises(ValueError, match="weight dimensions"):
        module.evaluate([3.0], [5.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="reverse dimensions"):
        module.reverse([3.0], [5.0], [], [])


def test_mh1_affine_mlp_rejects_inconsistent_layer_dimensions():
    definition = {
        "layers": [
            {
                "type": "linear",
                "weight": {"shape": [2, 2], "values": [1.0, 0.0, 0.0, 1.0]},
                "bias": {"shape": [2], "values": [0.0, 0.0]},
            },
            {
                "type": "layer_norm",
                "normalized_shape": [3],
                "eps": 1e-5,
                "weight": {"shape": [3], "values": [1.0, 1.0, 1.0]},
                "bias": {"shape": [3], "values": [0.0, 0.0, 0.0]},
            },
        ]
    }
    with pytest.raises(ValueError, match="dimensions are inconsistent"):
        native_symmetrix.AffineMLP(json.dumps(definition))


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_mh1_product_basis_rejects_malformed_tensor_layout(use_kokkos):
    linear = {
        "irreps_in": "1x0e",
        "irreps_out": "1x0e",
        "instructions": [{
            "i_in": 0,
            "i_out": 0,
            "path_weight": 1.0,
            "path_shape": [1, 1],
        }],
        "weight": {"shape": [1], "values": [1.0]},
        "bias": {"shape": [0], "values": []},
        "output_mask": {"shape": [1], "values": [1.0]},
    }
    definition = {
        "node_feats_irreps": "1x0e",
        "target_irreps": "1x0e",
        "use_sc": False,
        "use_agnostic_product": False,
        "linear": linear,
        "symmetric_contractions": {
            "irreps_in": "1x0e",
            "irreps_out": "1x0e",
            "contractions": [{
                "correlation": 1,
                "weights": [],
                "weights_max": {"shape": [1], "values": [1.0]},
                "u_tensors": [{"shape": [1, 1], "values": [1.0]}],
            }],
        },
    }
    if use_kokkos:
        if not hasattr(native_symmetrix, "E3ProductBasisKokkos"):
            pytest.skip("Symmetrix was built without Kokkos support")
        if not native_symmetrix._kokkos_is_initialized():
            native_symmetrix._init_kokkos()
        product = native_symmetrix.E3ProductBasisKokkos
    else:
        product = native_symmetrix.E3ProductBasis
    with pytest.raises(ValueError, match="tensor layout"):
        product(json.dumps(definition))


def test_mh1_generic_product_batch_supports_varying_elements_without_skip():
    linear = {
        "irreps_in": "2x0e",
        "irreps_out": "2x0e",
        "instructions": [{
            "i_in": 0,
            "i_out": 0,
            "path_weight": 1.0,
            "path_shape": [2, 2],
        }],
        "weight": {"shape": [4], "values": [1.0, 0.0, 0.0, 1.0]},
        "bias": {"shape": [0], "values": []},
        "output_mask": {"shape": [2], "values": [1.0, 1.0]},
    }
    definition = {
        "node_feats_irreps": "2x0e",
        "target_irreps": "2x0e",
        "use_sc": False,
        "use_agnostic_product": False,
        "linear": linear,
        "symmetric_contractions": {
            "irreps_in": "2x0e",
            "irreps_out": "2x0e",
            "contractions": [{
                "correlation": 1,
                "weights": [],
                "weights_max": {
                    "shape": [2, 1, 2],
                    "values": [2.0, 3.0, 5.0, 7.0],
                },
                "u_tensors": [{"shape": [1, 1], "values": [1.0]}],
            }],
        },
    }
    product = native_symmetrix.E3ProductBasis(json.dumps(definition))
    features = np.array([[0.2, -0.4], [1.1, 0.3], [-0.7, 0.9]])
    elements = [0, 1, 0]
    seeds = np.array([[0.6, -0.1], [-0.2, 0.8], [0.4, 0.5]])
    expected_values = np.concatenate([
        product.evaluate(row, [], element)
        for row, element in zip(features, elements)
    ])
    expected_adjoints = np.concatenate([
        product.reverse(row, element, seed)[0]
        for row, element, seed in zip(features, elements, seeds)
    ])
    actual_values = product.evaluate_batch(features.ravel(), [], elements, len(elements))
    actual_adjoints, skip_adjoints = product.reverse_batch(
        features.ravel(), elements, seeds.ravel(), len(elements)
    )
    assert np.allclose(actual_values, expected_values, atol=2e-14)
    assert np.allclose(actual_adjoints, expected_adjoints, atol=2e-14)
    assert np.allclose(skip_adjoints, 0.0)


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_mh1_native_rejects_inconsistent_graph_metadata(mh1_si_artifact, use_kokkos):
    _, model_path = mh1_si_artifact
    if use_kokkos:
        if not hasattr(native_symmetrix, "MACENonlinearKokkos"):
            pytest.skip("Symmetrix was built without Kokkos support")
        if not native_symmetrix._kokkos_is_initialized():
            native_symmetrix._init_kokkos()
        evaluator = native_symmetrix.MACENonlinearKokkos(str(model_path))
    else:
        evaluator = native_symmetrix.MACENonlinear(str(model_path))
    with pytest.raises(ValueError, match="neighbor counts do not match"):
        evaluator.compute_node_energies_forces(1, [0], [1], [], [], [], [])


@pytest.mark.parametrize("product_index", [0, 1])
def test_mh1_product_basis_forward_and_reverse_match_autograd(
    mh1_si_artifact, product_index
):
    data, _ = mh1_si_artifact
    model = torch.load(
        _mh1_model_path(), map_location=torch.device("cpu"), weights_only=False
    ).to(torch.float64)
    model = remove_pt_head(model, "matpes_r2scan")
    torch_module = model.products[product_index]
    native_module = native_symmetrix.E3ProductBasis(
        json.dumps(data["products"][product_index])
    )
    assert native_module.uses_compiled_plan
    assert native_module.compiled_term_count > 0
    rng = np.random.default_rng(456)
    features = rng.normal(scale=0.05, size=native_module.input_dimension)
    skip = rng.normal(scale=0.05, size=native_module.output_dimension)
    seed = rng.normal(scale=0.05, size=native_module.output_dimension)
    torch_features = torch.tensor(features, dtype=torch.float64, requires_grad=True)
    torch_skip = torch.tensor(skip, dtype=torch.float64, requires_grad=True)
    attrs = torch.zeros((1, 89), dtype=torch.float64)
    attrs[0, 13] = 1.0
    pieces = []
    offset = 0
    for multiplicity, irrep in torch_module.symmetric_contractions.irreps_in:
        width = irrep.dim
        size = multiplicity * width
        pieces.append(torch_features[offset:offset + size].reshape(multiplicity, width))
        offset += size
    feature_major = torch.cat(pieces, dim=1)[None, :, :]
    expected = torch_module(feature_major, torch_skip[None, :], attrs)[0]
    (expected * torch.tensor(seed, dtype=torch.float64)).sum().backward()
    actual = native_module.evaluate(features, skip, 0)
    feature_adj, skip_adj = native_module.reverse(features, 0, seed)
    assert np.allclose(actual, expected.detach().numpy(), atol=2e-6)
    assert np.allclose(feature_adj, torch_features.grad.numpy(), atol=2e-6)
    assert np.allclose(skip_adj, torch_skip.grad.numpy(), atol=2e-6)

    batch_features = np.stack([features, 0.7 * features, -0.4 * features])
    batch_skip = np.stack([skip, -0.2 * skip, 0.5 * skip])
    batch_seed = np.stack([seed, 0.3 * seed, -0.6 * seed])
    expected_batch = np.concatenate([
        native_module.evaluate(node, node_skip, 0)
        for node, node_skip in zip(batch_features, batch_skip)
    ])
    expected_feature_adjoints = []
    expected_skip_adjoints = []
    for node, node_seed in zip(batch_features, batch_seed):
        node_adjoint, node_skip_adjoint = native_module.reverse(node, 0, node_seed)
        expected_feature_adjoints.extend(node_adjoint)
        expected_skip_adjoints.extend(node_skip_adjoint)
    actual_batch = native_module.evaluate_batch(
        batch_features.ravel(), batch_skip.ravel(), [0, 0, 0], 3
    )
    actual_feature_adjoints, actual_skip_adjoints = native_module.reverse_batch(
        batch_features.ravel(), [0, 0, 0], batch_seed.ravel(), 3
    )
    assert np.allclose(actual_batch, expected_batch, atol=2e-12)
    assert np.allclose(actual_feature_adjoints, expected_feature_adjoints, atol=2e-12)
    assert np.allclose(actual_skip_adjoints, expected_skip_adjoints, atol=2e-12)


@pytest.mark.parametrize("product_index", [0, 1])
def test_mh1_kokkos_compiled_product_matches_serial_batch(
    mh1_si_artifact, product_index
):
    if not hasattr(native_symmetrix, "E3ProductBasisKokkos"):
        pytest.skip("Symmetrix was built without Kokkos support")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    data, _ = mh1_si_artifact
    definition = json.dumps(data["products"][product_index])
    serial = native_symmetrix.E3ProductBasis(definition)
    kokkos = native_symmetrix.E3ProductBasisKokkos(definition)
    assert kokkos.uses_compiled_plan
    assert kokkos.compiled_term_count == serial.compiled_term_count

    rng = np.random.default_rng(4096 + product_index)
    samples = 6
    features = rng.normal(scale=0.05, size=(samples, serial.input_dimension))
    skips = rng.normal(scale=0.05, size=(samples, serial.output_dimension))
    seeds = rng.normal(scale=0.05, size=(samples, serial.output_dimension))
    elements = [0] * samples
    expected = serial.evaluate_batch(
        features.ravel(), skips.ravel(), elements, samples
    )
    expected_feature_adjoints, expected_skip_adjoints = serial.reverse_batch(
        features.ravel(), elements, seeds.ravel(), samples
    )
    actual = kokkos.evaluate_batch(
        features.ravel(), skips.ravel(), elements, samples
    )
    feature_adjoints, skip_adjoints = kokkos.reverse_batch(
        features.ravel(), elements, seeds.ravel(), samples
    )
    assert np.allclose(actual, expected, atol=2e-12)
    assert np.allclose(feature_adjoints, expected_feature_adjoints, atol=2e-12)
    assert np.allclose(skip_adjoints, expected_skip_adjoints, atol=2e-12)


def test_mh1_conditioned_affine_mlp_matches_full_input(mh1_si_artifact):
    data, _ = mh1_si_artifact
    definition = data["interactions"][0]["conv_tp_weights"]
    full = native_symmetrix.AffineMLP(json.dumps(definition))
    rng = np.random.default_rng(918)
    dynamic_size = len(data["radial_embedding"]["basis"]["weights"]["values"])
    dynamic = rng.normal(scale=0.1, size=dynamic_size)
    source = rng.normal(scale=0.1, size=512)
    target = rng.normal(scale=0.1, size=full.input_size-dynamic_size-len(source))
    seed = rng.normal(scale=0.1, size=full.output_size)
    assert full.supports_conditioned_input(dynamic_size)
    source_contribution = full.first_layer_contribution(dynamic_size, source)
    target_contribution = full.first_layer_contribution(dynamic_size+len(source), target)
    complete = np.concatenate([dynamic, source, target])
    assert np.allclose(
        full.evaluate_conditioned(dynamic, source_contribution, target_contribution),
        full.evaluate(complete),
        atol=2e-12,
    )
    assert np.allclose(
        full.reverse_conditioned(
            dynamic, source_contribution, target_contribution, seed
        ),
        full.reverse(complete, seed)[:dynamic_size],
        atol=2e-12,
    )


@pytest.mark.parametrize("layer", [0, 1])
@pytest.mark.parametrize("module_name", ["conv_tp_weights", "density_fn"])
def test_mh1_conditioned_affine_mlp_batch_matches_scalar_calls(
    mh1_si_artifact, layer, module_name
):
    data, _ = mh1_si_artifact
    module = native_symmetrix.AffineMLP(
        json.dumps(data["interactions"][layer][module_name])
    )
    rng = np.random.default_rng(2048 + layer)
    samples = 5
    dynamic_size = len(data["radial_embedding"]["basis"]["weights"]["values"])
    dynamic = rng.normal(scale=0.1, size=(samples, dynamic_size))
    seeds = rng.normal(scale=0.1, size=(samples, module.output_size))
    first_contributions = []
    second_contributions = []
    for _ in range(samples):
        source = rng.normal(scale=0.1, size=512)
        target = rng.normal(scale=0.1, size=512)
        first_contributions.append(
            module.first_layer_contribution(dynamic_size, source)
        )
        second_contributions.append(
            module.first_layer_contribution(dynamic_size + 512, target)
        )
    row_contributions = np.asarray(first_contributions) + np.asarray(second_contributions)
    expected_outputs = []
    expected_adjoints = []
    for row, first, second, seed in zip(
        dynamic, first_contributions, second_contributions, seeds
    ):
        expected_outputs.extend(module.evaluate_conditioned(row, first, second))
        expected_adjoints.extend(module.reverse_conditioned(row, first, second, seed))
    outputs, adjoints = module.conditioned_batch(
        dynamic.ravel(),
        samples,
        dynamic_size,
        row_contributions.ravel(),
        seeds.ravel(),
    )
    assert np.allclose(outputs, expected_outputs, atol=2e-12)
    assert np.allclose(adjoints, expected_adjoints, atol=2e-12)
    if hasattr(native_symmetrix, "AffineMLPKokkos"):
        if not native_symmetrix._kokkos_is_initialized():
            native_symmetrix._init_kokkos()
        kokkos = native_symmetrix.AffineMLPKokkos(
            json.dumps(data["interactions"][layer][module_name])
        )
        assert kokkos.supports_conditioned_input(dynamic_size)
        kokkos_outputs, kokkos_adjoints = kokkos.conditioned_batch(
            dynamic.ravel(),
            samples,
            dynamic_size,
            row_contributions.ravel(),
            seeds.ravel(),
        )
        assert np.allclose(kokkos_outputs, expected_outputs, atol=2e-12)
        assert np.allclose(kokkos_adjoints, expected_adjoints, atol=2e-12)


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_mh1_native_rejects_neighbor_source_type_mismatch(
    mh1_h_si_artifact, use_kokkos
):
    _, model_path = mh1_h_si_artifact
    if use_kokkos:
        if not hasattr(native_symmetrix, "MACENonlinearKokkos"):
            pytest.skip("Symmetrix was built without Kokkos support")
        if not native_symmetrix._kokkos_is_initialized():
            native_symmetrix._init_kokkos()
        evaluator = native_symmetrix.MACENonlinearKokkos(str(model_path))
    else:
        evaluator = native_symmetrix.MACENonlinear(str(model_path))
    with pytest.raises(ValueError, match="invalid edge index, type"):
        evaluator.compute_node_energies_forces(
            2,
            [0, 1],
            [1, 0],
            [1],
            [0],
            [1.5, 0.0, 0.0],
            [1.5],
        )


def test_mh1_native_serial_float32_is_rejected(mh1_si_artifact):
    _, model_path = mh1_si_artifact
    with pytest.raises(ValueError, match="Native serial.*require dtype 'float64'"):
        Symmetrix(model_path, use_kokkos=False, dtype="float32")


def test_mh1_kokkos_float32_streamed_modes_match_float64(mh1_si_artifact):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Float32 Kokkos nonlinear support")
    _, model_path = mh1_si_artifact
    atoms = bulk("Si", "diamond", a=5.43, cubic=True).repeat((2, 1, 1))
    properties = ["energy", "energies", "forces", "stress"]

    reference = Symmetrix(
        model_path,
        use_kokkos=True,
        dtype="float64",
        streamed_edges="all",
    )
    reference.calculate(atoms.copy(), properties=properties)
    reference_results = {
        name: np.array(reference.results[name], copy=True) for name in properties
    }
    reference_workspace_bytes = reference.evaluator.edge_workspace_bytes

    float_results = {}
    float_workspace_bytes = None
    for mode in ("legacy", "all"):
        calculator = Symmetrix(
            model_path,
            use_kokkos=True,
            dtype="float32",
            streamed_edges=mode,
        )
        assert type(calculator.evaluator).__name__ == "MACENonlinearKokkosFloat"
        assert calculator.evaluator.scalar_size_bytes == 4
        assert calculator.evaluator.uses_mh1_fast_path
        assert calculator.evaluator.supports_streamed_edges
        assert calculator.evaluator.streamed_edges_mode == mode
        calculator.calculate(atoms.copy(), properties=properties)
        float_results[mode] = {
            name: np.array(calculator.results[name], copy=True)
            for name in properties
        }
        if mode == "all":
            float_workspace_bytes = calculator.evaluator.edge_workspace_bytes

    for mode in ("legacy", "all"):
        for name in properties:
            np.testing.assert_allclose(
                float_results[mode][name],
                reference_results[name],
                rtol=0.0,
                atol=2e-5,
            )
    for name in properties:
        np.testing.assert_allclose(
            float_results["all"][name],
            float_results["legacy"][name],
            rtol=0.0,
            atol=2e-5,
        )
    assert float_workspace_bytes <= reference_workspace_bytes * 0.51

    automatic = Symmetrix(model_path, use_kokkos=True, dtype="float32")
    assert automatic.streamed_edges == "all"


def test_mh1_kokkos_exposes_synchronized_linear_controls(mh1_si_artifact):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Float32 Kokkos nonlinear support")
    if not native_symmetrix._kokkos_is_initialized():
        native_symmetrix._init_kokkos()
    _, model_path = mh1_si_artifact
    evaluator = native_symmetrix.MACENonlinearKokkosFloat(str(model_path))
    expected_backend = (
        "packed_gemm"
        if native_symmetrix._kokkos_default_execution_space() == "Cuda"
        else "auto"
    )
    assert evaluator.e3_linear_backend == expected_backend
    assert evaluator.selected_e3_linear_backend(17) == "packed_gemm"
    assert evaluator.tensor_product_backend == "official_kokkos"
    execution_space = native_symmetrix._kokkos_default_execution_space()
    expected_tensor_execution = (
        "official_cuda_team"
        if execution_space == "Cuda"
        else "official_kokkos_mdrange"
    )
    assert evaluator.tensor_product_execution_backend == expected_tensor_execution
    assert evaluator.linear_workspace_bytes == 0
    assert evaluator.tensor_workspace_bytes == 0
    assert evaluator.precision_workspace_bytes == evaluator.edge_workspace_bytes
    evaluator.set_e3_linear_backend("scalar")
    assert evaluator.e3_linear_backend == "scalar"
    evaluator.set_e3_linear_backend("packed_gemm")
    assert evaluator.e3_linear_backend == "packed_gemm"
    with pytest.raises(ValueError, match="auto, scalar, or packed_gemm"):
        evaluator.set_e3_linear_backend("invalid")

    double_evaluator = native_symmetrix.MACENonlinearKokkos(str(model_path))
    assert (
        double_evaluator.tensor_product_execution_backend
        == "official_kokkos_mdrange"
    )
    expected_double_linear = "scalar" if execution_space == "Cuda" else "packed_gemm"
    assert double_evaluator.selected_e3_linear_backend(17) == expected_double_linear
    evaluator.fence()


def test_mh1_cuda_packed_linear_matches_scalar(mh1_si_artifact):
    if not hasattr(native_symmetrix, "MACENonlinearKokkosFloat"):
        pytest.skip("Symmetrix was built without Float32 Kokkos nonlinear support")
    if native_symmetrix._kokkos_default_execution_space() != "Cuda":
        pytest.skip("packed E3-linear regression requires Kokkos CUDA")
    _, model_path = mh1_si_artifact
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    properties = ["energy", "energies", "forces", "stress"]
    results = {}
    for backend in ("scalar", "packed_gemm"):
        calculator = Symmetrix(
            model_path,
            use_kokkos=True,
            dtype="float32",
            streamed_edges="all",
        )
        calculator.evaluator.set_e3_linear_backend(backend)
        calculator.calculate(atoms.copy(), properties=properties)
        results[backend] = {
            name: np.array(calculator.results[name], copy=True)
            for name in properties
        }
    for name in properties:
        np.testing.assert_allclose(
            results["packed_gemm"][name],
            results["scalar"][name],
            rtol=0.0,
            atol=2e-5,
        )


def test_legacy_evaluator_rejects_nonlinear_schema(mh1_si_artifact):
    _, model_path = mh1_si_artifact
    with pytest.raises((RuntimeError, ValueError), match="MACE_Nonlinear"):
        native_symmetrix.MACE(str(model_path))
