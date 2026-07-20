import json
import os
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

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
    actual = Symmetrix(path, use_kokkos=False, dtype="float64")
    actual.calculate(atoms.copy(), properties=["energy", "forces"])
    assert actual.results["energy"] == pytest.approx(expected.results["energy"], abs=2e-5)
    assert np.allclose(actual.results["forces"], expected.results["forces"], atol=2e-5)


def test_mh1_universal_extraction_preserves_all_species(tmp_path):
    data = extract_mace_data(_mh1_model_path(), head="matpes_r2scan")
    assert data["atomic_numbers"] == data["model_atomic_numbers"]
    assert data["model_indices"] == list(range(len(data["model_atomic_numbers"])))
    assert len(data["atomic_numbers"]) == 89
    path = tmp_path / "mh1-universal.json"
    path.write_text(json.dumps(data, separators=(",", ":")))
    evaluator = native_symmetrix.MACENonlinear(str(path))
    assert evaluator.uses_mh1_fast_path


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


def test_mh1_serial_fast_path_requires_exact_architecture(mh1_si_artifact, tmp_path):
    data, _ = mh1_si_artifact
    changed = json.loads(json.dumps(data))
    changed["interactions"][0]["hidden_irreps"] = "512x0e"
    path = tmp_path / "near-mh1.json"
    path.write_text(json.dumps(changed))
    evaluator = native_symmetrix.MACENonlinear(str(path))
    assert not evaluator.uses_mh1_fast_path


def test_mh1_serial_fast_path_accepts_equivalent_irrep_formatting(
    mh1_si_artifact, tmp_path
):
    data, _ = mh1_si_artifact
    changed = json.loads(json.dumps(data))
    changed["interactions"][1]["node_feats_irreps"] = " 512x0e + 512x1o "
    changed["interactions"][0]["gate"]["irreps_out"] = (
        " 512x0e + 512x1o + 512x2e + 512x3o "
    )
    path = tmp_path / "mh1-equivalent-irreps.json"
    path.write_text(json.dumps(changed, separators=(",", ":")))
    evaluator = native_symmetrix.MACENonlinear(str(path))
    assert evaluator.uses_mh1_fast_path


def test_mh1_serial_fast_path_requires_conditionable_edge_mlp(mh1_si_artifact, tmp_path):
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
    evaluator = native_symmetrix.MACENonlinear(str(path))
    assert not evaluator.uses_mh1_fast_path


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
        if not use_kokkos:
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


def test_mh1_kokkos_repeated_calculations(mh1_si_artifact):
    _, model_path = mh1_si_artifact
    serial = Symmetrix(model_path, use_kokkos=False, dtype="float64")
    kokkos = Symmetrix(model_path, use_kokkos=True, dtype="float64")
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
        assert kokkos.results["energy"] == pytest.approx(serial.results["energy"], abs=2e-12)
        assert np.allclose(kokkos.results["forces"], serial.results["forces"], atol=2e-12)
        assert np.allclose(kokkos.results["stress"], serial.results["stress"], atol=2e-12)
        if first_energy is None:
            first_energy = kokkos.results["energy"]
        elif displacement != 0.0:
            assert kokkos.results["energy"] != pytest.approx(first_energy, abs=1e-8)


def test_mh1_kokkos_handles_changes_in_supported_species(mh1_h_si_artifact):
    _, model_path = mh1_h_si_artifact
    serial = Symmetrix(model_path, use_kokkos=False, dtype="float64")
    kokkos = Symmetrix(model_path, use_kokkos=True, dtype="float64")
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
        assert kokkos.results["energy"] == pytest.approx(serial.results["energy"], abs=2e-12)
        assert np.allclose(kokkos.results["forces"], serial.results["forces"], atol=2e-12)
        assert np.allclose(kokkos.results["stress"], serial.results["stress"], atol=2e-12)


@pytest.mark.parametrize("use_kokkos", [False, True])
def test_mh1_native_force_matches_finite_difference(mh1_si_artifact, use_kokkos):
    _, model_path = mh1_si_artifact
    atoms = Atoms(
        "Si2",
        positions=[[0.0, 0.0, 0.0], [2.2, 0.1, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=False,
    )
    atoms.calc = Symmetrix(model_path, use_kokkos=use_kokkos, dtype="float64")
    force = atoms.get_forces()[1, 0]
    step = 1e-4
    displaced = atoms.copy()
    displaced.calc = atoms.calc
    displaced.positions[1, 0] += step
    energy_plus = displaced.get_potential_energy()
    displaced.positions[1, 0] -= 2 * step
    energy_minus = displaced.get_potential_energy()
    assert force == pytest.approx(-(energy_plus - energy_minus) / (2 * step), abs=2e-4)


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


def test_mh1_tensor_product_forward_and_reverse_match_autograd(mh1_si_artifact):
    data, _ = mh1_si_artifact
    model = torch.load(
        _mh1_model_path(), map_location=torch.device("cpu"), weights_only=False
    ).to(torch.float64)
    model = remove_pt_head(model, "matpes_r2scan")
    torch_module = model.interactions[0].conv_tp
    native_module = native_symmetrix.E3TensorProduct(
        json.dumps(data["interactions"][0]["conv_tp"])
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


def test_mh1_float32_is_rejected(mh1_si_artifact):
    _, model_path = mh1_si_artifact
    with pytest.raises(ValueError, match="require dtype 'float64'"):
        Symmetrix(model_path, dtype="float32")


def test_legacy_evaluator_rejects_nonlinear_schema(mh1_si_artifact):
    _, model_path = mh1_si_artifact
    with pytest.raises((RuntimeError, ValueError), match="MACE_Nonlinear"):
        native_symmetrix.MACE(str(model_path))
