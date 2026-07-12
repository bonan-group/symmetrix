import pytest

try:
    from symmetrix.extract_mace_data import extract_mace_data
except ImportError as exc:
    extract_mace_data = None
    extract_mace_data_import_error = exc
else:
    extract_mace_data_import_error = None


@pytest.mark.skipif(extract_mace_data is None, reason=f"extract_mace_data is not available: {extract_mace_data_import_error}")
def test_macefield_extractor_includes_field_schema(macefield_model_path):
    data = extract_mace_data(
        macefield_model_path,
        species=[7, 13],
        head="mp-dielectric",
        num_spline_points=8,
    )

    assert data["model_type"] == "MACEField"
    assert data["has_field_coupling"] is True
    assert len(data["field_couplings"]) == 1

    coupling = data["field_couplings"][0]
    assert coupling["field_feats_irreps_in1"] == "128x0e+128x1o"
    assert coupling["field_feats_irreps_in2"] == "1x1o"
    assert coupling["field_feats_irreps_out"] == "128x0e+128x1o"
    assert coupling["field_linear_irreps_in"] == "128x0e+128x1o"
    assert coupling["field_linear_irreps_out"] == "128x0e+128x1o"

    assert len(coupling["field_feats_weight"]) == 32768
    assert len(coupling["field_feats_output_mask"]) == 512
    assert len(coupling["field_linear_weight"]) == 32768
    assert coupling["field_linear_bias"] == []
    assert len(coupling["field_linear_output_mask"]) == 512

    assert len(data["H1_product_weights"]) == 32768
    assert len(data["H1_linear_up_weights"]) == 32768
