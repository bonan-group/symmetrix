#include "mace_nonlinear_schema.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "e3nn.hpp"

namespace {

const nlohmann::json& tensor(
    const nlohmann::json& parent, const char* key, const char* name)
{
    const auto& value = parent.at(key);
    if (!value.is_object() || !value.contains("shape") || !value.contains("values")
        || !value.at("shape").is_array() || !value.at("values").is_array())
        throw std::invalid_argument(
            std::string("MACE_Nonlinear ") + name + " tensor is malformed.");
    std::size_t size = 1;
    for (const auto& extent_value : value.at("shape")) {
        const int extent = extent_value.get<int>();
        if (extent < 0
            || (extent != 0
                && size > std::numeric_limits<std::size_t>::max()
                    / static_cast<std::size_t>(extent)))
            throw std::invalid_argument(
                std::string("MACE_Nonlinear ") + name + " tensor shape is invalid.");
        size *= static_cast<std::size_t>(extent);
    }
    if (size != value.at("values").size())
        throw std::invalid_argument(
            std::string("MACE_Nonlinear ") + name + " tensor shape is invalid.");
    return value;
}

void require_finite(double value, const char* name)
{
    if (!std::isfinite(value))
        throw std::invalid_argument(std::string("MACE_Nonlinear ") + name + " must be finite.");
}

bool irreps_equal(const std::string& actual, const std::string& expected)
{
    const Irreps left(actual);
    const Irreps right(expected);
    if (left.blocks.size() != right.blocks.size()) return false;
    for (int index=0; index<static_cast<int>(left.blocks.size()); ++index) {
        const auto& a = left.blocks[index];
        const auto& b = right.blocks[index];
        if (a.multiplicity != b.multiplicity || a.l != b.l || a.parity != b.parity)
            return false;
    }
    return true;
}

} // namespace

bool is_published_mh1_architecture(const nlohmann::json& data)
{
    if (data.at("num_interactions").get<int>() != 2
        || data.at("l_max").get<int>() != 3
        || data.at("radial_embedding").at("basis").at("weights").at("values").size() != 10
        || data.at("interactions").size() != 2
        || data.at("products").size() != 2
        || data.at("readouts").size() != 2)
        return false;
    const auto& first = data.at("interactions").at(0);
    const auto& second = data.at("interactions").at(1);
    if (first.at("class").get<std::string>()
            != "RealAgnosticResidualNonLinearInteractionBlock"
        || second.at("class").get<std::string>()
            != "RealAgnosticResidualNonLinearInteractionBlock"
        || !irreps_equal(first.at("node_feats_irreps").get<std::string>(), "512x0e")
        || !irreps_equal(second.at("node_feats_irreps").get<std::string>(), "512x0e+512x1o")
        || !irreps_equal(first.at("edge_irreps").get<std::string>(), "128x0e")
        || !irreps_equal(second.at("edge_irreps").get<std::string>(), "128x0e+128x1o")
        || !irreps_equal(first.at("target_irreps").get<std::string>(),
                         "512x0e+512x1o+512x2e+512x3o")
        || !irreps_equal(second.at("target_irreps").get<std::string>(),
                         "512x0e+512x1o+512x2e+512x3o")
        || !irreps_equal(first.at("hidden_irreps").get<std::string>(), "512x0e+512x1o")
        || !irreps_equal(second.at("hidden_irreps").get<std::string>(), "512x0e"))
        return false;
    const std::string gate_input = "2048x0e+512x1o+512x2e+512x3o";
    const std::string gate_output = "512x0e+512x1o+512x2e+512x3o";
    for (const auto* interaction : {&first, &second})
        if (!irreps_equal(
                interaction->at("gate").at("irreps_in").get<std::string>(), gate_input)
            || !irreps_equal(
                interaction->at("gate").at("irreps_out").get<std::string>(), gate_output))
            return false;
    return data.at("readouts").at(0).at("class").get<std::string>()
            == "LinearReadoutBlock"
        && data.at("readouts").at(1).at("class").get<std::string>()
            == "NonLinearReadoutBlock";
}

void validate_mace_nonlinear_schema(const nlohmann::json& data)
{
    if (!data.is_object()
        || data.value("model_type", std::string()) != "MACE_Nonlinear"
        || data.at("symmetrix_format_version").get<int>() != 3)
        throw std::invalid_argument(
            "MACE_Nonlinear evaluator requires format-version-3 nonlinear JSON.");

    const auto atomic_numbers = data.at("atomic_numbers").get<std::vector<int>>();
    const auto model_atomic_numbers =
        data.at("model_atomic_numbers").get<std::vector<int>>();
    const auto model_indices = data.at("model_indices").get<std::vector<int>>();
    if (atomic_numbers.empty() || model_atomic_numbers.empty()
        || model_indices.size() != atomic_numbers.size()
        || data.at("num_elements").get<int>() != static_cast<int>(atomic_numbers.size())
        || std::set<int>(atomic_numbers.begin(), atomic_numbers.end()).size()
            != atomic_numbers.size()
        || std::set<int>(model_atomic_numbers.begin(), model_atomic_numbers.end()).size()
            != model_atomic_numbers.size())
        throw std::invalid_argument("MACE_Nonlinear atomic-number metadata is inconsistent.");
    if (std::any_of(atomic_numbers.begin(), atomic_numbers.end(), [](int value) { return value <= 0; })
        || std::any_of(model_atomic_numbers.begin(), model_atomic_numbers.end(), [](int value) { return value <= 0; }))
        throw std::invalid_argument("MACE_Nonlinear atomic numbers must be positive.");
    for (int local_type=0; local_type<static_cast<int>(atomic_numbers.size()); ++local_type) {
        const int model_type = model_indices[local_type];
        if (model_type < 0 || model_type >= static_cast<int>(model_atomic_numbers.size())
            || model_atomic_numbers[model_type] != atomic_numbers[local_type])
            throw std::invalid_argument("MACE_Nonlinear model_indices are inconsistent.");
    }

    const double r_cut = data.at("r_cut").get<double>();
    const int l_max = data.at("l_max").get<int>();
    if (!(r_cut > 0.0) || !std::isfinite(r_cut) || l_max < 0
        || l_max >= static_cast<int>(std::sqrt(std::numeric_limits<int>::max()))-1)
        throw std::invalid_argument("MACE_Nonlinear cutoff or l_max is invalid.");

    const int num_interactions = data.at("num_interactions").get<int>();
    if (num_interactions <= 0 || !data.at("interactions").is_array()
        || !data.at("products").is_array() || !data.at("readouts").is_array()
        || data.at("interactions").size() != static_cast<std::size_t>(num_interactions)
        || data.at("products").size() != static_cast<std::size_t>(num_interactions)
        || data.at("readouts").size() != static_cast<std::size_t>(num_interactions))
        throw std::invalid_argument("MACE_Nonlinear layer counts are inconsistent.");

    const auto& radial = data.at("radial_embedding");
    const auto& basis = radial.at("basis");
    const auto& cutoff = radial.at("cutoff");
    if (basis.at("type").get<std::string>() != "bessel"
        || cutoff.at("type").get<std::string>() != "polynomial"
        || cutoff.at("p").get<int>() < 1
        || cutoff.at("r_max").get<double>() != r_cut)
        throw std::invalid_argument("MACE_Nonlinear radial basis or cutoff is unsupported.");
    const auto& basis_weights = tensor(basis, "weights", "radial basis weights");
    if (basis_weights.at("shape").size() != 1 || basis_weights.at("values").empty())
        throw std::invalid_argument("MACE_Nonlinear radial basis weights are invalid.");
    require_finite(basis.at("prefactor").get<double>(), "radial prefactor");

    const auto& transform = radial.at("distance_transform");
    const auto transform_type = transform.at("type").get<std::string>();
    if (transform_type != "none" && transform_type != "agnesi")
        throw std::invalid_argument("MACE_Nonlinear distance transform is unsupported.");
    if (transform_type == "agnesi") {
        require_finite(transform.at("a").get<double>(), "Agnesi a");
        require_finite(transform.at("q").get<double>(), "Agnesi q");
        require_finite(transform.at("p").get<double>(), "Agnesi p");
        const auto radii = transform.at("covalent_radii").get<std::vector<double>>();
        const int maximum_atomic_number =
            *std::max_element(model_atomic_numbers.begin(), model_atomic_numbers.end());
        if (maximum_atomic_number < 0
            || radii.size() <= static_cast<std::size_t>(maximum_atomic_number))
            throw std::invalid_argument("MACE_Nonlinear Agnesi covalent radii are incomplete.");
        for (int atomic_number : model_atomic_numbers)
            if (!(radii[atomic_number] > 0.0) || !std::isfinite(radii[atomic_number]))
                throw std::invalid_argument("MACE_Nonlinear Agnesi covalent radii are invalid.");
    }

    const auto& scale = tensor(data.at("scale_shift"), "scale", "scale");
    const auto& shift = tensor(data.at("scale_shift"), "shift", "shift");
    if (!scale.at("shape").empty() || !shift.at("shape").empty())
        throw std::invalid_argument("MACE_Nonlinear scale and shift must be scalar tensors.");
    require_finite(scale.at("values").at(0).get<double>(), "scale");
    require_finite(shift.at("values").at(0).get<double>(), "shift");

    const auto& atomic_energies = tensor(data, "atomic_energies", "atomic energies");
    const auto energy_shape = atomic_energies.at("shape").get<std::vector<int>>();
    if (energy_shape != std::vector<int>{1, static_cast<int>(model_atomic_numbers.size())})
        throw std::invalid_argument("MACE_Nonlinear atomic-energy tensor has an invalid shape.");
}
