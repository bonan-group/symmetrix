#include "e3nn_product.hpp"

#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace {

E3ProductBasis::Tensor tensor_from_json(const nlohmann::json& value)
{
    E3ProductBasis::Tensor tensor;
    tensor.shape = value.at("shape").get<std::vector<int>>();
    tensor.values = value.at("values").get<std::vector<double>>();
    std::size_t size = 1;
    for (int dimension : tensor.shape) {
        if (dimension <= 0
            || size > std::numeric_limits<std::size_t>::max()/static_cast<std::size_t>(dimension))
            throw std::invalid_argument("MACE_Nonlinear product tensor has an invalid shape.");
        size *= static_cast<std::size_t>(dimension);
    }
    if (size != tensor.values.size())
        throw std::invalid_argument("MACE_Nonlinear product tensor has an invalid shape.");
    return tensor;
}

} // namespace

int E3ProductBasis::Tensor::offset(const std::vector<int>& index) const
{
    if (index.size() != shape.size())
        throw std::invalid_argument("MACE_Nonlinear product tensor rank mismatch.");
    int result = 0;
    for (int axis=0; axis<static_cast<int>(shape.size()); ++axis) {
        if (index[axis] < 0 || index[axis] >= shape[axis])
            throw std::out_of_range("MACE_Nonlinear product tensor index is out of range.");
        result = result * shape[axis] + index[axis];
    }
    return result;
}

E3ProductBasis::E3ProductBasis(const nlohmann::json& data)
    : input(data.at("symmetric_contractions").at("irreps_in").get<std::string>()),
      output(data.at("symmetric_contractions").at("irreps_out").get<std::string>()),
      linear(data.at("linear")),
      use_sc(data.at("use_sc").get<bool>()),
      num_features(input.blocks.front().multiplicity),
      angular_dimension(0)
{
    if (linear.input_dimension() != output.dimension() || linear.output_dimension() != output.dimension())
        throw std::invalid_argument("MACE_Nonlinear product linear has incompatible irreps.");
    for (const auto& block : input.blocks) {
        if (block.multiplicity != num_features)
            throw std::invalid_argument("MACE_Nonlinear product requires a common feature multiplicity.");
        angular_dimension += 2*block.l + 1;
    }
    for (const auto& block : output.blocks)
        if (block.multiplicity != num_features)
            throw std::invalid_argument("MACE_Nonlinear product output multiplicity is incompatible.");
    const auto& serialized = data.at("symmetric_contractions").at("contractions");
    if (serialized.size() != output.blocks.size())
        throw std::invalid_argument("MACE_Nonlinear product has incompatible contraction outputs.");
    for (int contraction_index=0;
         contraction_index<static_cast<int>(serialized.size()); ++contraction_index) {
        const auto& value = serialized.at(contraction_index);
        const auto& output_block = output.blocks[contraction_index];
        Contraction contraction;
        contraction.correlation = value.at("correlation").get<int>();
        contraction.weights_max = tensor_from_json(value.at("weights_max"));
        for (const auto& weight : value.at("weights"))
            contraction.weights.push_back(tensor_from_json(weight));
        for (const auto& u : value.at("u_tensors"))
            contraction.u_tensors.push_back(tensor_from_json(u));
        if (contraction.correlation < 1
            || static_cast<int>(contraction.u_tensors.size()) != contraction.correlation
            || static_cast<int>(contraction.weights.size()) != contraction.correlation-1)
            throw std::invalid_argument("MACE_Nonlinear product has invalid correlation data.");
        int num_elements = -1;
        for (int degree=1; degree<=contraction.correlation; ++degree) {
            const auto& u = contraction.u_tensors[degree-1];
            const auto& weights = degree == contraction.correlation
                ? contraction.weights_max
                : contraction.weights[contraction.correlation-degree-1];
            const int output_axes = output_block.l == 0 ? 0 : 1;
            if (static_cast<int>(u.shape.size()) != degree+output_axes+1
                || (output_axes && u.shape[0] != 2*output_block.l+1)
                || weights.shape.size() != 3
                || weights.shape[1] != u.shape.back()
                || weights.shape[2] != num_features)
                throw std::invalid_argument("Unsupported MACE_Nonlinear product tensor layout.");
            for (int axis=0; axis<degree; ++axis)
                if (u.shape[output_axes+axis] != angular_dimension)
                    throw std::invalid_argument("Unsupported MACE_Nonlinear product angular tensor layout.");
            if (num_elements < 0)
                num_elements = weights.shape[0];
            else if (weights.shape[0] != num_elements)
                throw std::invalid_argument("MACE_Nonlinear product element dimensions are inconsistent.");
        }
        contractions.push_back(std::move(contraction));
    }
}

std::vector<double> E3ProductBasis::make_feature_major(const std::vector<double>& node_features) const
{
    if (static_cast<int>(node_features.size()) != input.dimension())
        throw std::invalid_argument("MACE_Nonlinear product feature size is invalid.");
    std::vector<double> feature_major(num_features*angular_dimension);
    int angular_offset = 0;
    for (const auto& block : input.blocks) {
        const int width = 2*block.l+1;
        for (int feature=0; feature<num_features; ++feature)
            for (int component=0; component<width; ++component)
                feature_major[feature*angular_dimension+angular_offset+component]
                    = node_features[block.offset+feature*width+component];
        angular_offset += width;
    }
    return feature_major;
}

std::vector<double> E3ProductBasis::make_irrep_major(const std::vector<double>& feature_major) const
{
    std::vector<double> result(output.dimension(), 0.0);
    int angular_offset = 0;
    for (const auto& block : output.blocks) {
        const int width = 2*block.l+1;
        for (int feature=0; feature<num_features; ++feature)
            for (int component=0; component<width; ++component)
                result[block.offset+feature*width+component]
                    = feature_major[feature*angular_dimension+angular_offset+component];
        angular_offset += width;
    }
    return result;
}

double E3ProductBasis::evaluate_term(
    const Tensor& u,
    const Tensor& weights,
    const std::vector<double>& feature_major,
    int output_component,
    int feature,
    int element) const
{
    const int output_axes = output_component >= 0 ? 1 : 0;
    const int degree = static_cast<int>(u.shape.size()) - output_axes - 1;
    if (degree < 1 || u.shape.back() != weights.shape[1]
        || u.shape[output_axes] != angular_dimension
        || weights.shape.size() != 3 || feature >= weights.shape[2])
        throw std::invalid_argument("Unsupported MACE_Nonlinear product tensor layout.");
    const int parameters = u.shape.back();
    int tuples = 1;
    for (int axis=0; axis<degree; ++axis) tuples *= u.shape[output_axes+axis];
    const int u_component_offset = output_component < 0 ? 0 : output_component*tuples*parameters;
    const int weight_element_offset = element*parameters*num_features;
    double sum = 0.0;
    for (int tuple=0; tuple<tuples; ++tuple) {
        int remainder = tuple;
        double monomial = 1.0;
        for (int axis=degree-1; axis>=0; --axis) {
            const int dimension = u.shape[output_axes+axis];
            const int index = remainder%dimension;
            remainder /= dimension;
            monomial *= feature_major[feature*angular_dimension+index];
        }
        double coefficient = 0.0;
        const int u_offset = u_component_offset+tuple*parameters;
        for (int parameter=0; parameter<parameters; ++parameter)
            coefficient += u.values[u_offset+parameter]
                * weights.values[weight_element_offset+parameter*num_features+feature];
        sum += coefficient*monomial;
    }
    return sum;
}

void E3ProductBasis::reverse_term(
    const Tensor& u,
    const Tensor& weights,
    const std::vector<double>& feature_major,
    int output_component,
    int feature,
    int element,
    double output_adjoint,
    std::vector<double>& feature_major_adjoint) const
{
    const int output_axes = output_component >= 0 ? 1 : 0;
    const int degree = static_cast<int>(u.shape.size()) - output_axes - 1;
    const int parameters = u.shape.back();
    int tuples = 1;
    for (int axis=0; axis<degree; ++axis) tuples *= u.shape[output_axes+axis];
    const int u_component_offset = output_component < 0 ? 0 : output_component*tuples*parameters;
    const int weight_element_offset = element*parameters*num_features;
    std::vector<int> indices(degree);
    for (int tuple=0; tuple<tuples; ++tuple) {
        int remainder = tuple;
        for (int axis=degree-1; axis>=0; --axis) {
            const int dimension = u.shape[output_axes+axis];
            indices[axis] = remainder%dimension;
            remainder /= dimension;
        }
        double coefficient = 0.0;
        const int u_offset = u_component_offset+tuple*parameters;
        for (int parameter=0; parameter<parameters; ++parameter)
            coefficient += u.values[u_offset+parameter]
                * weights.values[weight_element_offset+parameter*num_features+feature];
        coefficient *= output_adjoint;
        for (int differentiated=0; differentiated<degree; ++differentiated) {
            double derivative = coefficient;
            for (int factor=0; factor<degree; ++factor)
                if (factor != differentiated)
                    derivative *= feature_major[feature*angular_dimension+indices[factor]];
            feature_major_adjoint[feature*angular_dimension+indices[differentiated]] += derivative;
        }
    }
}

std::vector<double> E3ProductBasis::evaluate(
    const std::vector<double>& node_features,
    const std::vector<double>& skip_connection,
    int element) const
{
    const auto features = make_feature_major(node_features);
    std::vector<double> product_major(num_features*angular_dimension, 0.0);
    int output_angular_offset = 0;
    for (int block_index=0; block_index<static_cast<int>(output.blocks.size()); ++block_index) {
        const auto& block = output.blocks[block_index];
        const auto& contraction = contractions[block_index];
        if (element < 0 || element >= contraction.weights_max.shape[0])
            throw std::out_of_range("MACE_Nonlinear product element index is out of range.");
        for (int feature=0; feature<num_features; ++feature)
            for (int component=0; component<2*block.l+1; ++component)
                for (int degree=1; degree<=contraction.correlation; ++degree) {
                    const auto& u = contraction.u_tensors[degree-1];
                    const auto& weights = degree == contraction.correlation
                        ? contraction.weights_max
                        : contraction.weights[contraction.correlation-degree-1];
                    product_major[feature*angular_dimension+output_angular_offset+component]
                        += evaluate_term(u, weights, features,
                                         block.l == 0 ? -1 : component, feature, element);
                }
        output_angular_offset += 2*block.l+1;
    }
    auto result = linear.evaluate(make_irrep_major(product_major));
    if (use_sc) {
        if (skip_connection.size() != result.size())
            throw std::invalid_argument("MACE_Nonlinear product skip connection size is invalid.");
        for (int i=0; i<static_cast<int>(result.size()); ++i)
            result[i] += skip_connection[i];
    }
    return result;
}

void E3ProductBasis::reverse(
    const std::vector<double>& node_features,
    int element,
    const std::vector<double>& output_adjoint,
    std::vector<double>& node_features_adjoint,
    std::vector<double>& skip_connection_adjoint) const
{
    if (static_cast<int>(output_adjoint.size()) != output.dimension())
        throw std::invalid_argument("MACE_Nonlinear product output adjoint size is invalid.");
    std::vector<double> contracted_adjoint;
    linear.reverse(output_adjoint, contracted_adjoint);
    const auto features = make_feature_major(node_features);
    std::vector<double> feature_adjoint(num_features*angular_dimension, 0.0);
    int output_offset = 0;
    for (int block_index=0; block_index<static_cast<int>(output.blocks.size()); ++block_index) {
        const auto& block = output.blocks[block_index];
        const auto& contraction = contractions[block_index];
        const int width = 2*block.l+1;
        for (int feature=0; feature<num_features; ++feature)
            for (int component=0; component<width; ++component) {
                const double adjoint = contracted_adjoint[block.offset+feature*width+component];
                for (int degree=1; degree<=contraction.correlation; ++degree) {
                    const auto& weights = degree == contraction.correlation
                        ? contraction.weights_max
                        : contraction.weights[contraction.correlation-degree-1];
                    reverse_term(contraction.u_tensors[degree-1], weights, features,
                                 block.l == 0 ? -1 : component, feature, element,
                                 adjoint, feature_adjoint);
                }
            }
        output_offset += width;
    }
    node_features_adjoint.assign(input.dimension(), 0.0);
    int angular_offset = 0;
    for (const auto& block : input.blocks) {
        const int width = 2*block.l+1;
        for (int feature=0; feature<num_features; ++feature)
            for (int component=0; component<width; ++component)
                node_features_adjoint[block.offset+feature*width+component]
                    = feature_adjoint[feature*angular_dimension+angular_offset+component];
        angular_offset += width;
    }
    skip_connection_adjoint = use_sc
        ? output_adjoint
        : std::vector<double>(output.dimension(), 0.0);
}
