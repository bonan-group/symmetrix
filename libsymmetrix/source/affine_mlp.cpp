#include "affine_mlp.hpp"

#include <cmath>
#include <stdexcept>

#include "nlohmann/json.hpp"

namespace {

std::vector<double> tensor_values(const nlohmann::json& tensor, const std::string& name)
{
    if (!tensor.contains("shape") || !tensor.contains("values"))
        throw std::invalid_argument("AffineMLP " + name + " tensor is missing shape or values.");
    return tensor.at("values").get<std::vector<double>>();
}

std::vector<int> tensor_shape(const nlohmann::json& tensor, const std::string& name)
{
    if (!tensor.contains("shape"))
        throw std::invalid_argument("AffineMLP " + name + " tensor is missing shape.");
    return tensor.at("shape").get<std::vector<int>>();
}

double silu(double value)
{
    return value/(1.0+std::exp(-value));
}

double silu_derivative(double value)
{
    const double sigmoid = 1.0/(1.0+std::exp(-value));
    return sigmoid+value*sigmoid*(1.0-sigmoid);
}

}  // namespace

AffineMLP::AffineMLP(const nlohmann::json& definition)
{
    if (!definition.contains("layers") || !definition.at("layers").is_array())
        throw std::invalid_argument("AffineMLP requires a layers array.");

    int previous_output_size = -1;
    for (const auto& definition_layer : definition.at("layers")) {
        const auto type = definition_layer.at("type").get<std::string>();
        Layer layer;
        if (type == "linear") {
            layer.type = Layer::Type::Linear;
            const auto shape = tensor_shape(definition_layer.at("weight"), "linear weight");
            if (shape.size() != 2 || shape[0] <= 0 || shape[1] <= 0)
                throw std::invalid_argument("AffineMLP linear weight must be rank two.");
            layer.output_size = shape[0];
            layer.input_size = shape[1];
            layer.weight = tensor_values(definition_layer.at("weight"), "linear weight");
            layer.bias = tensor_values(definition_layer.at("bias"), "linear bias");
            if (static_cast<int>(layer.weight.size()) != layer.output_size*layer.input_size
                || static_cast<int>(layer.bias.size()) != layer.output_size)
                throw std::invalid_argument("AffineMLP linear tensor sizes are inconsistent.");
        } else if (type == "layer_norm") {
            layer.type = Layer::Type::LayerNorm;
            const auto normalized_shape = definition_layer.at("normalized_shape").get<std::vector<int>>();
            if (normalized_shape.size() != 1 || normalized_shape[0] <= 0)
                throw std::invalid_argument("AffineMLP only supports one-dimensional LayerNorm.");
            layer.input_size = normalized_shape[0];
            layer.output_size = layer.input_size;
            layer.eps = definition_layer.at("eps").get<double>();
            layer.weight = tensor_values(definition_layer.at("weight"), "LayerNorm weight");
            layer.bias = tensor_values(definition_layer.at("bias"), "LayerNorm bias");
            if (static_cast<int>(layer.weight.size()) != layer.input_size
                || static_cast<int>(layer.bias.size()) != layer.input_size || !(layer.eps > 0.0))
                throw std::invalid_argument("AffineMLP LayerNorm tensors are inconsistent.");
        } else if (type == "silu") {
            if (previous_output_size < 0)
                throw std::invalid_argument("AffineMLP SiLU cannot be the first layer.");
            layer.type = Layer::Type::SiLU;
            layer.input_size = previous_output_size;
            layer.output_size = previous_output_size;
        } else {
            throw std::invalid_argument("AffineMLP has unsupported layer type " + type + ".");
        }
        if (previous_output_size >= 0 && layer.input_size != previous_output_size)
            throw std::invalid_argument("AffineMLP layer dimensions are inconsistent.");
        previous_output_size = layer.output_size;
        layers.push_back(std::move(layer));
    }
    if (layers.empty())
        throw std::invalid_argument("AffineMLP requires at least one layer.");
}

int AffineMLP::input_size() const
{
    for (const auto& layer : layers)
        if (layer.type != Layer::Type::SiLU)
            return layer.input_size;
    throw std::logic_error("AffineMLP has no sized layer.");
}

int AffineMLP::output_size() const
{
    for (auto it=layers.rbegin(); it != layers.rend(); ++it)
        if (it->type != Layer::Type::SiLU)
            return it->output_size;
    throw std::logic_error("AffineMLP has no sized layer.");
}

bool AffineMLP::supports_conditioned_input(int dynamic_input_size) const
{
    return !layers.empty() && layers.front().type == Layer::Type::Linear
        && dynamic_input_size > 0 && dynamic_input_size < layers.front().input_size;
}

std::vector<double> AffineMLP::first_layer_contribution(
    int input_offset,
    const std::vector<double>& values) const
{
    if (layers.empty() || layers.front().type != Layer::Type::Linear
        || input_offset < 0 || values.empty()
        || input_offset+static_cast<int>(values.size()) > layers.front().input_size)
        throw std::invalid_argument("AffineMLP first-layer contribution dimensions are inconsistent.");
    const auto& first = layers.front();
    std::vector<double> result(first.output_size, 0.0);
    for (int row=0; row<first.output_size; ++row)
        for (int column=0; column<static_cast<int>(values.size()); ++column)
            result[row] += first.weight[row*first.input_size+input_offset+column]
                *values[column];
    return result;
}

std::vector<double> AffineMLP::evaluate(const std::vector<double>& input) const
{
    return evaluate_impl(input, nullptr, nullptr, nullptr);
}

std::vector<double> AffineMLP::evaluate_conditioned(
    const std::vector<double>& input,
    const std::vector<double>& first_contribution,
    const std::vector<double>& second_contribution) const
{
    return evaluate_impl(input, &first_contribution, &second_contribution, nullptr);
}

std::vector<double> AffineMLP::evaluate_impl(
    const std::vector<double>& input,
    const std::vector<double>* first_contribution,
    const std::vector<double>* second_contribution,
    std::vector<std::vector<double>>* tape) const
{
    const bool conditioned = first_contribution != nullptr;
    if ((!conditioned && static_cast<int>(input.size()) != input_size())
        || (conditioned && !supports_conditioned_input(input.size())))
        throw std::invalid_argument("AffineMLP input size does not match the first layer.");
    if (conditioned) {
        const int width = layers.front().output_size;
        if (static_cast<int>(first_contribution->size()) != width
            || second_contribution == nullptr
            || static_cast<int>(second_contribution->size()) != width)
            throw std::invalid_argument("AffineMLP conditioned contributions have invalid dimensions.");
    }
    auto values = input;
    if (tape) {
        tape->clear();
        tape->reserve(layers.size()+1);
        tape->push_back(values);
    }
    for (int layer_index=0; layer_index<static_cast<int>(layers.size()); ++layer_index) {
        const auto& layer = layers[layer_index];
        if (layer.type == Layer::Type::Linear) {
            const int active_input_size = conditioned && layer_index == 0
                ? static_cast<int>(values.size()) : layer.input_size;
            if (static_cast<int>(values.size()) != active_input_size)
                throw std::logic_error("AffineMLP linear input size is inconsistent.");
            auto output = layer.bias;
            if (conditioned && layer_index == 0)
                for (int row=0; row<layer.output_size; ++row)
                    output[row] += (*first_contribution)[row]+(*second_contribution)[row];
            for (int row=0; row<layer.output_size; ++row)
                for (int column=0; column<active_input_size; ++column)
                    output[row] += layer.weight[row*layer.input_size+column]*values[column];
            values = std::move(output);
        } else if (layer.type == Layer::Type::LayerNorm) {
            double mean = 0.0;
            for (const auto value : values)
                mean += value;
            mean /= values.size();
            double variance = 0.0;
            for (const auto value : values)
                variance += (value-mean)*(value-mean);
            variance /= values.size();
            const double inverse_stddev = 1.0/std::sqrt(variance+layer.eps);
            for (int index=0; index<layer.output_size; ++index)
                values[index] = (values[index]-mean)*inverse_stddev*layer.weight[index]+layer.bias[index];
        } else {
            for (auto& value : values)
                value = silu(value);
        }
        if (tape) tape->push_back(values);
    }
    return values;
}

std::vector<double> AffineMLP::evaluate_gradient(
    const std::vector<double>& input,
    const std::vector<double>& output_adjoint) const
{
    return evaluate_gradient_impl(input, nullptr, nullptr, output_adjoint);
}

std::vector<double> AffineMLP::evaluate_gradient_conditioned(
    const std::vector<double>& input,
    const std::vector<double>& first_contribution,
    const std::vector<double>& second_contribution,
    const std::vector<double>& output_adjoint) const
{
    return evaluate_gradient_impl(
        input, &first_contribution, &second_contribution, output_adjoint);
}

std::vector<double> AffineMLP::evaluate_gradient_impl(
    const std::vector<double>& input,
    const std::vector<double>* first_contribution,
    const std::vector<double>* second_contribution,
    const std::vector<double>& output_adjoint) const
{
    if (static_cast<int>(output_adjoint.size()) != output_size())
        throw std::invalid_argument("AffineMLP gradient inputs have invalid dimensions.");
    std::vector<std::vector<double>> values;
    evaluate_impl(input, first_contribution, second_contribution, &values);

    auto adjoint = output_adjoint;
    for (int layer_index=static_cast<int>(layers.size())-1; layer_index>=0; --layer_index) {
        const auto& layer = layers[layer_index];
        const auto& layer_input = values[layer_index];
        if (layer.type == Layer::Type::Linear) {
            const int active_input_size = layer_index == 0 && first_contribution != nullptr
                ? static_cast<int>(input.size()) : layer.input_size;
            auto input_adjoint = std::vector<double>(active_input_size, 0.0);
            for (int row=0; row<layer.output_size; ++row)
                for (int column=0; column<active_input_size; ++column)
                    input_adjoint[column] += layer.weight[row*layer.input_size+column]*adjoint[row];
            adjoint = std::move(input_adjoint);
        } else if (layer.type == Layer::Type::LayerNorm) {
            const int width = layer.input_size;
            double mean = 0.0;
            for (const auto value : layer_input)
                mean += value;
            mean /= width;
            double variance = 0.0;
            for (const auto value : layer_input)
                variance += (value-mean)*(value-mean);
            variance /= width;
            const double inverse_stddev = 1.0/std::sqrt(variance+layer.eps);
            auto normalized = std::vector<double>(width);
            auto scaled_adjoint = std::vector<double>(width);
            double sum_scaled = 0.0;
            double sum_scaled_normalized = 0.0;
            for (int index=0; index<width; ++index) {
                normalized[index] = (layer_input[index]-mean)*inverse_stddev;
                scaled_adjoint[index] = adjoint[index]*layer.weight[index];
                sum_scaled += scaled_adjoint[index];
                sum_scaled_normalized += scaled_adjoint[index]*normalized[index];
            }
            for (int index=0; index<width; ++index)
                adjoint[index] = inverse_stddev*(
                    width*scaled_adjoint[index]-sum_scaled-normalized[index]*sum_scaled_normalized
                )/width;
        } else {
            for (int index=0; index<static_cast<int>(adjoint.size()); ++index)
                adjoint[index] *= silu_derivative(layer_input[index]);
        }
    }
    return adjoint;
}
