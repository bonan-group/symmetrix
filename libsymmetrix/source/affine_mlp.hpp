#pragma once

#include <string>
#include <vector>

#include "nlohmann/json.hpp"

class AffineMLP {
public:
    AffineMLP() = default;
    explicit AffineMLP(const nlohmann::json& definition);

    int input_size() const;
    int output_size() const;
    std::vector<double> evaluate(const std::vector<double>& input) const;
    std::vector<double> evaluate_gradient(
        const std::vector<double>& input,
        const std::vector<double>& output_adjoint) const;

private:
    struct Layer {
        enum class Type { Linear, LayerNorm, SiLU };
        Type type;
        int input_size = 0;
        int output_size = 0;
        double eps = 0.0;
        std::vector<double> weight;
        std::vector<double> bias;
    };

    std::vector<Layer> layers;
};
