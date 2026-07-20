#pragma once

#include <vector>

#include "e3nn.hpp"

// Product-basis evaluation used by EquivariantProductBasisBlock.  The MACE
// schema stores the U tensors emitted by mace.tools.cg, allowing this class to
// evaluate arbitrary correlation order without Python or e3nn at runtime.
class E3ProductBasis {
public:
    struct Tensor {
        std::vector<int> shape;
        std::vector<double> values;
        int offset(const std::vector<int>& index) const;
    };

    explicit E3ProductBasis(const nlohmann::json& data);

    int input_dimension() const { return input.dimension(); }
    int output_dimension() const { return output.dimension(); }
    std::vector<double> evaluate(
        const std::vector<double>& node_features,
        const std::vector<double>& skip_connection,
        int element) const;
    void reverse(
        const std::vector<double>& node_features,
        int element,
        const std::vector<double>& output_adjoint,
        std::vector<double>& node_features_adjoint,
        std::vector<double>& skip_connection_adjoint) const;

private:
    struct Contraction {
        int correlation = 0;
        std::vector<Tensor> weights;
        Tensor weights_max;
        std::vector<Tensor> u_tensors;
    };

    std::vector<double> make_feature_major(const std::vector<double>& node_features) const;
    std::vector<double> make_irrep_major(const std::vector<double>& feature_major) const;
    double evaluate_term(
        const Tensor& u,
        const Tensor& weights,
        const std::vector<double>& feature_major,
        int output_component,
        int feature,
        int element) const;
    void reverse_term(
        const Tensor& u,
        const Tensor& weights,
        const std::vector<double>& feature_major,
        int output_component,
        int feature,
        int element,
        double output_adjoint,
        std::vector<double>& feature_major_adjoint) const;

    Irreps input;
    Irreps output;
    E3Linear linear;
    bool use_sc;
    int num_features;
    int angular_dimension;
    std::vector<Contraction> contractions;
};
