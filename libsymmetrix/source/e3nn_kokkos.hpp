#pragma once

#include <Kokkos_Core.hpp>

#include "e3nn.hpp"

class E3LinearKokkos {
public:
    E3LinearKokkos() = default;
    explicit E3LinearKokkos(const nlohmann::json& data);
    int input_dimension() const { return input_dimension_; }
    int output_dimension() const { return output_dimension_; }
    void evaluate(Kokkos::View<const double**,Kokkos::LayoutRight> input,
                  Kokkos::View<double**,Kokkos::LayoutRight> output) const;
    void reverse(Kokkos::View<const double**,Kokkos::LayoutRight> output_adjoint,
                 Kokkos::View<double**,Kokkos::LayoutRight> input_adjoint) const;

private:
    struct Instruction {
        int input_offset, output_offset, input_multiplicity, output_multiplicity, width, weight_offset;
        double path_weight;
    };
    int input_dimension_=0, output_dimension_=0;
    std::vector<Instruction> instructions;
    Kokkos::View<double*> weights, bias, output_mask;
};

class E3TensorProductKokkos {
public:
    E3TensorProductKokkos() = default;
    explicit E3TensorProductKokkos(const nlohmann::json& data);
    int input_1_dimension() const { return input_1_dimension_; }
    int input_2_dimension() const { return input_2_dimension_; }
    int output_dimension() const { return output_dimension_; }
    int weight_size() const { return weight_size_; }
    void evaluate(Kokkos::View<const double**,Kokkos::LayoutRight> input_1,
                  Kokkos::View<const double**,Kokkos::LayoutRight> input_2,
                  Kokkos::View<const double**,Kokkos::LayoutRight> weights,
                  Kokkos::View<double**,Kokkos::LayoutRight> output) const;
    void reverse(Kokkos::View<const double**,Kokkos::LayoutRight> input_1,
                 Kokkos::View<const double**,Kokkos::LayoutRight> input_2,
                 Kokkos::View<const double**,Kokkos::LayoutRight> weights,
                 Kokkos::View<const double**,Kokkos::LayoutRight> output_adjoint,
                 Kokkos::View<double**,Kokkos::LayoutRight> input_1_adjoint,
                 Kokkos::View<double**,Kokkos::LayoutRight> input_2_adjoint,
                 Kokkos::View<double**,Kokkos::LayoutRight> weights_adjoint) const;

private:
    struct Instruction {
        int input_1_offset, input_2_offset, output_offset;
        int multiplicity_1, multiplicity_2, output_multiplicity;
        int width_1, width_2, output_width, weight_offset;
        bool has_weight, uuu;
        double path_weight;
        Kokkos::View<double*> wigner;
    };
    int input_1_dimension_=0,input_2_dimension_=0,output_dimension_=0,weight_size_=0;
    std::vector<Instruction> instructions;
    Kokkos::View<double*> internal_weights, output_mask;
};
