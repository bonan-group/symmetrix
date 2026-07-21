#pragma once

#include <Kokkos_Core.hpp>

#include "nlohmann/json.hpp"

class AffineMLPKokkos {
public:
    AffineMLPKokkos() = default;
    explicit AffineMLPKokkos(const nlohmann::json& definition);

    int input_size() const;
    int output_size() const;
    bool supports_conditioned_input(int dynamic_input_size) const;
    void evaluate(
        Kokkos::View<const double**,Kokkos::LayoutRight> input,
        Kokkos::View<double**,Kokkos::LayoutRight> output);
    void reverse(
        Kokkos::View<const double**,Kokkos::LayoutRight> input,
        Kokkos::View<const double**,Kokkos::LayoutRight> output_adjoint,
        Kokkos::View<double**,Kokkos::LayoutRight> input_adjoint);
    void evaluate_conditioned(
        Kokkos::View<const double**,Kokkos::LayoutRight> input,
        Kokkos::View<const double**,Kokkos::LayoutRight> row_contributions,
        Kokkos::View<double**,Kokkos::LayoutRight> output);
    void reverse_from_tape(
        Kokkos::View<const double**,Kokkos::LayoutRight> output_adjoint,
        Kokkos::View<double**,Kokkos::LayoutRight> input_adjoint);

private:
    enum LayerType : int { Linear = 0, LayerNorm = 1, SiLU = 2 };
    Kokkos::View<int*,Kokkos::SharedSpace> types;
    Kokkos::View<int*,Kokkos::SharedSpace> input_sizes;
    Kokkos::View<int*,Kokkos::SharedSpace> output_sizes;
    Kokkos::View<double*,Kokkos::SharedSpace> eps;
    Kokkos::View<Kokkos::View<double**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> weights;
    Kokkos::View<Kokkos::View<double*>*,Kokkos::SharedSpace> biases;
    Kokkos::View<Kokkos::View<double**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> values;
    Kokkos::View<Kokkos::View<double**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> adjoints;
    Kokkos::View<Kokkos::View<double**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> value_storage;
    Kokkos::View<Kokkos::View<double**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> adjoint_storage;
    int tape_batch_size=-1,tape_input_size=-1;
    void prepare(int batch_size,int active_input_size);
    void forward(Kokkos::View<const double**,Kokkos::LayoutRight> input);
    void forward_impl(
        Kokkos::View<const double**,Kokkos::LayoutRight> input,
        Kokkos::View<const double**,Kokkos::LayoutRight> row_contributions);
};
