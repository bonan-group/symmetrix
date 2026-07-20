#pragma once

#include <Kokkos_Core.hpp>

#include "nlohmann/json.hpp"

class AffineMLPKokkos {
public:
    AffineMLPKokkos() = default;
    explicit AffineMLPKokkos(const nlohmann::json& definition);

    int input_size() const;
    int output_size() const;
    void evaluate(
        Kokkos::View<const double**,Kokkos::LayoutRight> input,
        Kokkos::View<double**,Kokkos::LayoutRight> output);
    void reverse(
        Kokkos::View<const double**,Kokkos::LayoutRight> input,
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
    void prepare(int batch_size);
    void forward(Kokkos::View<const double**,Kokkos::LayoutRight> input);
};
