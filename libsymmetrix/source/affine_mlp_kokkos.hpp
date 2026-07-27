#pragma once

#include <Kokkos_Core.hpp>

#include <cstddef>

#include "nlohmann/json.hpp"

template<typename Precision>
class AffineMLPKokkosT {
public:
    AffineMLPKokkosT() = default;
    explicit AffineMLPKokkosT(const nlohmann::json& definition);

    int input_size() const;
    int output_size() const;
    bool supports_conditioned_input(int dynamic_input_size) const;
    void evaluate(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
        Kokkos::View<Precision**,Kokkos::LayoutRight> output);
    void reverse(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint);
    void evaluate_conditioned(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> row_contributions,
        Kokkos::View<Precision**,Kokkos::LayoutRight> output);
    void reverse_from_tape(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint);
    void prepare_conditioned_weight(int active_input_size);
    int workspace_rows() const;
    std::size_t workspace_bytes() const;
    void clear_workspace();
    void forward_impl(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> row_contributions);

private:
    enum LayerType : int { Linear = 0, LayerNorm = 1, SiLU = 2 };
    Kokkos::View<int*,Kokkos::SharedSpace> types;
    Kokkos::View<int*,Kokkos::SharedSpace> input_sizes;
    Kokkos::View<int*,Kokkos::SharedSpace> output_sizes;
    Kokkos::View<Precision*,Kokkos::SharedSpace> eps;
    Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> weights;
    Kokkos::View<Kokkos::View<Precision*>*,Kokkos::SharedSpace> biases;
    Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> values;
    Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> adjoints;
    Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> value_storage;
    Kokkos::View<Kokkos::View<Precision**,Kokkos::LayoutRight>*,Kokkos::SharedSpace> adjoint_storage;
    Kokkos::View<Precision**,Kokkos::LayoutRight> conditioned_weight;
    int conditioned_input_size=-1;
    int tape_batch_size=-1,tape_input_size=-1;
    void prepare(int batch_size,int active_input_size);
    void forward(Kokkos::View<const Precision**,Kokkos::LayoutRight> input);
};

using AffineMLPKokkos = AffineMLPKokkosT<double>;
using AffineMLPFloatKokkos = AffineMLPKokkosT<float>;
