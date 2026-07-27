#pragma once

#include <Kokkos_Core.hpp>

#include "e3nn.hpp"

template<typename Precision>
class E3LinearKokkosT {
public:
    struct Instruction {
        int input_offset, output_offset, input_multiplicity, output_multiplicity, width, weight_offset;
        Precision path_weight;
        mutable Kokkos::View<Precision**,Kokkos::LayoutRight> packed_input,packed_output;
    };
    E3LinearKokkosT() = default;
    explicit E3LinearKokkosT(const nlohmann::json& data);
    int input_dimension() const { return input_dimension_; }
    int output_dimension() const { return output_dimension_; }
    void evaluate(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
                  Kokkos::View<Precision**,Kokkos::LayoutRight> output) const;
    void reverse(Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
                 Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint) const;
private:
    int input_dimension_=0, output_dimension_=0;
    std::vector<Instruction> instructions;
    Kokkos::View<Precision*> weights, bias, output_mask;
};

template<typename Precision>
class E3TensorProductKokkosT {
public:
    struct Instruction {
        int input_1_offset, input_2_offset, output_offset;
        int multiplicity_1, multiplicity_2, output_multiplicity;
        int width_1, width_2, output_width, weight_offset;
        bool has_weight, uuu;
        Precision path_weight;
        Kokkos::View<Precision*> wigner;
        Kokkos::View<int**,Kokkos::LayoutRight> sparse_indices;
        Kokkos::View<Precision*> sparse_values;
        Kokkos::View<int*> component_offsets;
        int sparse_count=0;
    };
    E3TensorProductKokkosT() = default;
    explicit E3TensorProductKokkosT(const nlohmann::json& data);
    int input_1_dimension() const { return input_1_dimension_; }
    int input_2_dimension() const { return input_2_dimension_; }
    int output_dimension() const { return output_dimension_; }
    int weight_size() const { return weight_size_; }
    bool has_internal_weights() const { return !internal_weights.empty(); }
    bool uses_mh1_fast_path() const { return mh1_fast_path; }
    void evaluate(Kokkos::View<const Precision**,Kokkos::LayoutRight> input_1,
                  Kokkos::View<const Precision**,Kokkos::LayoutRight> input_2,
                  Kokkos::View<const Precision**,Kokkos::LayoutRight> weights,
                  Kokkos::View<Precision**,Kokkos::LayoutRight> output) const;
    void reverse(Kokkos::View<const Precision**,Kokkos::LayoutRight> input_1,
                 Kokkos::View<const Precision**,Kokkos::LayoutRight> input_2,
                 Kokkos::View<const Precision**,Kokkos::LayoutRight> weights,
                 Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
                 Kokkos::View<Precision**,Kokkos::LayoutRight> input_1_adjoint,
                 Kokkos::View<Precision**,Kokkos::LayoutRight> input_2_adjoint,
                 Kokkos::View<Precision**,Kokkos::LayoutRight> weights_adjoint) const;
private:
    int input_1_dimension_=0,input_2_dimension_=0,output_dimension_=0,weight_size_=0;
    bool mh1_fast_path=false;
    int mh1_instruction_count_=0;
    std::vector<Instruction> instructions;
    Kokkos::View<Precision*> internal_weights, output_mask;
    Kokkos::View<int**,Kokkos::LayoutRight> mh1_instruction_data,
        mh1_sparse_indices,mh1_harmonic_terms;
    Kokkos::View<int*> mh1_component_offsets,mh1_harmonic_offsets;
    Kokkos::View<Precision*> mh1_path_weights,mh1_sparse_values,
        mh1_harmonic_values;
};

using E3LinearKokkos = E3LinearKokkosT<double>;
using E3LinearFloatKokkos = E3LinearKokkosT<float>;
using E3TensorProductKokkos = E3TensorProductKokkosT<double>;
using E3TensorProductFloatKokkos = E3TensorProductKokkosT<float>;
