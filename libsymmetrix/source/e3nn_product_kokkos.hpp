#pragma once

#include <Kokkos_Core.hpp>

#include "e3nn_kokkos.hpp"

template<typename Precision>
class E3ProductBasisKokkosT {
public:
    struct Tensor { Kokkos::View<Precision*> values; std::vector<int> shape; };
    E3ProductBasisKokkosT() = default;
    explicit E3ProductBasisKokkosT(const nlohmann::json& data);
    int input_dimension() const { return input_dimension_; }
    int output_dimension() const { return output_dimension_; }
    bool is_agnostic() const { return agnostic; }
    bool uses_compiled_plan() const { return !compiled_blocks.empty(); }
    int compiled_term_count() const { return compiled_terms; }
    void evaluate(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
                  Kokkos::View<const Precision**,Kokkos::LayoutRight> skip,
                  Kokkos::View<const int*> elements,
                  Kokkos::View<Precision**,Kokkos::LayoutRight> output);
    void reverse(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
                 Kokkos::View<const int*> elements,
                 Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
                 Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint,
                 Kokkos::View<Precision**,Kokkos::LayoutRight> skip_adjoint);
    void to_feature_major(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> source);

private:
    struct Contraction { int correlation; std::vector<Tensor> u; std::vector<Tensor> weights; };
    struct CompiledBlock {
        int angular_offset=0,output_offset=0,width=0,num_elements=0,term_offset=0;
        Kokkos::View<int*> component_offsets;
    };
    Irreps input{"1x0e"}, output{"1x0e"};
    E3LinearKokkosT<Precision> linear;
    int input_dimension_=0,output_dimension_=0,num_features=0,angular_dimension=0;
    bool use_sc=false,agnostic=false;
    int compiled_terms=0;
    std::vector<Contraction> contractions;
    std::vector<CompiledBlock> compiled_blocks;
    Kokkos::View<int**,Kokkos::LayoutRight> compiled_term_data;
    Kokkos::View<Precision***,Kokkos::LayoutRight> compiled_coefficients;
    Kokkos::View<Precision***,Kokkos::LayoutRight> feature_major, feature_major_adjoint;
    Kokkos::View<Precision**,Kokkos::LayoutRight> contracted, contracted_adjoint;
    Kokkos::View<Precision***,Kokkos::LayoutRight> feature_major_storage,
        feature_major_adjoint_storage;
    Kokkos::View<Precision**,Kokkos::LayoutRight> contracted_storage,
        contracted_adjoint_storage;
    void prepare(int batch);
};

using E3ProductBasisKokkos = E3ProductBasisKokkosT<double>;
using E3ProductBasisFloatKokkos = E3ProductBasisKokkosT<float>;
