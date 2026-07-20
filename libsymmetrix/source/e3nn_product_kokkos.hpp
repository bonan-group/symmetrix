#pragma once

#include <Kokkos_Core.hpp>

#include "e3nn_kokkos.hpp"

class E3ProductBasisKokkos {
public:
    struct Tensor { Kokkos::View<double*> values; std::vector<int> shape; };
    E3ProductBasisKokkos() = default;
    explicit E3ProductBasisKokkos(const nlohmann::json& data);
    int input_dimension() const { return input_dimension_; }
    int output_dimension() const { return output_dimension_; }
    bool is_agnostic() const { return agnostic; }
    void evaluate(Kokkos::View<const double**,Kokkos::LayoutRight> input,
                  Kokkos::View<const double**,Kokkos::LayoutRight> skip,
                  Kokkos::View<const int*> elements,
                  Kokkos::View<double**,Kokkos::LayoutRight> output);
    void reverse(Kokkos::View<const double**,Kokkos::LayoutRight> input,
                 Kokkos::View<const int*> elements,
                 Kokkos::View<const double**,Kokkos::LayoutRight> output_adjoint,
                 Kokkos::View<double**,Kokkos::LayoutRight> input_adjoint,
                 Kokkos::View<double**,Kokkos::LayoutRight> skip_adjoint);

private:
    struct Contraction { int correlation; std::vector<Tensor> u; std::vector<Tensor> weights; };
    Irreps input{"1x0e"}, output{"1x0e"};
    E3LinearKokkos linear;
    int input_dimension_=0,output_dimension_=0,num_features=0,angular_dimension=0;
    bool use_sc=false,agnostic=false;
    std::vector<Contraction> contractions;
    Kokkos::View<double***,Kokkos::LayoutRight> feature_major, feature_major_adjoint;
    Kokkos::View<double**,Kokkos::LayoutRight> contracted, contracted_adjoint;
    void prepare(int batch);
    void to_feature_major(Kokkos::View<const double**,Kokkos::LayoutRight> source);
};
