#pragma once

#include <Kokkos_Core.hpp>

#include <memory>

#include "e3nn.hpp"

template<typename Precision>
class E3LinearKokkosT {
public:
    enum class Backend { automatic, scalar, packed_gemm };
    struct Workspace {
        Kokkos::View<Precision**,Kokkos::LayoutRight> packed_input,
            packed_output;
        std::size_t bytes() const {
            return sizeof(Precision)*(packed_input.size()+packed_output.size());
        }
    };
    struct Instruction {
        int input_offset, output_offset, input_multiplicity, output_multiplicity, width, weight_offset;
        Precision path_weight;
    };
    E3LinearKokkosT() = default;
    explicit E3LinearKokkosT(const nlohmann::json& data);
    int input_dimension() const { return input_dimension_; }
    int output_dimension() const { return output_dimension_; }
    void set_backend(const std::string& backend);
    std::string backend() const;
    std::string selected_backend(std::size_t samples) const;
    std::size_t workspace_bytes() const;
    void set_workspace(std::shared_ptr<Workspace> workspace) {
        workspace_=std::move(workspace);
    }
    void evaluate(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
                  Kokkos::View<Precision**,Kokkos::LayoutRight> output) const;
    void reverse(Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
                 Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint) const;
private:
    bool use_scalar_backend(std::size_t samples) const;
    int input_dimension_=0, output_dimension_=0;
    Backend backend_=Backend::automatic;
    std::shared_ptr<Workspace> workspace_=std::make_shared<Workspace>();
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
    std::string backend() const {
        return mh1_fast_path ? "official_kokkos" : "generic_kokkos";
    }
    std::string execution_backend() const;
    int channel_team_size() const;
    int harmonic_team_size() const;
    bool supports_direct_node_reverse() const;
    std::size_t workspace_bytes() const { return 0; }
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
    // Adds source-node adjoints and overwrites both edge-adjoint outputs.
    bool try_reverse_from_nodes(
        Kokkos::View<const Precision**,Kokkos::LayoutRight> source_node_values,
        Kokkos::View<const int*> source_indices,
        int first_edge,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> edge_input_2,
        Kokkos::View<const Precision**,Kokkos::LayoutRight> edge_weights,
        Kokkos::View<const Precision**,Kokkos::LayoutRight>
            target_node_output_adjoint,
        Kokkos::View<const int*> target_indices,
        Kokkos::View<Precision**,Kokkos::LayoutRight>
            source_node_input_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight> edge_input_2_adjoint,
        Kokkos::View<Precision**,Kokkos::LayoutRight> edge_weights_adjoint) const;
private:
    static constexpr int max_direct_node_input_components_=16;
    int input_1_dimension_=0,input_2_dimension_=0,output_dimension_=0,weight_size_=0;
    bool mh1_fast_path=false,mh1_direct_node_layout_=false;
    int mh1_instruction_count_=0,mh1_multiplicity_=0,
        mh1_input_1_angular_dimension_=0;
    int mh1_cuda_channel_team_size_=0,mh1_cuda_harmonic_team_size_=0;
    std::vector<Instruction> instructions;
    Kokkos::View<Precision*> internal_weights, output_mask;
    Kokkos::View<int**,Kokkos::LayoutRight> mh1_instruction_data,
        mh1_sparse_indices,mh1_harmonic_terms,mh1_input_component_data;
    Kokkos::View<int*> mh1_component_offsets,mh1_harmonic_offsets;
    Kokkos::View<Precision*> mh1_path_weights,mh1_sparse_values,
        mh1_harmonic_values;
};

using E3LinearKokkos = E3LinearKokkosT<double>;
using E3LinearFloatKokkos = E3LinearKokkosT<float>;
using E3TensorProductKokkos = E3TensorProductKokkosT<double>;
using E3TensorProductFloatKokkos = E3TensorProductKokkosT<float>;
