#pragma once

#include <Kokkos_Core.hpp>

#include <memory>

#include "affine_mlp_kokkos.hpp"
#include "e3nn_kokkos.hpp"
#include "e3nn_product_kokkos.hpp"
#include "mace_streamed_edges.hpp"
#include "zbl_kokkos.hpp"

template<typename Precision>
class MaceNonlinearKokkosT {
public:
    using precision_type=Precision;
    explicit MaceNonlinearKokkosT(const std::string& filename);
    ~MaceNonlinearKokkosT();
    double r_cut=0.0;
    bool has_field_coupling=false;
    bool uses_mh1_fast_path() const { return mh1_fast_path; }
    bool supports_streamed_edges() const { return mh1_fast_path; }
    std::string streamed_edges_mode() const;
    void set_streamed_edges(std::string mode);
    int edge_workspace_rows() const;
    std::size_t edge_workspace_bytes() const;
    void fence() const { Kokkos::fence("MACE_Nonlinear public fence"); }
    void set_e3_linear_backend(const std::string& backend);
    std::string e3_linear_backend() const;
    std::string tensor_product_backend() const;
    std::size_t linear_workspace_bytes() const;
    std::size_t tensor_workspace_bytes() const;
    std::size_t precision_workspace_bytes() const;
    std::vector<int> atomic_numbers_host;
    Kokkos::View<int*> atomic_numbers;
    Kokkos::View<double*> atomic_energies,node_energies,node_forces;
    void compute_node_energies_forces(int num_nodes,Kokkos::View<const int*> node_types,
        Kokkos::View<const int*> num_neigh,Kokkos::View<const int*> neigh_indices,
        Kokkos::View<const int*> neigh_types,Kokkos::View<const double*> xyz,
        Kokkos::View<const double*> distances);
    void compute_Y(Kokkos::View<const double*> xyz);

private:
    struct SphericalHarmonicsState;
    explicit MaceNonlinearKokkosT(const nlohmann::json& data);

public:
    struct Gate {
        int scalar_size=0,gate_size=0,gated_size=0,output_size=0;
        std::vector<IrrepBlock> scalar_blocks,gated_blocks;
        std::vector<Precision> scalar_constants;
        std::vector<Precision> gate_constants;
        explicit Gate(const nlohmann::json& data);
        void evaluate(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
                      Kokkos::View<Precision**,Kokkos::LayoutRight> output) const;
        void reverse(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
                     Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
                     Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint) const;
    };
private:
    struct Interaction {
        E3LinearKokkosT<Precision> source_embedding,target_embedding,linear_up,skip,linear_res,linear_1,linear_2;
        E3TensorProductKokkosT<Precision> convolution;
        AffineMLPKokkosT<Precision> convolution_weights,density;
        Gate gate;
        Kokkos::View<Precision**,Kokkos::LayoutRight> convolution_source_contributions;
        Kokkos::View<Precision**,Kokkos::LayoutRight> convolution_target_contributions;
        Kokkos::View<Precision**,Kokkos::LayoutRight> density_source_contributions;
        Kokkos::View<Precision**,Kokkos::LayoutRight> density_target_contributions;
        Precision alpha,beta;
        explicit Interaction(const nlohmann::json& data);
        bool prepare_pair_conditioning(
            const nlohmann::json& data,int radial_size,int model_element_count,
            const std::vector<int>& selected_model_indices);
        void set_e3_linear_backend(const std::string& backend);
        void set_e3_linear_workspace(
            const std::shared_ptr<typename E3LinearKokkosT<Precision>::Workspace>& workspace);
        std::size_t linear_workspace_bytes() const;
    };
public:
    struct Readout {
        bool nonlinear=false; E3LinearKokkosT<Precision> linear,linear_1,linear_2; Precision activation_constant=1.0;
        Kokkos::View<Precision**,Kokkos::LayoutRight> hidden,activated,result,seed,
            activated_adj,hidden_adj;
        Kokkos::View<Precision**,Kokkos::LayoutRight> hidden_storage,activated_storage,
            result_storage,seed_storage,activated_adj_storage,hidden_adj_storage;
        explicit Readout(const nlohmann::json& data);
        void set_e3_linear_backend(const std::string& backend);
        void set_e3_linear_workspace(
            const std::shared_ptr<typename E3LinearKokkosT<Precision>::Workspace>& workspace);
        std::size_t linear_workspace_bytes() const;
        void evaluate(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,Kokkos::View<Precision*> output);
        void reverse(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,Precision scale,
                     Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint);
    };
private:
    struct LayerState {
        Kokkos::View<Precision**,Kokkos::LayoutRight> input,up,residual,skip,messages,linear_1_output,pre_gate,gated,interaction_output,output;
        Kokkos::View<Precision**,Kokkos::LayoutRight> source_embeddings,target_embeddings,edge_features,raw_weights,weights,edge_up,edge_messages;
        Kokkos::View<Precision**,Kokkos::LayoutRight> convolution_contributions,density_contributions;
        Kokkos::View<Precision**,Kokkos::LayoutRight> density_matrix,layer_adjoint,
            interaction_output_adj,skip_adj,gated_adj,pre_gate_adj,linear_adj,
            message_adj,residual_up_adj,up_adj,edge_message_adj,edge_up_adj,
            edge_harmonic_adj,weight_adj,raw_weight_adj,edge_feature_adj,
            density_raw_adj,density_feature_adj,up_input_adj,skip_input_adj;
        Kokkos::View<Precision*> density_raw,density_base,densities,density_adj;
        Kokkos::View<int*> agnostic_elements;

        Kokkos::View<Precision**,Kokkos::LayoutRight> up_storage,residual_storage,
            skip_storage,messages_storage,linear_1_output_storage,pre_gate_storage,
            gated_storage,interaction_output_storage,output_storage,
            source_embeddings_storage,target_embeddings_storage,
            edge_features_storage,raw_weights_storage,weights_storage,
            edge_up_storage,edge_messages_storage,convolution_contributions_storage,
            density_contributions_storage,density_matrix_storage,
            layer_adjoint_storage,interaction_output_adj_storage,skip_adj_storage,
            gated_adj_storage,pre_gate_adj_storage,linear_adj_storage,
            message_adj_storage,residual_up_adj_storage,up_adj_storage,
            edge_message_adj_storage,edge_up_adj_storage,edge_harmonic_adj_storage,
            weight_adj_storage,raw_weight_adj_storage,edge_feature_adj_storage,
            density_raw_adj_storage,density_feature_adj_storage,
            up_input_adj_storage,skip_input_adj_storage;
        Kokkos::View<Precision*> density_raw_storage,density_base_storage,
            densities_storage,density_adj_storage;
        Kokkos::View<int*> agnostic_elements_storage;
    };
    int l_max=0,num_lm=0,model_num_elements=0,cutoff_power=0,num_bessel=0;
    bool apply_cutoff=false,has_agnesi=false,has_zbl=false,mh1_fast_path=false;
    MACEStreamedEdgesMode streamed_edges=MACEStreamedEdgesMode::legacy;
#ifdef KOKKOS_ENABLE_CUDA
    static constexpr int streamed_edge_block_size=16384;
#else
    static constexpr int streamed_edge_block_size=1024;
#endif
    bool streams_layer(int layer) const;
    void release_layer_edge_workspace(int layer);
    double radial_prefactor=0.0,agnesi_a=0.0,agnesi_q=0.0,agnesi_p=0.0,scale=1.0,shift=0.0;
    Kokkos::View<int*> model_indices,model_atomic_numbers;
    Kokkos::View<Precision*> bessel_weights,covalent_radii,cutoffs,Y,Y_grad,xyz_shuffled,Y_grad_shuffled;
    Kokkos::View<Precision**,Kokkos::LayoutRight> attrs,radial,edge_harmonics,features;
    Kokkos::View<int*> targets,product_elements,offsets;
    Kokkos::View<Precision*> cutoffs_storage,Y_storage,Y_grad_storage,
        xyz_shuffled_storage,Y_grad_shuffled_storage,readout_contribution,
        readout_contribution_storage,cutoff_adjoints,cutoff_adjoints_storage;
    Kokkos::View<double*> node_energies_storage,node_forces_storage,
        zbl_energies,zbl_energies_storage,zbl_forces,zbl_forces_storage;
    Kokkos::View<Precision**,Kokkos::LayoutRight> attrs_storage,radial_storage,
        edge_harmonics_storage,features_storage,radial_adjoints,
        radial_adjoints_storage,harmonic_adjoints,harmonic_adjoints_storage;
    Kokkos::View<int*> targets_storage,product_elements_storage,offsets_storage;
    std::shared_ptr<typename E3LinearKokkosT<Precision>::Workspace>
        e3_linear_workspace;
    E3LinearKokkosT<Precision> node_embedding;
    std::vector<Interaction> interactions;
    std::vector<E3ProductBasisKokkosT<Precision>> products;
    std::vector<Readout> readouts;
    std::vector<LayerState> states;
    ZBLKokkos zbl;
    std::unique_ptr<SphericalHarmonicsState> spherical_harmonics_state;
};

using MaceNonlinearKokkos = MaceNonlinearKokkosT<double>;
using MaceNonlinearFloatKokkos = MaceNonlinearKokkosT<float>;
