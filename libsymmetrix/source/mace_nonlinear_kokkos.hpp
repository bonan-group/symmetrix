#pragma once

#include <Kokkos_Core.hpp>

#include <memory>

#include "affine_mlp_kokkos.hpp"
#include "e3nn_kokkos.hpp"
#include "e3nn_product_kokkos.hpp"
#include "zbl_kokkos.hpp"

class MaceNonlinearKokkos {
public:
    explicit MaceNonlinearKokkos(const std::string& filename);
    ~MaceNonlinearKokkos();
    double r_cut=0.0;
    bool has_field_coupling=false;
    bool uses_mh1_fast_path() const { return mh1_fast_path; }
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
    explicit MaceNonlinearKokkos(const nlohmann::json& data);

public:
    struct Gate {
        int scalar_size=0,gate_size=0,gated_size=0,output_size=0;
        std::vector<IrrepBlock> scalar_blocks,gated_blocks;
        Kokkos::View<double*> scalar_constants;
        Kokkos::View<double*> gate_constants;
        explicit Gate(const nlohmann::json& data);
        void evaluate(Kokkos::View<const double**,Kokkos::LayoutRight> input,
                      Kokkos::View<double**,Kokkos::LayoutRight> output) const;
        void reverse(Kokkos::View<const double**,Kokkos::LayoutRight> input,
                     Kokkos::View<const double**,Kokkos::LayoutRight> output_adjoint,
                     Kokkos::View<double**,Kokkos::LayoutRight> input_adjoint) const;
    };
private:
    struct Interaction {
        E3LinearKokkos source_embedding,target_embedding,linear_up,skip,linear_res,linear_1,linear_2;
        E3TensorProductKokkos convolution;
        AffineMLPKokkos convolution_weights,density;
        Gate gate;
        Kokkos::View<double**,Kokkos::LayoutRight> convolution_source_contributions;
        Kokkos::View<double**,Kokkos::LayoutRight> convolution_target_contributions;
        Kokkos::View<double**,Kokkos::LayoutRight> density_source_contributions;
        Kokkos::View<double**,Kokkos::LayoutRight> density_target_contributions;
        double alpha,beta;
        explicit Interaction(const nlohmann::json& data);
        bool prepare_pair_conditioning(
            const nlohmann::json& data,int radial_size,int model_element_count,
            const std::vector<int>& selected_model_indices);
    };
public:
    struct Readout {
        bool nonlinear=false; E3LinearKokkos linear,linear_1,linear_2; double activation_constant=1.0;
        Kokkos::View<double**,Kokkos::LayoutRight> hidden,activated,result,seed,
            activated_adj,hidden_adj;
        Kokkos::View<double**,Kokkos::LayoutRight> hidden_storage,activated_storage,
            result_storage,seed_storage,activated_adj_storage,hidden_adj_storage;
        explicit Readout(const nlohmann::json& data);
        void evaluate(Kokkos::View<const double**,Kokkos::LayoutRight> input,Kokkos::View<double*> output);
        void reverse(Kokkos::View<const double**,Kokkos::LayoutRight> input,double scale,
                     Kokkos::View<double**,Kokkos::LayoutRight> input_adjoint);
    };
private:
    struct LayerState {
        Kokkos::View<double**,Kokkos::LayoutRight> input,up,residual,skip,messages,linear_1_output,pre_gate,gated,interaction_output,output;
        Kokkos::View<double**,Kokkos::LayoutRight> source_embeddings,target_embeddings,edge_features,raw_weights,weights,edge_up,edge_messages;
        Kokkos::View<double**,Kokkos::LayoutRight> convolution_contributions,density_contributions;
        Kokkos::View<double**,Kokkos::LayoutRight> density_matrix,layer_adjoint,
            interaction_output_adj,skip_adj,gated_adj,pre_gate_adj,linear_adj,
            message_adj,residual_up_adj,up_adj,edge_message_adj,edge_up_adj,
            edge_harmonic_adj,weight_adj,raw_weight_adj,edge_feature_adj,
            density_raw_adj,density_feature_adj,up_input_adj,skip_input_adj;
        Kokkos::View<double*> density_raw,density_base,densities,density_adj;
        Kokkos::View<int*> agnostic_elements;

        Kokkos::View<double**,Kokkos::LayoutRight> up_storage,residual_storage,
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
        Kokkos::View<double*> density_raw_storage,density_base_storage,
            densities_storage,density_adj_storage;
        Kokkos::View<int*> agnostic_elements_storage;
    };
    int l_max=0,num_lm=0,model_num_elements=0,cutoff_power=0,num_bessel=0;
    bool apply_cutoff=false,has_agnesi=false,has_zbl=false,mh1_fast_path=false;
    double radial_prefactor=0.0,agnesi_a=0.0,agnesi_q=0.0,agnesi_p=0.0,scale=1.0,shift=0.0;
    Kokkos::View<int*> model_indices,model_atomic_numbers;
    Kokkos::View<double*> bessel_weights,covalent_radii,cutoffs,Y,Y_grad,xyz_shuffled,Y_grad_shuffled;
    Kokkos::View<double**,Kokkos::LayoutRight> attrs,radial,edge_harmonics,features;
    Kokkos::View<int*> targets,product_elements,offsets;
    Kokkos::View<double*> cutoffs_storage,Y_storage,Y_grad_storage,
        xyz_shuffled_storage,Y_grad_shuffled_storage,node_energies_storage,
        node_forces_storage,readout_contribution,readout_contribution_storage,
        cutoff_adjoints,cutoff_adjoints_storage,zbl_energies,zbl_energies_storage,
        zbl_forces,zbl_forces_storage;
    Kokkos::View<double**,Kokkos::LayoutRight> attrs_storage,radial_storage,
        edge_harmonics_storage,features_storage,radial_adjoints,
        radial_adjoints_storage,harmonic_adjoints,harmonic_adjoints_storage;
    Kokkos::View<int*> targets_storage,product_elements_storage,offsets_storage;
    E3LinearKokkos node_embedding;
    std::vector<Interaction> interactions;
    std::vector<E3ProductBasisKokkos> products;
    std::vector<Readout> readouts;
    std::vector<LayerState> states;
    ZBLKokkos zbl;
    std::unique_ptr<SphericalHarmonicsState> spherical_harmonics_state;
};
