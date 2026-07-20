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
    std::vector<int> atomic_numbers_host;
    Kokkos::View<int*> atomic_numbers;
    Kokkos::View<double*> atomic_energies,node_energies,node_forces;
    void compute_node_energies_forces(int num_nodes,Kokkos::View<const int*> node_types,
        Kokkos::View<const int*> num_neigh,Kokkos::View<const int*> neigh_indices,
        Kokkos::View<const int*> neigh_types,Kokkos::View<const double*> xyz,
        Kokkos::View<const double*> distances);

private:
    struct SphericalHarmonicsState;
    explicit MaceNonlinearKokkos(const nlohmann::json& data);

    struct Gate {
        int scalar_size=0,gate_size=0,gated_size=0,output_size=0;
        std::vector<IrrepBlock> scalar_blocks,gated_blocks;
        Kokkos::View<double*,Kokkos::SharedSpace> scalar_constants;
        Kokkos::View<double*,Kokkos::SharedSpace> gate_constants;
        explicit Gate(const nlohmann::json& data);
        void evaluate(Kokkos::View<const double**,Kokkos::LayoutRight> input,
                      Kokkos::View<double**,Kokkos::LayoutRight> output) const;
        void reverse(Kokkos::View<const double**,Kokkos::LayoutRight> input,
                     Kokkos::View<const double**,Kokkos::LayoutRight> output_adjoint,
                     Kokkos::View<double**,Kokkos::LayoutRight> input_adjoint) const;
    };
    struct Interaction {
        E3LinearKokkos source_embedding,target_embedding,linear_up,skip,linear_res,linear_1,linear_2;
        E3TensorProductKokkos convolution;
        AffineMLPKokkos convolution_weights,density;
        Gate gate;
        double alpha,beta;
        explicit Interaction(const nlohmann::json& data);
    };
    struct Readout {
        bool nonlinear=false; E3LinearKokkos linear,linear_1,linear_2; double activation_constant=1.0;
        explicit Readout(const nlohmann::json& data);
        void evaluate(Kokkos::View<const double**,Kokkos::LayoutRight> input,Kokkos::View<double*> output);
        void reverse(Kokkos::View<const double**,Kokkos::LayoutRight> input,double scale,
                     Kokkos::View<double**,Kokkos::LayoutRight> input_adjoint);
    };
    struct LayerState {
        Kokkos::View<double**,Kokkos::LayoutRight> input,up,residual,skip,messages,linear_1_output,pre_gate,gated,interaction_output,output;
        Kokkos::View<double**,Kokkos::LayoutRight> source_embeddings,target_embeddings,edge_features,raw_weights,weights,edge_up,edge_messages;
        Kokkos::View<double*> density_raw,density_base,densities;
    };
    int l_max=0,num_lm=0,model_num_elements=0,cutoff_power=0,num_bessel=0;
    bool apply_cutoff=false,has_agnesi=false,has_zbl=false;
    double radial_prefactor=0.0,agnesi_a=0.0,agnesi_q=0.0,agnesi_p=0.0,scale=1.0,shift=0.0;
    Kokkos::View<int*> model_indices,model_atomic_numbers;
    Kokkos::View<double*> bessel_weights,covalent_radii,cutoffs,Y,Y_grad,xyz_shuffled,Y_grad_shuffled;
    Kokkos::View<double**,Kokkos::LayoutRight> attrs,radial,edge_harmonics,features;
    Kokkos::View<int*> targets,product_elements,offsets;
    E3LinearKokkos node_embedding;
    std::vector<Interaction> interactions;
    std::vector<E3ProductBasisKokkos> products;
    std::vector<Readout> readouts;
    std::vector<LayerState> states;
    ZBLKokkos zbl;
    std::unique_ptr<SphericalHarmonicsState> spherical_harmonics_state;
    void compute_Y(Kokkos::View<const double*> xyz);
};
