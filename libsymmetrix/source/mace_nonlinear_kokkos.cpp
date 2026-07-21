#include "mace_nonlinear_kokkos.hpp"
#include "mace_nonlinear_schema.hpp"

#include <algorithm>
#include <fstream>
#include <utility>

#include "affine_mlp.hpp"
#include "sphericart.hpp"
#include "sphericart_cuda.hpp"
#include "tools_kokkos.hpp"

namespace {
std::vector<double> tensor_values(const nlohmann::json& value){return value.at("values").get<std::vector<double>>();}
nlohmann::json load_model_json(const std::string& filename){std::ifstream stream(filename);if(!stream)throw std::runtime_error("Could not open MACE_Nonlinear Kokkos model file: "+filename);return nlohmann::json::parse(stream);}
const nlohmann::json& checked_interaction(const nlohmann::json& data){if(data.at("class").get<std::string>()!="RealAgnosticResidualNonLinearInteractionBlock")throw std::invalid_argument("MACE_Nonlinear Kokkos interaction class is unsupported.");return data;}
const nlohmann::json& readout_component(const nlohmann::json& data,int component){const auto readout_class=data.at("class").get<std::string>();if(readout_class=="LinearReadoutBlock")return data.at("linear");if(readout_class=="NonLinearReadoutBlock")return data.at(component==2?"linear_2":"linear_1");throw std::invalid_argument("MACE_Nonlinear Kokkos readout class is unsupported: "+readout_class);}

template<class View>
void ensure_view(View& view,View& storage,int extent)
{
    if(storage.extent(0)<static_cast<std::size_t>(extent))
        Kokkos::realloc(Kokkos::WithoutInitializing,storage,extent);
    view=Kokkos::subview(storage,std::make_pair(0,extent));
}

template<class View>
void ensure_view(View& view,View& storage,int first,int second)
{
    if(storage.extent(0)<static_cast<std::size_t>(first)
        ||storage.extent(1)!=static_cast<std::size_t>(second))
        Kokkos::realloc(Kokkos::WithoutInitializing,storage,first,second);
    view=Kokkos::subview(storage,std::make_pair(0,first),Kokkos::ALL);
}
}

struct MaceNonlinearKokkos::SphericalHarmonicsState {
#ifdef SYMMETRIX_SPHERICART_CUDA
    explicit SphericalHarmonicsState(int l_max) : calculator(l_max) {}
    sphericart::cuda::SphericalHarmonics<double> calculator;
#else
    explicit SphericalHarmonicsState(int l_max) : calculator(l_max) {}
    sphericart::SphericalHarmonics<double> calculator;
#endif
};

MaceNonlinearKokkos::~MaceNonlinearKokkos() = default;

MaceNonlinearKokkos::Gate::Gate(const nlohmann::json& data)
{
    if(data.at("scalar_activation").get<std::string>()!="silu"||data.at("gate_activation").get<std::string>()!="sigmoid")throw std::invalid_argument("MACE_Nonlinear Kokkos gate has unsupported activations.");
    Irreps scalars(data.at("irreps_scalars").get<std::string>()),gates(data.at("irreps_gates").get<std::string>()),gated(data.at("irreps_gated").get<std::string>()),output(data.at("irreps_out").get<std::string>());
    scalar_size=scalars.dimension(); gate_size=gates.dimension(); gated_size=gated.dimension(); output_size=output.dimension(); scalar_blocks=scalars.blocks; gated_blocks=gated.blocks;
    scalar_constants=toKokkosView("scalar activation constants",data.at("scalar_activation_constants").get<std::vector<double>>()); gate_constants=toKokkosView("gate constants",data.at("gate_activation_constants").get<std::vector<double>>());
    if(scalar_constants.size()!=scalar_blocks.size()||gate_constants.size()!=gated_blocks.size())throw std::invalid_argument("MACE_Nonlinear Kokkos gate activation counts are inconsistent.");
}

void MaceNonlinearKokkos::Gate::evaluate(Kokkos::View<const double**,Kokkos::LayoutRight> input,Kokkos::View<double**,Kokkos::LayoutRight> output) const
{
    for(int block_index=0;block_index<static_cast<int>(scalar_blocks.size());++block_index){const auto block=scalar_blocks[block_index];const double constant=scalar_constants(block_index);Kokkos::parallel_for("nonlinear gate scalars",Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{static_cast<int>(input.extent(0)),block.dimension()}),KOKKOS_LAMBDA(int sample,int index){const int offset=block.offset+index;const double x=input(sample,offset);output(sample,offset)=constant*x/(1.0+Kokkos::exp(-x));});}
    int gate_offset=scalar_size,gated_offset=scalar_size+gate_size,output_offset=scalar_size;
    for(int block_index=0;block_index<static_cast<int>(gated_blocks.size());++block_index){const auto block=gated_blocks[block_index];const int width=2*block.l+1;const int go=gate_offset,gio=gated_offset,oo=output_offset;const double constant=gate_constants(block_index);
        Kokkos::parallel_for("nonlinear gate tensors",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{static_cast<int>(input.extent(0)),block.multiplicity,width}),KOKKOS_LAMBDA(int sample,int feature,int component){const double probability=1.0/(1.0+Kokkos::exp(-input(sample,go+feature)));output(sample,oo+feature*width+component)=constant*probability*input(sample,gio+feature*width+component);});
        gate_offset+=block.multiplicity;gated_offset+=block.dimension();output_offset+=block.dimension();}
}

void MaceNonlinearKokkos::Gate::reverse(Kokkos::View<const double**,Kokkos::LayoutRight> input,Kokkos::View<const double**,Kokkos::LayoutRight> output_adjoint,Kokkos::View<double**,Kokkos::LayoutRight> input_adjoint) const
{
    Kokkos::deep_copy(input_adjoint,0.0);for(int block_index=0;block_index<static_cast<int>(scalar_blocks.size());++block_index){const auto block=scalar_blocks[block_index];const double constant=scalar_constants(block_index);Kokkos::parallel_for("reverse gate scalars",Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{static_cast<int>(input.extent(0)),block.dimension()}),KOKKOS_LAMBDA(int sample,int index){const int offset=block.offset+index;const double x=input(sample,offset),probability=1.0/(1.0+Kokkos::exp(-x));input_adjoint(sample,offset)=output_adjoint(sample,offset)*constant*(probability+x*probability*(1.0-probability));});}
    int gate_offset=scalar_size,gated_offset=scalar_size+gate_size,output_offset=scalar_size;
    for(int block_index=0;block_index<static_cast<int>(gated_blocks.size());++block_index){const auto block=gated_blocks[block_index];const int width=2*block.l+1,go=gate_offset,gio=gated_offset,oo=output_offset;const double constant=gate_constants(block_index);
        Kokkos::parallel_for("reverse gate tensors",Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{static_cast<int>(input.extent(0)),block.multiplicity}),KOKKOS_LAMBDA(int sample,int feature){const double probability=1.0/(1.0+Kokkos::exp(-input(sample,go+feature))),gate=constant*probability;double gate_adjoint=0.0;for(int component=0;component<width;++component){input_adjoint(sample,gio+feature*width+component)=gate*output_adjoint(sample,oo+feature*width+component);gate_adjoint+=input(sample,gio+feature*width+component)*output_adjoint(sample,oo+feature*width+component);}input_adjoint(sample,go+feature)=gate_adjoint*constant*probability*(1.0-probability);});
        gate_offset+=block.multiplicity;gated_offset+=block.dimension();output_offset+=block.dimension();}
}

MaceNonlinearKokkos::Interaction::Interaction(const nlohmann::json& data)
    :source_embedding(checked_interaction(data).at("source_embedding")),target_embedding(data.at("target_embedding")),linear_up(data.at("linear_up")),skip(data.at("skip_tp")),linear_res(data.at("linear_res")),linear_1(data.at("linear_1")),linear_2(data.at("linear_2")),convolution(data.at("conv_tp")),convolution_weights(data.at("conv_tp_weights")),density(data.at("density_fn")),gate(data.at("gate")),alpha(data.at("alpha").get<double>()),beta(data.at("beta").get<double>()){}

bool MaceNonlinearKokkos::Interaction::prepare_pair_conditioning(
    const nlohmann::json& data,int radial_size,int model_element_count,
    const std::vector<int>& selected_model_indices)
{
    AffineMLP convolution_mlp(data.at("conv_tp_weights"));
    AffineMLP density_mlp(data.at("density_fn"));
    if(!convolution_mlp.supports_conditioned_input(radial_size)
        ||!density_mlp.supports_conditioned_input(radial_size))
        return false;
    E3Linear source_linear(data.at("source_embedding"));
    E3Linear target_linear(data.at("target_embedding"));
    const int types=selected_model_indices.size();
    const int convolution_width=data.at("conv_tp_weights").at("layers").at(0)
        .at("weight").at("shape").at(0).get<int>();
    const int density_width=data.at("density_fn").at("layers").at(0)
        .at("weight").at("shape").at(0).get<int>();
    std::vector<double> convolution_source(types*convolution_width);
    std::vector<double> convolution_target(types*convolution_width);
    std::vector<double> density_source(types*density_width);
    std::vector<double> density_target(types*density_width);
    const int target_offset=radial_size+source_linear.output_dimension();
    for(int type=0;type<types;++type) {
        std::vector<double> attrs(model_element_count,0.0);
        attrs.at(selected_model_indices.at(type))=1.0;
        const auto source=source_linear.evaluate(attrs);
        const auto target=target_linear.evaluate(attrs);
        const auto convolution_source_row=
            convolution_mlp.first_layer_contribution(radial_size,source);
        const auto convolution_target_row=
            convolution_mlp.first_layer_contribution(target_offset,target);
        const auto density_source_row=
            density_mlp.first_layer_contribution(radial_size,source);
        const auto density_target_row=
            density_mlp.first_layer_contribution(target_offset,target);
        std::copy(convolution_source_row.begin(),convolution_source_row.end(),
            convolution_source.begin()+type*convolution_width);
        std::copy(convolution_target_row.begin(),convolution_target_row.end(),
            convolution_target.begin()+type*convolution_width);
        std::copy(density_source_row.begin(),density_source_row.end(),
            density_source.begin()+type*density_width);
        std::copy(density_target_row.begin(),density_target_row.end(),
            density_target.begin()+type*density_width);
    }
    set_kokkos_view(
        convolution_source_contributions,convolution_source,types,convolution_width);
    set_kokkos_view(
        convolution_target_contributions,convolution_target,types,convolution_width);
    set_kokkos_view(density_source_contributions,density_source,types,density_width);
    set_kokkos_view(density_target_contributions,density_target,types,density_width);
    return true;
}

MaceNonlinearKokkos::Readout::Readout(const nlohmann::json& data)
    :nonlinear(data.at("class").get<std::string>()=="NonLinearReadoutBlock"),linear(readout_component(data,0)),linear_1(readout_component(data,1)),linear_2(readout_component(data,2))
{if(nonlinear){if(data.at("activation").get<std::string>()!="silu")throw std::invalid_argument("MACE_Nonlinear Kokkos readout activation is unsupported.");activation_constant=data.at("activation_constants").at(0).get<double>();}}

void MaceNonlinearKokkos::Readout::evaluate(Kokkos::View<const double**,Kokkos::LayoutRight> input,Kokkos::View<double*> output)
{
    ensure_view(result,result_storage,input.extent(0),1);
    if(!nonlinear){linear.evaluate(input,result);Kokkos::deep_copy(output,Kokkos::subview(result,Kokkos::ALL,0));return;}
    ensure_view(hidden,hidden_storage,input.extent(0),linear_1.output_dimension());
    ensure_view(activated,activated_storage,input.extent(0),linear_1.output_dimension());
    auto local_hidden=hidden;auto local_activated=activated;
    linear_1.evaluate(input,hidden);const double constant=activation_constant;Kokkos::parallel_for("readout silu",hidden.size(),KOKKOS_LAMBDA(int flat){const int i=flat/local_hidden.extent(1),j=flat%local_hidden.extent(1);const double x=local_hidden(i,j);local_activated(i,j)=constant*x/(1.0+Kokkos::exp(-x));});linear_2.evaluate(activated,result);Kokkos::deep_copy(output,Kokkos::subview(result,Kokkos::ALL,0));
}

void MaceNonlinearKokkos::Readout::reverse(Kokkos::View<const double**,Kokkos::LayoutRight> input,double scale_value,Kokkos::View<double**,Kokkos::LayoutRight> input_adjoint)
{
    ensure_view(seed,seed_storage,input.extent(0),1);Kokkos::deep_copy(seed,scale_value);
    if(!nonlinear){linear.reverse(seed,input_adjoint);return;}
    ensure_view(hidden,hidden_storage,input.extent(0),linear_1.output_dimension());
    ensure_view(activated_adj,activated_adj_storage,input.extent(0),linear_1.output_dimension());
    ensure_view(hidden_adj,hidden_adj_storage,input.extent(0),linear_1.output_dimension());
    auto local_hidden=hidden;auto local_activated_adj=activated_adj;auto local_hidden_adj=hidden_adj;
    linear_1.evaluate(input,hidden);linear_2.reverse(seed,activated_adj);const double constant=activation_constant;Kokkos::parallel_for("reverse readout silu",hidden.size(),KOKKOS_LAMBDA(int flat){const int i=flat/local_hidden.extent(1),j=flat%local_hidden.extent(1);const double x=local_hidden(i,j),probability=1.0/(1.0+Kokkos::exp(-x));local_hidden_adj(i,j)=local_activated_adj(i,j)*constant*(probability+x*probability*(1.0-probability));});linear_1.reverse(hidden_adj,input_adjoint);
}

MaceNonlinearKokkos::MaceNonlinearKokkos(const std::string& filename)
    :MaceNonlinearKokkos(load_model_json(filename))
{}

MaceNonlinearKokkos::MaceNonlinearKokkos(const nlohmann::json& data)
    :node_embedding(data.at("node_embedding"))
{
    validate_mace_nonlinear_schema(data);
    atomic_numbers_host=data.at("atomic_numbers").get<std::vector<int>>();atomic_numbers=toKokkosView("atomic numbers",atomic_numbers_host);model_indices=toKokkosView("model indices",data.at("model_indices").get<std::vector<int>>());model_atomic_numbers=toKokkosView("model atomic numbers",data.at("model_atomic_numbers").get<std::vector<int>>());model_num_elements=model_atomic_numbers.size();r_cut=data.at("r_cut").get<double>();l_max=data.at("l_max").get<int>();num_lm=(l_max+1)*(l_max+1);
    const auto& radial_data=data.at("radial_embedding");apply_cutoff=radial_data.at("apply_cutoff").get<bool>();bessel_weights=toKokkosView("bessel weights",tensor_values(radial_data.at("basis").at("weights")));num_bessel=bessel_weights.size();radial_prefactor=radial_data.at("basis").at("prefactor").get<double>();cutoff_power=radial_data.at("cutoff").at("p").get<int>();const auto& transform=radial_data.at("distance_transform");has_agnesi=transform.at("type").get<std::string>()=="agnesi";if(has_agnesi){agnesi_a=transform.at("a").get<double>();agnesi_q=transform.at("q").get<double>();agnesi_p=transform.at("p").get<double>();covalent_radii=toKokkosView("covalent radii",transform.at("covalent_radii").get<std::vector<double>>());}
    scale=tensor_values(data.at("scale_shift").at("scale")).at(0);shift=tensor_values(data.at("scale_shift").at("shift")).at(0);const auto all_e0=tensor_values(data.at("atomic_energies"));std::vector<double> e0(atomic_numbers_host.size());auto indices=data.at("model_indices").get<std::vector<int>>();for(int i=0;i<e0.size();++i)e0[i]=all_e0.at(indices[i]);atomic_energies=toKokkosView("atomic energies",e0);
    for(const auto& value:data.at("interactions"))interactions.emplace_back(value);for(const auto& value:data.at("products"))products.emplace_back(value);for(const auto& value:data.at("readouts"))readouts.emplace_back(value);
    if(interactions.size()!=products.size()||interactions.size()!=readouts.size())
        throw std::invalid_argument("MACE_Nonlinear Kokkos layer counts are inconsistent.");
#ifdef KOKKOS_ENABLE_CUDA
    mh1_fast_path=false;
#else
    mh1_fast_path=is_published_mh1_architecture(data)
        &&std::all_of(products.begin(),products.end(),[](const auto& product) {
            return product.uses_compiled_plan();
        })
        &&std::all_of(interactions.begin(),interactions.end(),[](const auto& interaction) {
            return interaction.convolution.uses_mh1_fast_path();
        });
    if(mh1_fast_path)
        for(int layer=0;layer<static_cast<int>(interactions.size());++layer)
            mh1_fast_path=interactions[layer].prepare_pair_conditioning(
                data.at("interactions").at(layer),num_bessel,model_num_elements,indices)
                &&mh1_fast_path;
#endif
    states.resize(interactions.size());has_zbl=data.at("has_zbl").get<bool>();if(has_zbl){const auto& value=data.at("zbl");zbl=ZBLKokkos(value.at("a_exp").get<double>(),value.at("a_prefactor").get<double>(),tensor_values(value.at("c")),tensor_values(value.at("covalent_radii")),static_cast<int>(tensor_values(value.at("p")).at(0)));}
    spherical_harmonics_state=std::make_unique<SphericalHarmonicsState>(l_max);
}

void MaceNonlinearKokkos::compute_Y(Kokkos::View<const double*> xyz)
{
    const int edges=xyz.size()/3;
    ensure_view(Y,Y_storage,edges*num_lm);
    ensure_view(Y_grad,Y_grad_storage,3*edges*num_lm);
    ensure_view(xyz_shuffled,xyz_shuffled_storage,3*edges);
    ensure_view(Y_grad_shuffled,Y_grad_shuffled_storage,3*edges*num_lm);
    auto shuffled=xyz_shuffled;Kokkos::parallel_for("nonlinear shuffle xyz",edges,KOKKOS_LAMBDA(int i){shuffled(3*i)=xyz(3*i+2);shuffled(3*i+1)=xyz(3*i);shuffled(3*i+2)=xyz(3*i+1);});Kokkos::fence();
#ifndef SYMMETRIX_SPHERICART_CUDA
    auto host_xyz=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),xyz_shuffled);auto host_y=Kokkos::create_mirror_view(Y);auto host_grad=Kokkos::create_mirror_view(Y_grad);spherical_harmonics_state->calculator.compute_array_with_gradients(host_xyz.data(),3*edges,host_y.data(),edges*num_lm,host_grad.data(),3*edges*num_lm);Kokkos::deep_copy(Y,host_y);Kokkos::deep_copy(Y_grad,host_grad);
#else
    spherical_harmonics_state->calculator.compute_with_gradients(xyz_shuffled.data(),edges,Y.data(),Y_grad.data());
#endif
    Kokkos::deep_copy(Y_grad_shuffled,Y_grad);auto y=Y;auto grad=Y_grad;auto old=Y_grad_shuffled;const int nlm=num_lm;const double factor=2.0*Kokkos::sqrt(M_PI);Kokkos::parallel_for("nonlinear normalize harmonics",edges,KOKKOS_LAMBDA(int edge){for(int lm=0;lm<nlm;++lm){y(edge*nlm+lm)*=factor;grad((3*edge+0)*nlm+lm)=factor*old((3*edge+1)*nlm+lm);grad((3*edge+1)*nlm+lm)=factor*old((3*edge+2)*nlm+lm);grad((3*edge+2)*nlm+lm)=factor*old((3*edge+0)*nlm+lm);}});
}

void MaceNonlinearKokkos::compute_node_energies_forces(int num_nodes,Kokkos::View<const int*> node_types,Kokkos::View<const int*> num_neigh,Kokkos::View<const int*> neigh_indices,Kokkos::View<const int*> neigh_types,Kokkos::View<const double*> xyz,Kokkos::View<const double*> distances)
{
    if(num_nodes<0||node_types.extent(0)!=static_cast<std::size_t>(num_nodes)||num_neigh.extent(0)!=static_cast<std::size_t>(num_nodes)||neigh_indices.extent(0)!=distances.extent(0)||neigh_types.extent(0)!=distances.extent(0)||xyz.extent(0)!=3*distances.extent(0))throw std::invalid_argument("MACE_Nonlinear Kokkos graph input sizes are inconsistent.");
    const int edges=distances.size();
    const int local_types=atomic_numbers_host.size();
    int invalid_nodes=0;
    Kokkos::parallel_reduce("validate nonlinear nodes",num_nodes,KOKKOS_LAMBDA(int node,int& invalid){if(node_types(node)<0||node_types(node)>=local_types||num_neigh(node)<0)++invalid;},invalid_nodes);
    if(invalid_nodes)throw std::invalid_argument("MACE_Nonlinear Kokkos graph has an invalid node type or neighbor count.");
    long long edge_total=0;
    Kokkos::parallel_reduce("validate nonlinear neighbor total",num_nodes,KOKKOS_LAMBDA(int node,long long& total){total+=num_neigh(node);},edge_total);
    if(edge_total!=edges)throw std::invalid_argument("MACE_Nonlinear Kokkos neighbor counts do not match the edge arrays.");
    int invalid_edges=0;
    Kokkos::parallel_reduce("validate nonlinear edges",edges,KOKKOS_LAMBDA(int edge,int& invalid){if(neigh_indices(edge)<0||neigh_indices(edge)>=num_nodes||neigh_types(edge)<0||neigh_types(edge)>=local_types||neigh_types(edge)!=node_types(neigh_indices(edge))||!Kokkos::isfinite(distances(edge))||!(distances(edge)>0.0)||!Kokkos::isfinite(xyz(3*edge))||!Kokkos::isfinite(xyz(3*edge+1))||!Kokkos::isfinite(xyz(3*edge+2)))++invalid;},invalid_edges);
    if(invalid_edges)throw std::invalid_argument("MACE_Nonlinear Kokkos graph has an invalid edge index, type, distance, or vector.");
    compute_Y(xyz);

    ensure_view(offsets,offsets_storage,num_nodes+1);
    ensure_view(targets,targets_storage,edges);
    Kokkos::deep_copy(offsets,0);
    auto local_offsets=offsets;
    Kokkos::parallel_scan("nonlinear edge offsets",num_nodes,
        KOKKOS_LAMBDA(int node,int& update,bool final) {
            const int value=num_neigh(node);
            if(final) {
                local_offsets(node)=update;
                if(node==num_nodes-1) local_offsets(num_nodes)=update+value;
            }
            update+=value;
        });
    auto local_targets=targets;
    Kokkos::parallel_for("nonlinear edge targets",num_nodes,KOKKOS_LAMBDA(int node) {
        for(int edge=local_offsets(node);edge<local_offsets(node+1);++edge)
            local_targets(edge)=node;
    });

    ensure_view(attrs,attrs_storage,num_nodes,model_num_elements);
    Kokkos::deep_copy(attrs,0.0);
    ensure_view(product_elements,product_elements_storage,num_nodes);
    auto local_attrs=attrs;
    auto indices=model_indices;
    auto elements=product_elements;
    Kokkos::parallel_for("nonlinear attrs",num_nodes,KOKKOS_LAMBDA(int node) {
        const int element=indices(node_types(node));
        local_attrs(node,element)=1.0;
        elements(node)=element;
    });
    ensure_view(features,features_storage,num_nodes,node_embedding.output_dimension());
    node_embedding.evaluate(attrs,features);

    ensure_view(cutoffs,cutoffs_storage,edges);
    ensure_view(radial,radial_storage,edges,num_bessel);
    ensure_view(edge_harmonics,edge_harmonics_storage,edges,num_lm);
    auto local_cutoffs=cutoffs;
    auto local_radial=radial;
    auto local_harmonics=edge_harmonics;
    auto bw=bessel_weights;
    auto radii=covalent_radii;
    auto model_z=model_atomic_numbers;
    auto flat_y=Y;
    const double rc=r_cut,prefactor=radial_prefactor,a=agnesi_a,q=agnesi_q,p=agnesi_p;
    const int cp=cutoff_power,nb=num_bessel,nlm=num_lm;
    const bool agnesi=has_agnesi,embed_cutoff=apply_cutoff;
    Kokkos::parallel_for("nonlinear radial",edges,KOKKOS_LAMBDA(int edge) {
        const double distance=distances(edge);
        const double x=distance/rc;
        const double envelope=distance<rc
            ?1.0-0.5*(cp+1.0)*(cp+2.0)*Kokkos::pow(x,cp)
                +cp*(cp+2.0)*Kokkos::pow(x,cp+1)
                -0.5*cp*(cp+1.0)*Kokkos::pow(x,cp+2)
            :0.0;
        local_cutoffs(edge)=envelope;
        double transformed=distance;
        if(agnesi) {
            const int source_z=model_z(indices(neigh_types(edge)));
            const int target_z=model_z(indices(node_types(local_targets(edge))));
            const double r0=.5*(radii(source_z)+radii(target_z));
            const double ratio=distance/r0;
            transformed=1.0/(1.0+a*Kokkos::pow(ratio,q)
                /(1.0+Kokkos::pow(ratio,q-p)));
        }
        for(int k=0;k<nb;++k)
            local_radial(edge,k)=prefactor*Kokkos::sin(bw(k)*transformed)
                /transformed*(embed_cutoff?envelope:1.0);
        for(int lm=0;lm<nlm;++lm)
            local_harmonics(edge,lm)=flat_y(edge*nlm+lm);
    });

    for(int layer=0;layer<static_cast<int>(interactions.size());++layer) {
        auto& interaction=interactions[layer];
        auto& state=states[layer];
        state.input=features;
        const int up_width=interaction.linear_up.output_dimension();
        const int res_width=interaction.linear_res.output_dimension();
        const int skip_width=interaction.skip.output_dimension();
        const int message_width=interaction.convolution.output_dimension();
        const int output_width=interaction.linear_2.output_dimension();
        ensure_view(state.up,state.up_storage,num_nodes,up_width);
        ensure_view(state.residual,state.residual_storage,num_nodes,res_width);
        ensure_view(state.skip,state.skip_storage,num_nodes,skip_width);
        interaction.linear_up.evaluate(features,state.up);
        interaction.linear_res.evaluate(state.up,state.residual);
        interaction.skip.evaluate(features,state.skip);

        if(mh1_fast_path) {
            const int convolution_width=
                interaction.convolution_source_contributions.extent(1);
            const int density_width=interaction.density_source_contributions.extent(1);
            ensure_view(state.convolution_contributions,
                state.convolution_contributions_storage,edges,convolution_width);
            ensure_view(state.density_contributions,
                state.density_contributions_storage,edges,density_width);
            auto convolution_source=interaction.convolution_source_contributions;
            auto convolution_target=interaction.convolution_target_contributions;
            auto density_source=interaction.density_source_contributions;
            auto density_target=interaction.density_target_contributions;
            auto convolution_contributions=state.convolution_contributions;
            auto density_contributions=state.density_contributions;
            Kokkos::parallel_for("nonlinear edge conditioning",edges,KOKKOS_LAMBDA(int edge) {
                const int source_type=neigh_types(edge);
                const int target_type=node_types(local_targets(edge));
                for(int column=0;column<convolution_width;++column)
                    convolution_contributions(edge,column)=
                        convolution_source(source_type,column)+convolution_target(target_type,column);
                for(int column=0;column<density_width;++column)
                    density_contributions(edge,column)=
                        density_source(source_type,column)+density_target(target_type,column);
            });
        } else {
            ensure_view(state.source_embeddings,state.source_embeddings_storage,num_nodes,
                interaction.source_embedding.output_dimension());
            ensure_view(state.target_embeddings,state.target_embeddings_storage,num_nodes,
                interaction.target_embedding.output_dimension());
            interaction.source_embedding.evaluate(attrs,state.source_embeddings);
            interaction.target_embedding.evaluate(attrs,state.target_embeddings);
            const int edge_width=num_bessel+state.source_embeddings.extent(1)
                +state.target_embeddings.extent(1);
            ensure_view(state.edge_features,state.edge_features_storage,edges,edge_width);
            auto source_embed=state.source_embeddings;
            auto target_embed=state.target_embeddings;
            auto edge_features=state.edge_features;
            Kokkos::parallel_for("nonlinear edge features",edges,KOKKOS_LAMBDA(int edge) {
                int column=0;
                for(int k=0;k<nb;++k) edge_features(edge,column++)=local_radial(edge,k);
                const int source=neigh_indices(edge),target=local_targets(edge);
                for(int k=0;k<source_embed.extent(1);++k)
                    edge_features(edge,column++)=source_embed(source,k);
                for(int k=0;k<target_embed.extent(1);++k)
                    edge_features(edge,column++)=target_embed(target,k);
            });
        }

        ensure_view(state.raw_weights,state.raw_weights_storage,edges,
            interaction.convolution.weight_size());
        ensure_view(state.weights,state.weights_storage,edges,
            interaction.convolution.weight_size());
        if(mh1_fast_path)
            interaction.convolution_weights.evaluate_conditioned(
                radial,state.convolution_contributions,state.raw_weights);
        else interaction.convolution_weights.evaluate(state.edge_features,state.raw_weights);
        Kokkos::deep_copy(state.weights,state.raw_weights);
        if(!apply_cutoff) {
            auto weights=state.weights;
            Kokkos::parallel_for("nonlinear weight cutoff",weights.size(),
                KOKKOS_LAMBDA(int flat) {
                    const int edge=flat/weights.extent(1);
                    weights(edge,flat%weights.extent(1))*=local_cutoffs(edge);
                });
        }

        ensure_view(state.edge_up,state.edge_up_storage,edges,up_width);
        auto edge_up=state.edge_up;
        auto up=state.up;
        Kokkos::parallel_for("nonlinear gather edge up",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{edges,up_width}),
            KOKKOS_LAMBDA(int edge,int k) {
                edge_up(edge,k)=up(neigh_indices(edge),k);
            });
        ensure_view(state.edge_messages,state.edge_messages_storage,edges,message_width);
        interaction.convolution.evaluate(
            state.edge_up,edge_harmonics,state.weights,state.edge_messages);
        ensure_view(state.messages,state.messages_storage,num_nodes,message_width);
        auto messages=state.messages;
        auto edge_messages=state.edge_messages;
        Kokkos::parallel_for("nonlinear gather messages",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{num_nodes,message_width}),
            KOKKOS_LAMBDA(int node,int k) {
                double value=0.0;
                for(int edge=local_offsets(node);edge<local_offsets(node+1);++edge)
                    value+=edge_messages(edge,k);
                messages(node,k)=value;
            });

        ensure_view(state.density_matrix,state.density_matrix_storage,edges,1);
        ensure_view(state.density_raw,state.density_raw_storage,edges);
        ensure_view(state.density_base,state.density_base_storage,edges);
        ensure_view(state.densities,state.densities_storage,num_nodes);
        if(mh1_fast_path)
            interaction.density.evaluate_conditioned(
                radial,state.density_contributions,state.density_matrix);
        else interaction.density.evaluate(state.edge_features,state.density_matrix);
        auto density_raw=state.density_raw;
        auto density_base=state.density_base;
        auto densities=state.densities;
        auto density_matrix=state.density_matrix;
        Kokkos::parallel_for("nonlinear density",edges,KOKKOS_LAMBDA(int edge) {
            const double raw=density_matrix(edge,0);
            const double base=Kokkos::tanh(raw*raw);
            density_raw(edge)=raw;
            density_base(edge)=base;
        });
        Kokkos::parallel_for("nonlinear density gather",num_nodes,KOKKOS_LAMBDA(int node) {
            double value=0.0;
            for(int edge=local_offsets(node);edge<local_offsets(node+1);++edge)
                value+=density_base(edge)*(embed_cutoff?1.0:local_cutoffs(edge));
            densities(node)=value;
        });

        ensure_view(state.linear_1_output,state.linear_1_output_storage,num_nodes,
            interaction.linear_1.output_dimension());
        interaction.linear_1.evaluate(state.messages,state.linear_1_output);
        ensure_view(state.pre_gate,state.pre_gate_storage,num_nodes,res_width);
        auto linear_value=state.linear_1_output;
        auto residual=state.residual;
        auto pre_gate=state.pre_gate;
        const double alpha=interaction.alpha,beta=interaction.beta;
        Kokkos::parallel_for("nonlinear normalization",pre_gate.size(),
            KOKKOS_LAMBDA(int flat) {
                const int node=flat/pre_gate.extent(1),k=flat%pre_gate.extent(1);
                pre_gate(node,k)=linear_value(node,k)/(alpha+beta*densities(node))
                    +residual(node,k);
            });
        ensure_view(state.gated,state.gated_storage,num_nodes,interaction.gate.output_size);
        interaction.gate.evaluate(state.pre_gate,state.gated);
        ensure_view(state.interaction_output,state.interaction_output_storage,
            num_nodes,output_width);
        interaction.linear_2.evaluate(state.gated,state.interaction_output);
        ensure_view(state.output,state.output_storage,num_nodes,
            products[layer].output_dimension());
        Kokkos::View<const int*> layer_elements=product_elements;
        if(products[layer].is_agnostic()) {
            ensure_view(state.agnostic_elements,state.agnostic_elements_storage,num_nodes);
            Kokkos::deep_copy(state.agnostic_elements,0);
            layer_elements=state.agnostic_elements;
        }
        products[layer].evaluate(
            state.interaction_output,state.skip,layer_elements,state.output);
        features=state.output;
    }

    ensure_view(node_energies,node_energies_storage,num_nodes);
    Kokkos::deep_copy(node_energies,0.0);
    ensure_view(readout_contribution,readout_contribution_storage,num_nodes);
    for(int layer=0;layer<static_cast<int>(readouts.size());++layer) {
        auto& state=states[layer];
        readouts[layer].evaluate(state.output,readout_contribution);
        auto energies=node_energies;
        auto contribution=readout_contribution;
        Kokkos::parallel_for("sum readout",num_nodes,KOKKOS_LAMBDA(int node) {
            energies(node)+=contribution(node);
        });
        ensure_view(state.layer_adjoint,state.layer_adjoint_storage,num_nodes,
            products[layer].output_dimension());
        readouts[layer].reverse(state.output,scale,state.layer_adjoint);
    }
    auto energies=node_energies;
    auto e0=atomic_energies;
    const double energy_scale=scale,energy_shift=shift;
    Kokkos::parallel_for("scale nonlinear energies",num_nodes,KOKKOS_LAMBDA(int node) {
        energies(node)=e0(node_types(node))+energy_scale*energies(node)+energy_shift;
    });

    ensure_view(radial_adjoints,radial_adjoints_storage,edges,num_bessel);
    ensure_view(harmonic_adjoints,harmonic_adjoints_storage,edges,num_lm);
    ensure_view(cutoff_adjoints,cutoff_adjoints_storage,edges);
    Kokkos::deep_copy(radial_adjoints,0.0);
    Kokkos::deep_copy(harmonic_adjoints,0.0);
    Kokkos::deep_copy(cutoff_adjoints,0.0);
    auto radial_adjoint_values=radial_adjoints;
    auto harmonic_adjoint_values=harmonic_adjoints;
    auto cutoff_adjoint_values=cutoff_adjoints;

    for(int layer=static_cast<int>(interactions.size())-1;layer>=0;--layer) {
        auto& interaction=interactions[layer];
        auto& state=states[layer];
        const int input_width=interaction.linear_up.input_dimension();
        const int up_width=interaction.linear_up.output_dimension();
        const int message_width=interaction.convolution.output_dimension();
        const int interaction_width=interaction.linear_2.output_dimension();
        const int pre_gate_width=state.pre_gate.extent(1);
        const int skip_width=interaction.skip.output_dimension();

        ensure_view(state.interaction_output_adj,state.interaction_output_adj_storage,
            num_nodes,interaction_width);
        ensure_view(state.skip_adj,state.skip_adj_storage,num_nodes,skip_width);
        Kokkos::View<const int*> layer_elements=products[layer].is_agnostic()
            ?Kokkos::View<const int*>(state.agnostic_elements)
            :Kokkos::View<const int*>(product_elements);
        products[layer].reverse(
            state.interaction_output,layer_elements,state.layer_adjoint,
            state.interaction_output_adj,state.skip_adj);

        ensure_view(state.gated_adj,state.gated_adj_storage,num_nodes,
            state.gated.extent(1));
        ensure_view(state.pre_gate_adj,state.pre_gate_adj_storage,num_nodes,
            pre_gate_width);
        interaction.linear_2.reverse(state.interaction_output_adj,state.gated_adj);
        interaction.gate.reverse(state.pre_gate,state.gated_adj,state.pre_gate_adj);

        ensure_view(state.linear_adj,state.linear_adj_storage,num_nodes,pre_gate_width);
        ensure_view(state.density_adj,state.density_adj_storage,num_nodes);
        Kokkos::deep_copy(state.density_adj,0.0);
        auto linear_adj=state.linear_adj;
        auto density_adj=state.density_adj;
        auto pre_gate_adj=state.pre_gate_adj;
        auto linear_forward=state.linear_1_output;
        auto layer_densities=state.densities;
        const double alpha=interaction.alpha,beta=interaction.beta;
        Kokkos::parallel_for("reverse nonlinear normalization",num_nodes,
            KOKKOS_LAMBDA(int node) {
                const double normalization=alpha+beta*layer_densities(node);
                double density_value=0.0;
                for(int k=0;k<pre_gate_width;++k) {
                    linear_adj(node,k)=pre_gate_adj(node,k)/normalization;
                    density_value-=pre_gate_adj(node,k)*linear_forward(node,k)*beta
                        /(normalization*normalization);
                }
                density_adj(node)=density_value;
            });

        ensure_view(state.message_adj,state.message_adj_storage,num_nodes,message_width);
        interaction.linear_1.reverse(state.linear_adj,state.message_adj);
        ensure_view(state.residual_up_adj,state.residual_up_adj_storage,num_nodes,up_width);
        interaction.linear_res.reverse(state.pre_gate_adj,state.residual_up_adj);
        ensure_view(state.up_adj,state.up_adj_storage,num_nodes,up_width);
        Kokkos::deep_copy(state.up_adj,state.residual_up_adj);

        ensure_view(state.edge_message_adj,state.edge_message_adj_storage,edges,
            message_width);
        auto edge_message_adj=state.edge_message_adj;
        auto local_message_adj=state.message_adj;
        Kokkos::parallel_for("gather edge message adjoint",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{edges,message_width}),
            KOKKOS_LAMBDA(int edge,int k) {
                edge_message_adj(edge,k)=local_message_adj(local_targets(edge),k);
            });
        ensure_view(state.edge_up_adj,state.edge_up_adj_storage,edges,up_width);
        ensure_view(state.edge_harmonic_adj,state.edge_harmonic_adj_storage,edges,num_lm);
        ensure_view(state.weight_adj,state.weight_adj_storage,edges,
            interaction.convolution.weight_size());
        interaction.convolution.reverse(
            state.edge_up,edge_harmonics,state.weights,state.edge_message_adj,
            state.edge_up_adj,state.edge_harmonic_adj,state.weight_adj);
        auto local_up_adj=state.up_adj;
        auto edge_up_adj=state.edge_up_adj;
        auto edge_harmonic_adj=state.edge_harmonic_adj;
        Kokkos::parallel_for("scatter edge up adjoint",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{edges,up_width}),
            KOKKOS_LAMBDA(int edge,int k) {
                Kokkos::atomic_add(&local_up_adj(neigh_indices(edge),k),edge_up_adj(edge,k));
            });
        Kokkos::parallel_for("sum harmonic adjoints",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{edges,num_lm}),
            KOKKOS_LAMBDA(int edge,int lm) {
                harmonic_adjoint_values(edge,lm)+=edge_harmonic_adj(edge,lm);
            });

        ensure_view(state.raw_weight_adj,state.raw_weight_adj_storage,edges,
            interaction.convolution.weight_size());
        auto raw_weight_adj=state.raw_weight_adj;
        auto weight_adj=state.weight_adj;
        auto raw_weights=state.raw_weights;
        if(!apply_cutoff) {
            Kokkos::parallel_for("reverse convolution cutoff",edges,
                KOKKOS_LAMBDA(int edge) {
                    double cutoff_value=0.0;
                    for(int k=0;k<weight_adj.extent(1);++k) {
                        cutoff_value+=weight_adj(edge,k)*raw_weights(edge,k);
                        raw_weight_adj(edge,k)=weight_adj(edge,k)*local_cutoffs(edge);
                    }
                    cutoff_adjoint_values(edge)+=cutoff_value;
                });
        } else Kokkos::deep_copy(raw_weight_adj,weight_adj);

        const int edge_feature_width=mh1_fast_path?num_bessel:state.edge_features.extent(1);
        ensure_view(state.edge_feature_adj,state.edge_feature_adj_storage,edges,
            edge_feature_width);
        if(mh1_fast_path)
            interaction.convolution_weights.reverse_from_tape(
                state.raw_weight_adj,state.edge_feature_adj);
        else interaction.convolution_weights.reverse(
            state.edge_features,state.raw_weight_adj,state.edge_feature_adj);
        ensure_view(state.density_raw_adj,state.density_raw_adj_storage,edges,1);
        auto density_raw_adj=state.density_raw_adj;
        auto density_raw=state.density_raw;
        auto density_base=state.density_base;
        Kokkos::parallel_for("reverse density envelope",edges,KOKKOS_LAMBDA(int edge) {
            double value=density_adj(local_targets(edge))
                *(1.0-density_base(edge)*density_base(edge))*2.0*density_raw(edge);
            if(!embed_cutoff) {
                cutoff_adjoint_values(edge)+=density_adj(local_targets(edge))*density_base(edge);
                value*=local_cutoffs(edge);
            }
            density_raw_adj(edge,0)=value;
        });
        ensure_view(state.density_feature_adj,state.density_feature_adj_storage,
            edges,edge_feature_width);
        if(mh1_fast_path)
            interaction.density.reverse_from_tape(
                state.density_raw_adj,state.density_feature_adj);
        else interaction.density.reverse(
            state.edge_features,state.density_raw_adj,state.density_feature_adj);
        auto edge_feature_adj=state.edge_feature_adj;
        auto density_feature_adj=state.density_feature_adj;
        Kokkos::parallel_for("sum radial adjoints",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{edges,num_bessel}),
            KOKKOS_LAMBDA(int edge,int k) {
                radial_adjoint_values(edge,k)+=edge_feature_adj(edge,k)
                    +density_feature_adj(edge,k);
            });

        ensure_view(state.up_input_adj,state.up_input_adj_storage,num_nodes,input_width);
        ensure_view(state.skip_input_adj,state.skip_input_adj_storage,num_nodes,input_width);
        interaction.linear_up.reverse(state.up_adj,state.up_input_adj);
        interaction.skip.reverse(state.skip_adj,state.skip_input_adj);
        if(layer>0) {
            auto previous=states[layer-1].layer_adjoint;
            auto up_input_adj=state.up_input_adj;
            auto skip_input_adj=state.skip_input_adj;
            Kokkos::parallel_for("propagate layer adjoint",previous.size(),
                KOKKOS_LAMBDA(int flat) {
                    const int node=flat/previous.extent(1),k=flat%previous.extent(1);
                    previous(node,k)+=up_input_adj(node,k)+skip_input_adj(node,k);
                });
        }
    }

    ensure_view(node_forces,node_forces_storage,xyz.size());
    auto forces=node_forces;
    auto flat_grad=Y_grad;
    Kokkos::parallel_for("nonlinear edge forces",edges,KOKKOS_LAMBDA(int edge) {
        const double distance=distances(edge);
        const double x=distance/rc;
        double envelope_derivative=0.0;
        if(distance<rc)
            envelope_derivative=(-0.5*cp*(cp+1.0)*(cp+2.0)*Kokkos::pow(x,cp-1)
                +cp*(cp+1.0)*(cp+2.0)*Kokkos::pow(x,cp)
                -0.5*cp*(cp+1.0)*(cp+2.0)*Kokkos::pow(x,cp+1))/rc;
        double transformed=distance,transform_derivative=1.0;
        if(agnesi) {
            const int source_z=model_z(indices(neigh_types(edge)));
            const int target_z=model_z(indices(node_types(local_targets(edge))));
            const double r0=.5*(radii(source_z)+radii(target_z));
            const double ratio=distance/r0;
            const double ratio_power=Kokkos::pow(ratio,q-p);
            const double denominator=1.0+ratio_power;
            const double g=a*Kokkos::pow(ratio,q)/denominator;
            const double dgdx=a*(q*Kokkos::pow(ratio,q-1.0)*denominator
                -(q-p)*Kokkos::pow(ratio,2.0*q-p-1.0))
                /(denominator*denominator);
            transformed=1.0/(1.0+g);
            transform_derivative=-dgdx/(r0*(1.0+g)*(1.0+g));
        }
        double distance_adjoint=cutoff_adjoint_values(edge)*envelope_derivative;
        for(int k=0;k<nb;++k) {
            const double weight=bw(k);
            const double base=prefactor*Kokkos::sin(weight*transformed)/transformed;
            const double base_derivative=prefactor
                *(weight*Kokkos::cos(weight*transformed)*transformed
                    -Kokkos::sin(weight*transformed))
                /(transformed*transformed)*transform_derivative;
            const double derivative=embed_cutoff
                ?base_derivative*local_cutoffs(edge)+base*envelope_derivative
                :base_derivative;
            distance_adjoint+=radial_adjoint_values(edge,k)*derivative;
        }
        for(int component=0;component<3;++component) {
            double vector_adjoint=distance_adjoint*xyz(3*edge+component)/distance;
            for(int lm=0;lm<nlm;++lm)
                vector_adjoint+=harmonic_adjoint_values(edge,lm)
                    *flat_grad((3*edge+component)*nlm+lm);
            forces(3*edge+component)=-vector_adjoint;
        }
    });

    if(has_zbl) {
        ensure_view(zbl_energies,zbl_energies_storage,num_nodes);
        ensure_view(zbl_forces,zbl_forces_storage,xyz.size());
        Kokkos::deep_copy(zbl_energies,0.0);
        Kokkos::deep_copy(zbl_forces,0.0);
        zbl.compute_ZBL(num_nodes,node_types,num_neigh,neigh_types,atomic_numbers,
            distances,xyz,zbl_energies,zbl_forces);
        auto zbl_e=zbl_energies;
        auto zbl_f=zbl_forces;
        Kokkos::parallel_for("add zbl energy",num_nodes,KOKKOS_LAMBDA(int node) {
            energies(node)+=energy_scale*zbl_e(node);
        });
        Kokkos::parallel_for("add zbl forces",xyz.size(),KOKKOS_LAMBDA(int index) {
            forces(index)+=energy_scale*zbl_f(index);
        });
    }
}
