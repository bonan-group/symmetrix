#include "mace_nonlinear_kokkos.hpp"
#include "mace_nonlinear_schema.hpp"

#include <fstream>

#include "sphericart.hpp"
#include "sphericart_cuda.hpp"
#include "tools_kokkos.hpp"

namespace {
std::vector<double> tensor_values(const nlohmann::json& value){return value.at("values").get<std::vector<double>>();}
nlohmann::json load_model_json(const std::string& filename){std::ifstream stream(filename);if(!stream)throw std::runtime_error("Could not open MACE_Nonlinear Kokkos model file: "+filename);return nlohmann::json::parse(stream);}
const nlohmann::json& checked_interaction(const nlohmann::json& data){if(data.at("class").get<std::string>()!="RealAgnosticResidualNonLinearInteractionBlock")throw std::invalid_argument("MACE_Nonlinear Kokkos interaction class is unsupported.");return data;}
const nlohmann::json& readout_component(const nlohmann::json& data,int component){const auto readout_class=data.at("class").get<std::string>();if(readout_class=="LinearReadoutBlock")return data.at("linear");if(readout_class=="NonLinearReadoutBlock")return data.at(component==2?"linear_2":"linear_1");throw std::invalid_argument("MACE_Nonlinear Kokkos readout class is unsupported: "+readout_class);}
}

struct MaceNonlinearKokkos::SphericalHarmonicsState {
#ifdef SYMMETRIX_SPHERICART_CUDA
    explicit SphericalHarmonicsState(int l_max) : calculator(l_max) {}
    sphericart::cuda::SphericalHarmonics<double> calculator;
#else
    explicit SphericalHarmonicsState(int) {}
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

MaceNonlinearKokkos::Readout::Readout(const nlohmann::json& data)
    :nonlinear(data.at("class").get<std::string>()=="NonLinearReadoutBlock"),linear(readout_component(data,0)),linear_1(readout_component(data,1)),linear_2(readout_component(data,2))
{if(nonlinear){if(data.at("activation").get<std::string>()!="silu")throw std::invalid_argument("MACE_Nonlinear Kokkos readout activation is unsupported.");activation_constant=data.at("activation_constants").at(0).get<double>();}}

void MaceNonlinearKokkos::Readout::evaluate(Kokkos::View<const double**,Kokkos::LayoutRight> input,Kokkos::View<double*> output)
{
    if(!nonlinear){Kokkos::View<double**,Kokkos::LayoutRight> result("readout",input.extent(0),1);linear.evaluate(input,result);Kokkos::deep_copy(output,Kokkos::subview(result,Kokkos::ALL,0));return;}
    Kokkos::View<double**,Kokkos::LayoutRight> hidden("readout hidden",input.extent(0),linear_1.output_dimension()),activated("readout activated",input.extent(0),linear_1.output_dimension()),result("readout result",input.extent(0),1);linear_1.evaluate(input,hidden);const double constant=activation_constant;Kokkos::parallel_for("readout silu",hidden.size(),KOKKOS_LAMBDA(int flat){const int i=flat/hidden.extent(1),j=flat%hidden.extent(1);const double x=hidden(i,j);activated(i,j)=constant*x/(1.0+Kokkos::exp(-x));});linear_2.evaluate(activated,result);Kokkos::deep_copy(output,Kokkos::subview(result,Kokkos::ALL,0));
}

void MaceNonlinearKokkos::Readout::reverse(Kokkos::View<const double**,Kokkos::LayoutRight> input,double scale_value,Kokkos::View<double**,Kokkos::LayoutRight> input_adjoint)
{
    Kokkos::View<double**,Kokkos::LayoutRight> seed("readout seed",input.extent(0),1);Kokkos::deep_copy(seed,scale_value);
    if(!nonlinear){linear.reverse(seed,input_adjoint);return;}
    Kokkos::View<double**,Kokkos::LayoutRight> hidden("reverse readout hidden",input.extent(0),linear_1.output_dimension()),activated_adj("reverse readout activated adj",input.extent(0),linear_1.output_dimension()),hidden_adj("reverse readout hidden adj",input.extent(0),linear_1.output_dimension());linear_1.evaluate(input,hidden);linear_2.reverse(seed,activated_adj);const double constant=activation_constant;Kokkos::parallel_for("reverse readout silu",hidden.size(),KOKKOS_LAMBDA(int flat){const int i=flat/hidden.extent(1),j=flat%hidden.extent(1);const double x=hidden(i,j),probability=1.0/(1.0+Kokkos::exp(-x));hidden_adj(i,j)=activated_adj(i,j)*constant*(probability+x*probability*(1.0-probability));});linear_1.reverse(hidden_adj,input_adjoint);
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
    states.resize(interactions.size());has_zbl=data.at("has_zbl").get<bool>();if(has_zbl){const auto& value=data.at("zbl");zbl=ZBLKokkos(value.at("a_exp").get<double>(),value.at("a_prefactor").get<double>(),tensor_values(value.at("c")),tensor_values(value.at("covalent_radii")),static_cast<int>(tensor_values(value.at("p")).at(0)));}
    spherical_harmonics_state=std::make_unique<SphericalHarmonicsState>(l_max);
}

void MaceNonlinearKokkos::compute_Y(Kokkos::View<const double*> xyz)
{
    const int edges=xyz.size()/3;if(Y.size()!=edges*num_lm){Kokkos::realloc(Y,edges*num_lm);Kokkos::realloc(Y_grad,3*edges*num_lm);Kokkos::realloc(xyz_shuffled,3*edges);Kokkos::realloc(Y_grad_shuffled,3*edges*num_lm);}auto shuffled=xyz_shuffled;Kokkos::parallel_for("nonlinear shuffle xyz",edges,KOKKOS_LAMBDA(int i){shuffled(3*i)=xyz(3*i+2);shuffled(3*i+1)=xyz(3*i);shuffled(3*i+2)=xyz(3*i+1);});Kokkos::fence();
#ifndef SYMMETRIX_SPHERICART_CUDA
    auto host_xyz=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),xyz_shuffled);auto host_y=Kokkos::create_mirror_view(Y);auto host_grad=Kokkos::create_mirror_view(Y_grad);sphericart::SphericalHarmonics<double> calculator(l_max);calculator.compute_array_with_gradients(host_xyz.data(),3*edges,host_y.data(),edges*num_lm,host_grad.data(),3*edges*num_lm);Kokkos::deep_copy(Y,host_y);Kokkos::deep_copy(Y_grad,host_grad);
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
    Kokkos::parallel_reduce("validate nonlinear edges",edges,KOKKOS_LAMBDA(int edge,int& invalid){if(neigh_indices(edge)<0||neigh_indices(edge)>=num_nodes||neigh_types(edge)<0||neigh_types(edge)>=local_types||!Kokkos::isfinite(distances(edge))||!(distances(edge)>0.0)||!Kokkos::isfinite(xyz(3*edge))||!Kokkos::isfinite(xyz(3*edge+1))||!Kokkos::isfinite(xyz(3*edge+2)))++invalid;},invalid_edges);
    if(invalid_edges)throw std::invalid_argument("MACE_Nonlinear Kokkos graph has an invalid edge index, type, distance, or vector.");
    compute_Y(xyz);

    Kokkos::realloc(offsets,num_nodes+1);
    Kokkos::realloc(targets,edges);
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

    Kokkos::realloc(attrs,num_nodes,model_num_elements);
    Kokkos::deep_copy(attrs,0.0);
    Kokkos::realloc(product_elements,num_nodes);
    auto local_attrs=attrs;
    auto indices=model_indices;
    auto elements=product_elements;
    Kokkos::parallel_for("nonlinear attrs",num_nodes,KOKKOS_LAMBDA(int node) {
        const int element=indices(node_types(node));
        local_attrs(node,element)=1.0;
        elements(node)=element;
    });
    Kokkos::realloc(features,num_nodes,node_embedding.output_dimension());
    node_embedding.evaluate(attrs,features);

    Kokkos::realloc(cutoffs,edges);
    Kokkos::realloc(radial,edges,num_bessel);
    Kokkos::realloc(edge_harmonics,edges,num_lm);
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

    std::vector<Kokkos::View<int*>> agnostic_elements(interactions.size());
    for(int layer=0;layer<static_cast<int>(interactions.size());++layer) {
        auto& interaction=interactions[layer];
        auto& state=states[layer];
        state.input=features;
        const int up_width=interaction.linear_up.output_dimension();
        const int res_width=interaction.linear_res.output_dimension();
        const int skip_width=interaction.skip.output_dimension();
        const int message_width=interaction.convolution.output_dimension();
        const int output_width=interaction.linear_2.output_dimension();
        Kokkos::realloc(state.up,num_nodes,up_width);
        Kokkos::realloc(state.residual,num_nodes,res_width);
        Kokkos::realloc(state.skip,num_nodes,skip_width);
        interaction.linear_up.evaluate(features,state.up);
        interaction.linear_res.evaluate(state.up,state.residual);
        interaction.skip.evaluate(features,state.skip);

        Kokkos::realloc(state.source_embeddings,num_nodes,
            interaction.source_embedding.output_dimension());
        Kokkos::realloc(state.target_embeddings,num_nodes,
            interaction.target_embedding.output_dimension());
        interaction.source_embedding.evaluate(attrs,state.source_embeddings);
        interaction.target_embedding.evaluate(attrs,state.target_embeddings);
        const int edge_width=num_bessel+state.source_embeddings.extent(1)
            +state.target_embeddings.extent(1);
        Kokkos::realloc(state.edge_features,edges,edge_width);
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

        Kokkos::realloc(state.raw_weights,edges,interaction.convolution.weight_size());
        Kokkos::realloc(state.weights,edges,interaction.convolution.weight_size());
        interaction.convolution_weights.evaluate(state.edge_features,state.raw_weights);
        Kokkos::deep_copy(state.weights,state.raw_weights);
        if(!apply_cutoff) {
            auto weights=state.weights;
            Kokkos::parallel_for("nonlinear weight cutoff",weights.size(),
                KOKKOS_LAMBDA(int flat) {
                    const int edge=flat/weights.extent(1);
                    weights(edge,flat%weights.extent(1))*=local_cutoffs(edge);
                });
        }

        Kokkos::realloc(state.edge_up,edges,up_width);
        auto edge_up=state.edge_up;
        auto up=state.up;
        Kokkos::parallel_for("nonlinear gather edge up",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{edges,up_width}),
            KOKKOS_LAMBDA(int edge,int k) {
                edge_up(edge,k)=up(neigh_indices(edge),k);
            });
        Kokkos::realloc(state.edge_messages,edges,message_width);
        interaction.convolution.evaluate(
            state.edge_up,edge_harmonics,state.weights,state.edge_messages);
        Kokkos::realloc(state.messages,num_nodes,message_width);
        Kokkos::deep_copy(state.messages,0.0);
        auto messages=state.messages;
        auto edge_messages=state.edge_messages;
        Kokkos::parallel_for("nonlinear scatter messages",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{edges,message_width}),
            KOKKOS_LAMBDA(int edge,int k) {
                Kokkos::atomic_add(&messages(local_targets(edge),k),edge_messages(edge,k));
            });

        Kokkos::View<double**,Kokkos::LayoutRight> density_matrix(
            "density raw matrix",edges,1);
        Kokkos::realloc(state.density_raw,edges);
        Kokkos::realloc(state.density_base,edges);
        Kokkos::realloc(state.densities,num_nodes);
        Kokkos::deep_copy(state.densities,0.0);
        interaction.density.evaluate(state.edge_features,density_matrix);
        auto density_raw=state.density_raw;
        auto density_base=state.density_base;
        auto densities=state.densities;
        Kokkos::parallel_for("nonlinear density",edges,KOKKOS_LAMBDA(int edge) {
            const double raw=density_matrix(edge,0);
            const double base=Kokkos::tanh(raw*raw);
            density_raw(edge)=raw;
            density_base(edge)=base;
            Kokkos::atomic_add(&densities(local_targets(edge)),
                base*(embed_cutoff?1.0:local_cutoffs(edge)));
        });

        Kokkos::realloc(state.linear_1_output,num_nodes,
            interaction.linear_1.output_dimension());
        interaction.linear_1.evaluate(state.messages,state.linear_1_output);
        Kokkos::realloc(state.pre_gate,num_nodes,res_width);
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
        Kokkos::realloc(state.gated,num_nodes,interaction.gate.output_size);
        interaction.gate.evaluate(state.pre_gate,state.gated);
        Kokkos::realloc(state.interaction_output,num_nodes,output_width);
        interaction.linear_2.evaluate(state.gated,state.interaction_output);
        Kokkos::realloc(state.output,num_nodes,products[layer].output_dimension());
        Kokkos::View<const int*> layer_elements=product_elements;
        if(products[layer].is_agnostic()) {
            agnostic_elements[layer]=Kokkos::View<int*>("agnostic product elements",num_nodes);
            Kokkos::deep_copy(agnostic_elements[layer],0);
            layer_elements=agnostic_elements[layer];
        }
        products[layer].evaluate(
            state.interaction_output,state.skip,layer_elements,state.output);
        features=state.output;
    }

    Kokkos::realloc(node_energies,num_nodes);
    Kokkos::deep_copy(node_energies,0.0);
    Kokkos::View<double*> contribution("readout contribution",num_nodes);
    std::vector<Kokkos::View<double**,Kokkos::LayoutRight>> layer_adjoints(
        interactions.size());
    for(int layer=0;layer<static_cast<int>(readouts.size());++layer) {
        readouts[layer].evaluate(states[layer].output,contribution);
        auto energies=node_energies;
        Kokkos::parallel_for("sum readout",num_nodes,KOKKOS_LAMBDA(int node) {
            energies(node)+=contribution(node);
        });
        layer_adjoints[layer]=Kokkos::View<double**,Kokkos::LayoutRight>(
            "nonlinear layer adjoint",num_nodes,products[layer].output_dimension());
        readouts[layer].reverse(states[layer].output,scale,layer_adjoints[layer]);
    }
    auto energies=node_energies;
    auto e0=atomic_energies;
    const double energy_scale=scale,energy_shift=shift;
    Kokkos::parallel_for("scale nonlinear energies",num_nodes,KOKKOS_LAMBDA(int node) {
        energies(node)=e0(node_types(node))+energy_scale*energies(node)+energy_shift;
    });

    Kokkos::View<double**,Kokkos::LayoutRight> radial_adjoints(
        "nonlinear radial adjoints",edges,num_bessel);
    Kokkos::View<double**,Kokkos::LayoutRight> harmonic_adjoints(
        "nonlinear harmonic adjoints",edges,num_lm);
    Kokkos::View<double*> cutoff_adjoints("nonlinear cutoff adjoints",edges);
    Kokkos::deep_copy(radial_adjoints,0.0);
    Kokkos::deep_copy(harmonic_adjoints,0.0);
    Kokkos::deep_copy(cutoff_adjoints,0.0);

    for(int layer=static_cast<int>(interactions.size())-1;layer>=0;--layer) {
        auto& interaction=interactions[layer];
        auto& state=states[layer];
        const int input_width=interaction.linear_up.input_dimension();
        const int up_width=interaction.linear_up.output_dimension();
        const int message_width=interaction.convolution.output_dimension();
        const int interaction_width=interaction.linear_2.output_dimension();
        const int pre_gate_width=state.pre_gate.extent(1);
        const int skip_width=interaction.skip.output_dimension();

        Kokkos::View<double**,Kokkos::LayoutRight> interaction_output_adj(
            "interaction output adjoint",num_nodes,interaction_width);
        Kokkos::View<double**,Kokkos::LayoutRight> skip_adj(
            "product skip adjoint",num_nodes,skip_width);
        Kokkos::View<const int*> layer_elements=products[layer].is_agnostic()
            ?Kokkos::View<const int*>(agnostic_elements[layer])
            :Kokkos::View<const int*>(product_elements);
        products[layer].reverse(
            state.interaction_output,layer_elements,layer_adjoints[layer],
            interaction_output_adj,skip_adj);

        Kokkos::View<double**,Kokkos::LayoutRight> gated_adj(
            "gated adjoint",num_nodes,state.gated.extent(1));
        Kokkos::View<double**,Kokkos::LayoutRight> pre_gate_adj(
            "pre gate adjoint",num_nodes,pre_gate_width);
        interaction.linear_2.reverse(interaction_output_adj,gated_adj);
        interaction.gate.reverse(state.pre_gate,gated_adj,pre_gate_adj);

        Kokkos::View<double**,Kokkos::LayoutRight> linear_adj(
            "normalized message adjoint",num_nodes,pre_gate_width);
        Kokkos::View<double*> density_adj("density adjoint",num_nodes);
        Kokkos::deep_copy(density_adj,0.0);
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

        Kokkos::View<double**,Kokkos::LayoutRight> message_adj(
            "message adjoint",num_nodes,message_width);
        interaction.linear_1.reverse(linear_adj,message_adj);
        Kokkos::View<double**,Kokkos::LayoutRight> residual_up_adj(
            "residual up adjoint",num_nodes,up_width);
        interaction.linear_res.reverse(pre_gate_adj,residual_up_adj);
        Kokkos::View<double**,Kokkos::LayoutRight> up_adj(
            "up adjoint",num_nodes,up_width);
        Kokkos::deep_copy(up_adj,residual_up_adj);

        Kokkos::View<double**,Kokkos::LayoutRight> edge_message_adj(
            "edge message adjoint",edges,message_width);
        auto local_message_adj=message_adj;
        Kokkos::parallel_for("gather edge message adjoint",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{edges,message_width}),
            KOKKOS_LAMBDA(int edge,int k) {
                edge_message_adj(edge,k)=local_message_adj(local_targets(edge),k);
            });
        Kokkos::View<double**,Kokkos::LayoutRight> edge_up_adj(
            "edge up adjoint",edges,up_width);
        Kokkos::View<double**,Kokkos::LayoutRight> edge_harmonic_adj(
            "edge harmonic adjoint",edges,num_lm);
        Kokkos::View<double**,Kokkos::LayoutRight> weight_adj(
            "convolution weight adjoint",edges,interaction.convolution.weight_size());
        interaction.convolution.reverse(
            state.edge_up,edge_harmonics,state.weights,edge_message_adj,
            edge_up_adj,edge_harmonic_adj,weight_adj);
        auto local_up_adj=up_adj;
        Kokkos::parallel_for("scatter edge up adjoint",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{edges,up_width}),
            KOKKOS_LAMBDA(int edge,int k) {
                Kokkos::atomic_add(&local_up_adj(neigh_indices(edge),k),edge_up_adj(edge,k));
            });
        Kokkos::parallel_for("sum harmonic adjoints",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{edges,num_lm}),
            KOKKOS_LAMBDA(int edge,int lm) {
                harmonic_adjoints(edge,lm)+=edge_harmonic_adj(edge,lm);
            });

        Kokkos::View<double**,Kokkos::LayoutRight> raw_weight_adj(
            "raw convolution weight adjoint",edges,interaction.convolution.weight_size());
        auto raw_weights=state.raw_weights;
        if(!apply_cutoff) {
            Kokkos::parallel_for("reverse convolution cutoff",edges,
                KOKKOS_LAMBDA(int edge) {
                    double cutoff_value=0.0;
                    for(int k=0;k<weight_adj.extent(1);++k) {
                        cutoff_value+=weight_adj(edge,k)*raw_weights(edge,k);
                        raw_weight_adj(edge,k)=weight_adj(edge,k)*local_cutoffs(edge);
                    }
                    cutoff_adjoints(edge)+=cutoff_value;
                });
        } else Kokkos::deep_copy(raw_weight_adj,weight_adj);

        Kokkos::View<double**,Kokkos::LayoutRight> edge_feature_adj(
            "edge feature adjoint",edges,state.edge_features.extent(1));
        interaction.convolution_weights.reverse(
            state.edge_features,raw_weight_adj,edge_feature_adj);
        Kokkos::View<double**,Kokkos::LayoutRight> density_raw_adj(
            "density raw adjoint",edges,1);
        auto density_raw=state.density_raw;
        auto density_base=state.density_base;
        Kokkos::parallel_for("reverse density envelope",edges,KOKKOS_LAMBDA(int edge) {
            double value=density_adj(local_targets(edge))
                *(1.0-density_base(edge)*density_base(edge))*2.0*density_raw(edge);
            if(!embed_cutoff) {
                cutoff_adjoints(edge)+=density_adj(local_targets(edge))*density_base(edge);
                value*=local_cutoffs(edge);
            }
            density_raw_adj(edge,0)=value;
        });
        Kokkos::View<double**,Kokkos::LayoutRight> density_feature_adj(
            "density feature adjoint",edges,state.edge_features.extent(1));
        interaction.density.reverse(
            state.edge_features,density_raw_adj,density_feature_adj);
        Kokkos::parallel_for("sum radial adjoints",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{edges,num_bessel}),
            KOKKOS_LAMBDA(int edge,int k) {
                radial_adjoints(edge,k)+=edge_feature_adj(edge,k)
                    +density_feature_adj(edge,k);
            });

        Kokkos::View<double**,Kokkos::LayoutRight> up_input_adj(
            "up input adjoint",num_nodes,input_width);
        Kokkos::View<double**,Kokkos::LayoutRight> skip_input_adj(
            "skip input adjoint",num_nodes,input_width);
        interaction.linear_up.reverse(up_adj,up_input_adj);
        interaction.skip.reverse(skip_adj,skip_input_adj);
        if(layer>0) {
            auto previous=layer_adjoints[layer-1];
            Kokkos::parallel_for("propagate layer adjoint",previous.size(),
                KOKKOS_LAMBDA(int flat) {
                    const int node=flat/previous.extent(1),k=flat%previous.extent(1);
                    previous(node,k)+=up_input_adj(node,k)+skip_input_adj(node,k);
                });
        }
    }

    Kokkos::realloc(node_forces,xyz.size());
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
        double distance_adjoint=cutoff_adjoints(edge)*envelope_derivative;
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
            distance_adjoint+=radial_adjoints(edge,k)*derivative;
        }
        for(int component=0;component<3;++component) {
            double vector_adjoint=distance_adjoint*xyz(3*edge+component)/distance;
            for(int lm=0;lm<nlm;++lm)
                vector_adjoint+=harmonic_adjoints(edge,lm)
                    *flat_grad((3*edge+component)*nlm+lm);
            forces(3*edge+component)=-vector_adjoint;
        }
    });

    if(has_zbl) {
        Kokkos::View<double*> zbl_e("zbl e",num_nodes),zbl_f("zbl f",xyz.size());
        Kokkos::deep_copy(zbl_e,0.0);
        Kokkos::deep_copy(zbl_f,0.0);
        zbl.compute_ZBL(num_nodes,node_types,num_neigh,neigh_types,atomic_numbers,
            distances,xyz,zbl_e,zbl_f);
        Kokkos::parallel_for("add zbl energy",num_nodes,KOKKOS_LAMBDA(int node) {
            energies(node)+=energy_scale*zbl_e(node);
        });
        Kokkos::parallel_for("add zbl forces",xyz.size(),KOKKOS_LAMBDA(int index) {
            forces(index)+=energy_scale*zbl_f(index);
        });
    }
}
