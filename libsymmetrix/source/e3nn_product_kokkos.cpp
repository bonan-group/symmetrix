#include "e3nn_product_kokkos.hpp"
#include "e3nn_product.hpp"

#include <stdexcept>

#include "tools_kokkos.hpp"

namespace { E3ProductBasisKokkos::Tensor load_tensor(const nlohmann::json& data) { return {toKokkosView("product tensor",data.at("values").get<std::vector<double>>()),data.at("shape").get<std::vector<int>>()}; } }

E3ProductBasisKokkos::E3ProductBasisKokkos(const nlohmann::json& data)
    : input(data.at("symmetric_contractions").at("irreps_in").get<std::string>()),
      output(data.at("symmetric_contractions").at("irreps_out").get<std::string>()),
      linear(data.at("linear"))
{
    E3ProductBasis validated(data);
    input_dimension_=input.dimension(); output_dimension_=output.dimension(); num_features=input.blocks.front().multiplicity;
    for(const auto& block:input.blocks) { if(block.multiplicity!=num_features) throw std::invalid_argument("Kokkos product requires common multiplicity."); angular_dimension+=2*block.l+1; }
    use_sc=data.at("use_sc").get<bool>(); agnostic=data.value("use_agnostic_product",false);
    for(const auto& value:data.at("symmetric_contractions").at("contractions")) {
        Contraction contraction; contraction.correlation=value.at("correlation").get<int>();
        for(const auto& tensor:value.at("u_tensors")) contraction.u.push_back(load_tensor(tensor));
        for(auto it=value.at("weights").rbegin();it!=value.at("weights").rend();++it) contraction.weights.push_back(load_tensor(*it));
        contraction.weights.push_back(load_tensor(value.at("weights_max"))); contractions.push_back(std::move(contraction));
    }
}

void E3ProductBasisKokkos::prepare(int batch)
{
    if(feature_major.extent(0)!=batch) Kokkos::realloc(feature_major,batch,num_features,angular_dimension);
    if(feature_major_adjoint.extent(0)!=batch) Kokkos::realloc(feature_major_adjoint,batch,num_features,angular_dimension);
    if(contracted.extent(0)!=batch) Kokkos::realloc(contracted,batch,output_dimension_);
    if(contracted_adjoint.extent(0)!=batch) Kokkos::realloc(contracted_adjoint,batch,output_dimension_);
}

void E3ProductBasisKokkos::to_feature_major(Kokkos::View<const double**,Kokkos::LayoutRight> source)
{
    int angular_offset=0; auto destination=feature_major;
    for(const auto block:input.blocks) { const int width=2*block.l+1; const int offset=angular_offset;
        Kokkos::parallel_for("product layout",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{static_cast<int>(source.extent(0)),num_features,width}),KOKKOS_LAMBDA(int sample,int feature,int component) { destination(sample,feature,offset+component)=source(sample,block.offset+feature*width+component); }); angular_offset+=width;
    }
}

void E3ProductBasisKokkos::evaluate(Kokkos::View<const double**,Kokkos::LayoutRight> source,Kokkos::View<const double**,Kokkos::LayoutRight> skip,Kokkos::View<const int*> elements,Kokkos::View<double**,Kokkos::LayoutRight> result)
{
    prepare(source.extent(0)); to_feature_major(source); Kokkos::deep_copy(contracted,0.0); auto features=feature_major; auto target=contracted; const int feature_count=num_features;
    for(int block_index=0;block_index<static_cast<int>(output.blocks.size());++block_index) { const auto block=output.blocks[block_index]; const int width=2*block.l+1; const auto& contraction=contractions[block_index];
        for(int degree=1;degree<=contraction.correlation;++degree) { const auto& u=contraction.u[degree-1]; const auto& weights=contraction.weights[degree-1]; const auto u_values=u.values; const auto weight_values=weights.values; const int output_axes=block.l==0?0:1; const int parameters=u.shape.back(); int tuples=1; for(int axis=0;axis<degree;++axis) tuples*=u.shape[output_axes+axis]; const int tuple_dimension=angular_dimension;
            Kokkos::parallel_for("product contraction",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{static_cast<int>(source.extent(0)),feature_count,width}),KOKKOS_LAMBDA(int sample,int feature,int component) { const int element=elements(sample); double sum=0.0; const int component_offset=(output_axes?component*tuples*parameters:0); const int weight_offset=element*parameters*feature_count;
                for(int tuple=0;tuple<tuples;++tuple) { int remainder=tuple; double monomial=1.0; for(int axis=degree-1;axis>=0;--axis) { const int index=remainder%tuple_dimension; remainder/=tuple_dimension; monomial*=features(sample,feature,index); } double coefficient=0.0; for(int parameter=0;parameter<parameters;++parameter) coefficient+=u_values(component_offset+tuple*parameters+parameter)*weight_values(weight_offset+parameter*feature_count+feature); sum+=coefficient*monomial; }
                target(sample,block.offset+feature*width+component)+=sum; });
        }
    }
    linear.evaluate(contracted,result); if(use_sc) Kokkos::parallel_for("product skip",result.size(),KOKKOS_LAMBDA(int flat) { const int sample=flat/result.extent(1),column=flat%result.extent(1); result(sample,column)+=skip(sample,column); });
}

void E3ProductBasisKokkos::reverse(Kokkos::View<const double**,Kokkos::LayoutRight> source,Kokkos::View<const int*> elements,Kokkos::View<const double**,Kokkos::LayoutRight> output_adjoint,Kokkos::View<double**,Kokkos::LayoutRight> input_adjoint,Kokkos::View<double**,Kokkos::LayoutRight> skip_adjoint)
{
    prepare(source.extent(0)); to_feature_major(source); linear.reverse(output_adjoint,contracted_adjoint); Kokkos::deep_copy(feature_major_adjoint,0.0); auto features=feature_major; auto features_adj=feature_major_adjoint; auto target_adj=contracted_adjoint; const int feature_count=num_features;
    for(int block_index=0;block_index<static_cast<int>(output.blocks.size());++block_index) { const auto block=output.blocks[block_index]; const int width=2*block.l+1; const auto& contraction=contractions[block_index];
        for(int degree=1;degree<=contraction.correlation;++degree) { const auto& u=contraction.u[degree-1]; const auto& weights=contraction.weights[degree-1]; const auto u_values=u.values; const auto weight_values=weights.values; const int output_axes=block.l==0?0:1; const int parameters=u.shape.back(); int tuples=1; for(int axis=0;axis<degree;++axis) tuples*=u.shape[output_axes+axis]; const int tuple_dimension=angular_dimension;
            Kokkos::parallel_for("product reverse",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{static_cast<int>(source.extent(0)),feature_count,width}),KOKKOS_LAMBDA(int sample,int feature,int component) { const int element=elements(sample); const double adjoint=target_adj(sample,block.offset+feature*width+component); const int component_offset=(output_axes?component*tuples*parameters:0); const int weight_offset=element*parameters*feature_count;
                for(int tuple=0;tuple<tuples;++tuple) { double coefficient=0.0; for(int parameter=0;parameter<parameters;++parameter) coefficient+=u_values(component_offset+tuple*parameters+parameter)*weight_values(weight_offset+parameter*feature_count+feature); coefficient*=adjoint; for(int differentiated=0;differentiated<degree;++differentiated) { int remainder=tuple,differentiated_index=0; double derivative=coefficient; for(int axis=degree-1;axis>=0;--axis) { const int index=remainder%tuple_dimension; remainder/=tuple_dimension; if(axis==differentiated)differentiated_index=index;else derivative*=features(sample,feature,index); } Kokkos::atomic_add(&features_adj(sample,feature,differentiated_index),derivative); } }
            });
        }
    }
    Kokkos::deep_copy(input_adjoint,0.0); int angular_offset=0;
    for(const auto block:input.blocks) { const int width=2*block.l+1; const int offset=angular_offset; Kokkos::parallel_for("product reverse layout",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{static_cast<int>(source.extent(0)),num_features,width}),KOKKOS_LAMBDA(int sample,int feature,int component) { input_adjoint(sample,block.offset+feature*width+component)=features_adj(sample,feature,offset+component); }); angular_offset+=width; }
    if(use_sc) Kokkos::deep_copy(skip_adjoint,output_adjoint); else Kokkos::deep_copy(skip_adjoint,0.0);
}
