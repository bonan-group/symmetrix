#include "e3nn_product_kokkos.hpp"
#include "e3nn_product.hpp"

#include <stdexcept>
#include <utility>

#include "tools_kokkos.hpp"

namespace {
template<typename Precision>
typename E3ProductBasisKokkosT<Precision>::Tensor load_tensor(
    const nlohmann::json& data)
{
    return {
        toKokkosView(
            "product tensor",
            data.at("values").get<std::vector<Precision>>()),
        data.at("shape").get<std::vector<int>>()};
}
}

template<typename Precision>
E3ProductBasisKokkosT<Precision>::E3ProductBasisKokkosT(const nlohmann::json& data)
    : input(data.at("symmetric_contractions").at("irreps_in").get<std::string>()),
      output(data.at("symmetric_contractions").at("irreps_out").get<std::string>()),
      linear(data.at("linear"))
{
    E3ProductBasis validated(data);
    input_dimension_=input.dimension(); output_dimension_=output.dimension(); num_features=input.blocks.front().multiplicity;
    for(const auto& block:input.blocks) { if(block.multiplicity!=num_features) throw std::invalid_argument("Kokkos product requires common multiplicity."); angular_dimension+=2*block.l+1; }
    use_sc=data.at("use_sc").get<bool>(); agnostic=data.value("use_agnostic_product",false);
    if(validated.uses_compiled_plan()) {
        const auto& plan=validated.compiled_plan();
        if(angular_dimension!=16||plan.empty())
            throw std::invalid_argument("Kokkos compiled product plan has an invalid angular layout.");
        int num_elements=plan.front().num_elements;
        for(const auto& block:plan) {
            if(block.num_elements!=num_elements)
                throw std::invalid_argument("Kokkos compiled product element dimensions are inconsistent.");
            compiled_terms+=static_cast<int>(block.terms.size());
        }
        compiled_term_data=Kokkos::View<int**,Kokkos::LayoutRight>(
            "compiled product terms",compiled_terms,4);
        compiled_coefficients=Kokkos::View<Precision***,Kokkos::LayoutRight>(
            "compiled product coefficients",compiled_terms,num_elements,num_features);
        auto host_terms=Kokkos::create_mirror_view(compiled_term_data);
        auto host_coefficients=Kokkos::create_mirror_view(compiled_coefficients);
        int term_offset=0;
        for(int block_index=0;block_index<static_cast<int>(plan.size());++block_index) {
            const auto& source=plan[block_index];
            CompiledBlock block;
            block.angular_offset=source.angular_offset;
            block.output_offset=output.blocks.at(block_index).offset;
            block.width=source.width;
            block.num_elements=source.num_elements;
            block.term_offset=term_offset;
            block.component_offsets=toKokkosView(
                "compiled product component offsets",source.component_offsets);
            for(const auto& term:source.terms) {
                host_terms(term_offset,0)=term.degree;
                for(int axis=0;axis<3;++axis)
                    host_terms(term_offset,axis+1)=term.indices[axis];
                for(int element=0;element<num_elements;++element)
                    for(int feature=0;feature<num_features;++feature)
                        host_coefficients(term_offset,element,feature)=
                            term.coefficients[element*num_features+feature];
                ++term_offset;
            }
            compiled_blocks.push_back(std::move(block));
        }
        Kokkos::deep_copy(compiled_term_data,host_terms);
        Kokkos::deep_copy(compiled_coefficients,host_coefficients);
        return;
    }
    for(const auto& value:data.at("symmetric_contractions").at("contractions")) {
        Contraction contraction; contraction.correlation=value.at("correlation").get<int>();
        for(const auto& tensor:value.at("u_tensors")) contraction.u.push_back(load_tensor<Precision>(tensor));
        for(auto it=value.at("weights").rbegin();it!=value.at("weights").rend();++it) contraction.weights.push_back(load_tensor<Precision>(*it));
        contraction.weights.push_back(load_tensor<Precision>(value.at("weights_max"))); contractions.push_back(std::move(contraction));
    }
}

template<typename Precision>
void E3ProductBasisKokkosT<Precision>::prepare(int batch)
{
    if(feature_major_storage.extent(0)<batch)
        Kokkos::realloc(feature_major_storage,batch,num_features,angular_dimension);
    if(feature_major_adjoint_storage.extent(0)<batch)
        Kokkos::realloc(feature_major_adjoint_storage,batch,num_features,angular_dimension);
    if(contracted_storage.extent(0)<batch)
        Kokkos::realloc(contracted_storage,batch,output_dimension_);
    if(contracted_adjoint_storage.extent(0)<batch)
        Kokkos::realloc(contracted_adjoint_storage,batch,output_dimension_);
    feature_major=Kokkos::subview(
        feature_major_storage,std::make_pair(0,batch),Kokkos::ALL,Kokkos::ALL);
    feature_major_adjoint=Kokkos::subview(
        feature_major_adjoint_storage,std::make_pair(0,batch),Kokkos::ALL,Kokkos::ALL);
    contracted=Kokkos::subview(
        contracted_storage,std::make_pair(0,batch),Kokkos::ALL);
    contracted_adjoint=Kokkos::subview(
        contracted_adjoint_storage,std::make_pair(0,batch),Kokkos::ALL);
}

template<typename Precision>
void E3ProductBasisKokkosT<Precision>::to_feature_major(Kokkos::View<const Precision**,Kokkos::LayoutRight> source)
{
    int angular_offset=0; auto destination=feature_major;
    for(const auto block:input.blocks) { const int width=2*block.l+1; const int offset=angular_offset;
        Kokkos::parallel_for("product layout",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{static_cast<int>(source.extent(0)),num_features,width}),KOKKOS_LAMBDA(int sample,int feature,int component) { destination(sample,feature,offset+component)=source(sample,block.offset+feature*width+component); }); angular_offset+=width;
    }
}

template<typename Precision>
void E3ProductBasisKokkosT<Precision>::evaluate(Kokkos::View<const Precision**,Kokkos::LayoutRight> source,Kokkos::View<const Precision**,Kokkos::LayoutRight> skip,Kokkos::View<const int*> elements,Kokkos::View<Precision**,Kokkos::LayoutRight> result)
{
    prepare(source.extent(0)); to_feature_major(source); Kokkos::deep_copy(contracted,Precision(0)); auto features=feature_major; auto target=contracted; const int feature_count=num_features;
    if(uses_compiled_plan()) {
        int invalid_elements=0;
        const int element_count=compiled_blocks.front().num_elements;
        Kokkos::parallel_reduce(
            "validate compiled product elements",source.extent(0),
            KOKKOS_LAMBDA(int sample,int& invalid) {
                if(elements(sample)<0||elements(sample)>=element_count) ++invalid;
            },invalid_elements);
        if(invalid_elements)
            throw std::out_of_range("MACE_Nonlinear product element index is out of range.");
        auto terms=compiled_term_data;
        auto coefficients=compiled_coefficients;
        for(const auto& block:compiled_blocks) {
            auto component_offsets=block.component_offsets;
            const int term_offset=block.term_offset;
            const int output_offset=block.output_offset;
            const int width=block.width;
            const int element_count=block.num_elements;
            Kokkos::parallel_for(
                "compiled product contraction",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                    {0,0,0},{static_cast<int>(source.extent(0)),feature_count,width}),
                KOKKOS_LAMBDA(int sample,int feature,int component) {
                    const int element=elements(sample);
                    if(element<0||element>=element_count) return;
                    Precision sum=Precision(0);
                    const int first=term_offset+component_offsets(component);
                    const int last=term_offset+component_offsets(component+1);
                    for(int term=first;term<last;++term) {
                        const int degree=terms(term,0);
                        Precision monomial=features(sample,feature,terms(term,1));
                        if(degree>1) monomial*=features(sample,feature,terms(term,2));
                        if(degree>2) monomial*=features(sample,feature,terms(term,3));
                        sum+=coefficients(term,element,feature)*monomial;
                    }
                    target(sample,output_offset+feature*width+component)=sum;
                });
        }
        linear.evaluate(contracted,result);
        if(use_sc) Kokkos::parallel_for("product skip",result.size(),KOKKOS_LAMBDA(int flat) { const int sample=flat/result.extent(1),column=flat%result.extent(1); result(sample,column)+=skip(sample,column); });
        return;
    }
    for(int block_index=0;block_index<static_cast<int>(output.blocks.size());++block_index) { const auto block=output.blocks[block_index]; const int width=2*block.l+1; const auto& contraction=contractions[block_index];
        for(int degree=1;degree<=contraction.correlation;++degree) { const auto& u=contraction.u[degree-1]; const auto& weights=contraction.weights[degree-1]; const auto u_values=u.values; const auto weight_values=weights.values; const int output_axes=block.l==0?0:1; const int parameters=u.shape.back(); int tuples=1; for(int axis=0;axis<degree;++axis) tuples*=u.shape[output_axes+axis]; const int tuple_dimension=angular_dimension;
            Kokkos::parallel_for("product contraction",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{static_cast<int>(source.extent(0)),feature_count,width}),KOKKOS_LAMBDA(int sample,int feature,int component) { const int element=elements(sample); Precision sum=Precision(0); const int component_offset=(output_axes?component*tuples*parameters:0); const int weight_offset=element*parameters*feature_count;
                for(int tuple=0;tuple<tuples;++tuple) { int remainder=tuple; Precision monomial=Precision(1); for(int axis=degree-1;axis>=0;--axis) { const int index=remainder%tuple_dimension; remainder/=tuple_dimension; monomial*=features(sample,feature,index); } Precision coefficient=Precision(0); for(int parameter=0;parameter<parameters;++parameter) coefficient+=u_values(component_offset+tuple*parameters+parameter)*weight_values(weight_offset+parameter*feature_count+feature); sum+=coefficient*monomial; }
                target(sample,block.offset+feature*width+component)+=sum; });
        }
    }
    linear.evaluate(contracted,result); if(use_sc) Kokkos::parallel_for("product skip",result.size(),KOKKOS_LAMBDA(int flat) { const int sample=flat/result.extent(1),column=flat%result.extent(1); result(sample,column)+=skip(sample,column); });
}

template<typename Precision>
void E3ProductBasisKokkosT<Precision>::reverse(Kokkos::View<const Precision**,Kokkos::LayoutRight> source,Kokkos::View<const int*> elements,Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint,Kokkos::View<Precision**,Kokkos::LayoutRight> skip_adjoint)
{
    prepare(source.extent(0)); to_feature_major(source); linear.reverse(output_adjoint,contracted_adjoint); Kokkos::deep_copy(feature_major_adjoint,Precision(0)); auto features=feature_major; auto features_adj=feature_major_adjoint; auto target_adj=contracted_adjoint; const int feature_count=num_features;
    if(uses_compiled_plan()) {
        int invalid_elements=0;
        const int element_count=compiled_blocks.front().num_elements;
        Kokkos::parallel_reduce(
            "validate compiled product reverse elements",source.extent(0),
            KOKKOS_LAMBDA(int sample,int& invalid) {
                if(elements(sample)<0||elements(sample)>=element_count) ++invalid;
            },invalid_elements);
        if(invalid_elements)
            throw std::out_of_range("MACE_Nonlinear product element index is out of range.");
        auto terms=compiled_term_data;
        auto coefficients=compiled_coefficients;
        for(const auto& block:compiled_blocks) {
            auto component_offsets=block.component_offsets;
            const int term_offset=block.term_offset;
            const int output_offset=block.output_offset;
            const int width=block.width;
            const int element_count=block.num_elements;
            Kokkos::parallel_for(
                "compiled product reverse",
                Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                    {0,0},{static_cast<int>(source.extent(0)),feature_count}),
                KOKKOS_LAMBDA(int sample,int feature) {
                    Precision local[16];
                    for(int angular=0;angular<16;++angular)
                        local[angular]=features_adj(sample,feature,angular);
                    const int element=elements(sample);
                    if(element>=0&&element<element_count) {
                        for(int component=0;component<width;++component) {
                            const Precision output_value=target_adj(
                                sample,output_offset+feature*width+component);
                            const int first=term_offset+component_offsets(component);
                            const int last=term_offset+component_offsets(component+1);
                            for(int term=first;term<last;++term) {
                                const Precision common=coefficients(term,element,feature)*output_value;
                                const int degree=terms(term,0);
                                const int i0=terms(term,1);
                                if(degree==1) {
                                    local[i0]+=common;
                                    continue;
                                }
                                const int i1=terms(term,2);
                                const Precision x0=features(sample,feature,i0);
                                const Precision x1=features(sample,feature,i1);
                                if(degree==2) {
                                    local[i0]+=common*x1;
                                    local[i1]+=common*x0;
                                    continue;
                                }
                                const int i2=terms(term,3);
                                const Precision x2=features(sample,feature,i2);
                                local[i0]+=common*x1*x2;
                                local[i1]+=common*x0*x2;
                                local[i2]+=common*x0*x1;
                            }
                        }
                    }
                    for(int angular=0;angular<16;++angular)
                        features_adj(sample,feature,angular)=local[angular];
                });
        }
        Kokkos::deep_copy(input_adjoint,Precision(0)); int angular_offset=0;
        for(const auto block:input.blocks) { const int width=2*block.l+1; const int offset=angular_offset; Kokkos::parallel_for("product reverse layout",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{static_cast<int>(source.extent(0)),num_features,width}),KOKKOS_LAMBDA(int sample,int feature,int component) { input_adjoint(sample,block.offset+feature*width+component)=features_adj(sample,feature,offset+component); }); angular_offset+=width; }
        if(use_sc) Kokkos::deep_copy(skip_adjoint,output_adjoint); else Kokkos::deep_copy(skip_adjoint,Precision(0));
        return;
    }
    for(int block_index=0;block_index<static_cast<int>(output.blocks.size());++block_index) { const auto block=output.blocks[block_index]; const int width=2*block.l+1; const auto& contraction=contractions[block_index];
        for(int degree=1;degree<=contraction.correlation;++degree) { const auto& u=contraction.u[degree-1]; const auto& weights=contraction.weights[degree-1]; const auto u_values=u.values; const auto weight_values=weights.values; const int output_axes=block.l==0?0:1; const int parameters=u.shape.back(); int tuples=1; for(int axis=0;axis<degree;++axis) tuples*=u.shape[output_axes+axis]; const int tuple_dimension=angular_dimension;
            Kokkos::parallel_for("product reverse",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{static_cast<int>(source.extent(0)),feature_count,width}),KOKKOS_LAMBDA(int sample,int feature,int component) { const int element=elements(sample); const Precision adjoint=target_adj(sample,block.offset+feature*width+component); const int component_offset=(output_axes?component*tuples*parameters:0); const int weight_offset=element*parameters*feature_count;
                for(int tuple=0;tuple<tuples;++tuple) { Precision coefficient=Precision(0); for(int parameter=0;parameter<parameters;++parameter) coefficient+=u_values(component_offset+tuple*parameters+parameter)*weight_values(weight_offset+parameter*feature_count+feature); coefficient*=adjoint; for(int differentiated=0;differentiated<degree;++differentiated) { int remainder=tuple,differentiated_index=0; Precision derivative=coefficient; for(int axis=degree-1;axis>=0;--axis) { const int index=remainder%tuple_dimension; remainder/=tuple_dimension; if(axis==differentiated)differentiated_index=index;else derivative*=features(sample,feature,index); } Kokkos::atomic_add(&features_adj(sample,feature,differentiated_index),derivative); } }
            });
        }
    }
    Kokkos::deep_copy(input_adjoint,Precision(0)); int angular_offset=0;
    for(const auto block:input.blocks) { const int width=2*block.l+1; const int offset=angular_offset; Kokkos::parallel_for("product reverse layout",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{static_cast<int>(source.extent(0)),num_features,width}),KOKKOS_LAMBDA(int sample,int feature,int component) { input_adjoint(sample,block.offset+feature*width+component)=features_adj(sample,feature,offset+component); }); angular_offset+=width; }
    if(use_sc) Kokkos::deep_copy(skip_adjoint,output_adjoint); else Kokkos::deep_copy(skip_adjoint,Precision(0));
}

template class E3ProductBasisKokkosT<float>;
template class E3ProductBasisKokkosT<double>;
