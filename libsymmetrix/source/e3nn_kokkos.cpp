#include "e3nn_kokkos.hpp"

#include <numeric>
#include <stdexcept>
#include <utility>

#include "KokkosBlas.hpp"
#include "tools_kokkos.hpp"

namespace {
std::vector<double> tensor_values(const nlohmann::json& value) { return value.at("values").get<std::vector<double>>(); }
int shape_product(const std::vector<int>& shape) { return std::accumulate(shape.begin(),shape.end(),1,std::multiplies<int>()); }
}

E3LinearKokkos::E3LinearKokkos(const nlohmann::json& data)
{
    E3Linear validated(data);
    Irreps input(data.at("irreps_in").get<std::string>()), output(data.at("irreps_out").get<std::string>());
    input_dimension_=input.dimension(); output_dimension_=output.dimension(); int offset=0;
    for (const auto& value:data.at("instructions")) {
        const auto& in=input.blocks.at(value.at("i_in").get<int>()); const auto& out=output.blocks.at(value.at("i_out").get<int>());
        const auto shape=value.at("path_shape").get<std::vector<int>>();
        instructions.push_back({in.offset,out.offset,in.multiplicity,out.multiplicity,2*in.l+1,offset,value.at("path_weight").get<double>()});
        offset+=shape_product(shape);
    }
    weights=toKokkosView("e3 linear weights",tensor_values(data.at("weight")));
    bias=toKokkosView("e3 linear bias",tensor_values(data.at("bias")));
    output_mask=toKokkosView("e3 linear output mask",tensor_values(data.at("output_mask")));
    if (weights.size()!=offset) throw std::invalid_argument("Kokkos e3 linear weight size is invalid.");
}

void E3LinearKokkos::evaluate(Kokkos::View<const double**,Kokkos::LayoutRight> input,Kokkos::View<double**,Kokkos::LayoutRight> output) const
{
    if(input.extent(1)!=static_cast<std::size_t>(input_dimension_)
        ||output.extent(0)!=input.extent(0)
        ||output.extent(1)!=static_cast<std::size_t>(output_dimension_))
        throw std::invalid_argument("Kokkos e3 linear batch dimensions are inconsistent.");
    Kokkos::deep_copy(output,0.0); auto all_weights=weights;
#ifdef KOKKOS_ENABLE_CUDA
    const bool use_scalar_path=true;
#else
    const bool use_scalar_path=input.extent(0)<=4;
#endif
    if(use_scalar_path) {
        for(const auto instruction:instructions)
            Kokkos::parallel_for(
                "e3 linear small batch",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                    {0,0,0},{static_cast<int>(input.extent(0)),
                        instruction.output_multiplicity,instruction.width}),
                KOKKOS_LAMBDA(int sample,int target,int component) {
                    double value=0.0;
                    for(int source=0;source<instruction.input_multiplicity;++source)
                        value+=instruction.path_weight
                            *all_weights(instruction.weight_offset
                                +source*instruction.output_multiplicity+target)
                            *input(sample,instruction.input_offset
                                +source*instruction.width+component);
                    output(sample,instruction.output_offset
                        +target*instruction.width+component)+=value;
                });
        auto mask=output_mask; auto all_bias=bias;
        Kokkos::parallel_for("e3 linear mask",output.size(),KOKKOS_LAMBDA(int flat) {
            const int sample=flat/output.extent(1),column=flat%output.extent(1);
            output(sample,column)=(output(sample,column)
                +(all_bias.size()?all_bias(column):0.0))*mask(column);
        });
        return;
    }
    for (const auto& instruction:instructions) {
        const int samples=input.extent(0);
        const int columns=samples*instruction.width;
        if(instruction.packed_input.extent(0)!=instruction.input_multiplicity
            ||instruction.packed_input.extent(1)<columns)
            Kokkos::realloc(
                instruction.packed_input,instruction.input_multiplicity,columns);
        if(instruction.packed_output.extent(0)!=instruction.output_multiplicity
            ||instruction.packed_output.extent(1)<columns)
            Kokkos::realloc(
                instruction.packed_output,instruction.output_multiplicity,columns);
        auto packed_input=Kokkos::subview(
            instruction.packed_input,Kokkos::ALL,std::make_pair(0,columns));
        auto packed_output=Kokkos::subview(
            instruction.packed_output,Kokkos::ALL,std::make_pair(0,columns));
        Kokkos::parallel_for(
            "e3 linear pack",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {0,0,0},{instruction.input_multiplicity,samples,instruction.width}),
            KOKKOS_LAMBDA(int channel,int sample,int component) {
                packed_input(channel,sample*instruction.width+component)=
                    input(sample,instruction.input_offset
                        +channel*instruction.width+component);
            });
        using weight_matrix=Kokkos::View<
            const double**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
        weight_matrix weight(
            all_weights.data()+instruction.weight_offset,
            instruction.input_multiplicity,instruction.output_multiplicity);
        KokkosBlas::gemm(
            "T","N",instruction.path_weight,weight,packed_input,0.0,packed_output);
        Kokkos::parallel_for(
            "e3 linear unpack",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {0,0,0},{instruction.output_multiplicity,samples,instruction.width}),
            KOKKOS_LAMBDA(int channel,int sample,int component) {
                output(sample,instruction.output_offset
                    +channel*instruction.width+component)
                    +=packed_output(channel,sample*instruction.width+component);
            });
    }
    auto mask=output_mask; auto all_bias=bias;
    Kokkos::parallel_for("e3 linear mask",output.size(),KOKKOS_LAMBDA(int flat) { const int sample=flat/output.extent(1),column=flat%output.extent(1); output(sample,column)=(output(sample,column)+(all_bias.size()?all_bias(column):0.0))*mask(column); });
}

void E3LinearKokkos::reverse(Kokkos::View<const double**,Kokkos::LayoutRight> output_adjoint,Kokkos::View<double**,Kokkos::LayoutRight> input_adjoint) const
{
    if(output_adjoint.extent(1)!=static_cast<std::size_t>(output_dimension_)
        ||input_adjoint.extent(0)!=output_adjoint.extent(0)
        ||input_adjoint.extent(1)!=static_cast<std::size_t>(input_dimension_))
        throw std::invalid_argument("Kokkos e3 linear reverse batch dimensions are inconsistent.");
    Kokkos::deep_copy(input_adjoint,0.0); auto all_weights=weights; auto mask=output_mask;
#ifdef KOKKOS_ENABLE_CUDA
    const bool use_scalar_path=true;
#else
    const bool use_scalar_path=output_adjoint.extent(0)<=4;
#endif
    if(use_scalar_path) {
        for(const auto instruction:instructions)
            Kokkos::parallel_for(
                "e3 reverse linear small batch",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                    {0,0,0},{static_cast<int>(output_adjoint.extent(0)),
                        instruction.input_multiplicity,instruction.width}),
                KOKKOS_LAMBDA(int sample,int source,int component) {
                    double value=0.0;
                    for(int target=0;target<instruction.output_multiplicity;++target) {
                        const int output_index=instruction.output_offset
                            +target*instruction.width+component;
                        value+=instruction.path_weight
                            *all_weights(instruction.weight_offset
                                +source*instruction.output_multiplicity+target)
                            *mask(output_index)*output_adjoint(sample,output_index);
                    }
                    input_adjoint(sample,instruction.input_offset
                        +source*instruction.width+component)+=value;
                });
        return;
    }
    for (const auto& instruction:instructions) {
        const int samples=output_adjoint.extent(0);
        const int columns=samples*instruction.width;
        if(instruction.packed_input.extent(0)!=instruction.input_multiplicity
            ||instruction.packed_input.extent(1)<columns)
            Kokkos::realloc(
                instruction.packed_input,instruction.input_multiplicity,columns);
        if(instruction.packed_output.extent(0)!=instruction.output_multiplicity
            ||instruction.packed_output.extent(1)<columns)
            Kokkos::realloc(
                instruction.packed_output,instruction.output_multiplicity,columns);
        auto packed_input=Kokkos::subview(
            instruction.packed_input,Kokkos::ALL,std::make_pair(0,columns));
        auto packed_output=Kokkos::subview(
            instruction.packed_output,Kokkos::ALL,std::make_pair(0,columns));
        Kokkos::parallel_for(
            "e3 reverse linear pack",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {0,0,0},{instruction.output_multiplicity,samples,instruction.width}),
            KOKKOS_LAMBDA(int channel,int sample,int component) {
                const int output_index=instruction.output_offset
                    +channel*instruction.width+component;
                packed_output(channel,sample*instruction.width+component)=
                    mask(output_index)*output_adjoint(sample,output_index);
            });
        using weight_matrix=Kokkos::View<
            const double**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
        weight_matrix weight(
            all_weights.data()+instruction.weight_offset,
            instruction.input_multiplicity,instruction.output_multiplicity);
        KokkosBlas::gemm(
            "N","N",instruction.path_weight,weight,packed_output,0.0,packed_input);
        Kokkos::parallel_for(
            "e3 reverse linear unpack",
            Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                {0,0,0},{instruction.input_multiplicity,samples,instruction.width}),
            KOKKOS_LAMBDA(int channel,int sample,int component) {
                input_adjoint(sample,instruction.input_offset
                    +channel*instruction.width+component)
                    +=packed_input(channel,sample*instruction.width+component);
            });
    }
}

E3TensorProductKokkos::E3TensorProductKokkos(const nlohmann::json& data)
{
    E3TensorProduct validated(data);
    Irreps in1(data.at("irreps_in1").get<std::string>()),in2(data.at("irreps_in2").get<std::string>()),out(data.at("irreps_out").get<std::string>());
    input_1_dimension_=in1.dimension(); input_2_dimension_=in2.dimension(); output_dimension_=out.dimension(); int offset=0;
    bool official_layout=!validated.instructions.empty();
    std::vector<int> mh1_instruction_data_host;
    std::vector<double> mh1_path_weights_host;
    std::vector<int> mh1_component_offsets_host;
    std::vector<int> mh1_sparse_indices_host;
    std::vector<double> mh1_sparse_values_host;
    struct HarmonicTerm { int instruction,a,c; double value; };
    std::vector<std::vector<HarmonicTerm>> mh1_harmonic_terms_host(
        input_2_dimension_);
    int instruction_index=0;
    for (const auto& value:data.at("instructions")) {
        const auto& a=in1.blocks.at(value.at("i_in1").get<int>()); const auto& b=in2.blocks.at(value.at("i_in2").get<int>()); const auto& c=out.blocks.at(value.at("i_out").get<int>());
        const bool has=value.at("has_weight").get<bool>(); const auto mode=value.at("connection_mode").get<std::string>();
        if (mode!="uvu"&&mode!="uuu") throw std::invalid_argument("Unsupported Kokkos tensor-product mode: "+mode);
        Instruction instruction{a.offset,b.offset,c.offset,a.multiplicity,b.multiplicity,c.multiplicity,2*a.l+1,2*b.l+1,2*c.l+1,offset,has,mode=="uuu",value.at("path_weight").get<double>(),toKokkosView("e3 wigner",tensor_values(value.at("wigner_3j")))};
        const auto path_shape=value.at("path_shape").get<std::vector<int>>();
        official_layout=official_layout&&mode=="uvu"&&has
            &&a.multiplicity==128&&b.multiplicity==1&&c.multiplicity==128
            &&instruction.width_1<=3
            &&path_shape==std::vector<int>({128,1});
        if(official_layout) {
            const auto& entries=validated.instructions.at(instruction_index).nonzero_wigner;
            std::vector<int> component_offsets(instruction.output_width+1,0);
            std::vector<int> indices;
            std::vector<double> values;
            for(int component=0;component<instruction.output_width;++component) {
                for(const auto& entry:entries)
                    if(entry.c==component) {
                        indices.push_back(entry.a);
                        indices.push_back(entry.b);
                        indices.push_back(entry.c);
                        values.push_back(entry.value);
                    }
                component_offsets[component+1]=values.size();
            }
            set_kokkos_view(
                instruction.sparse_indices,indices,values.size(),3);
            instruction.sparse_values=toKokkosView("e3 sparse wigner",values);
            instruction.component_offsets=toKokkosView(
                "e3 sparse component offsets",component_offsets);
            instruction.sparse_count=values.size();

            const int component_base=mh1_component_offsets_host.size();
            const int sparse_base=mh1_sparse_values_host.size();
            mh1_instruction_data_host.insert(
                mh1_instruction_data_host.end(),
                {instruction.input_1_offset,instruction.input_2_offset,
                    instruction.output_offset,instruction.width_1,
                    instruction.output_width,instruction.weight_offset,
                    component_base});
            mh1_path_weights_host.push_back(instruction.path_weight);
            for(const int component_offset:component_offsets)
                mh1_component_offsets_host.push_back(
                    sparse_base+component_offset);
            for(int entry=0;entry<static_cast<int>(values.size());++entry) {
                const int a=indices[3*entry];
                const int b=indices[3*entry+1];
                const int c=indices[3*entry+2];
                mh1_sparse_indices_host.insert(
                    mh1_sparse_indices_host.end(),{a,b,c});
                mh1_sparse_values_host.push_back(values[entry]);
                mh1_harmonic_terms_host.at(instruction.input_2_offset+b)
                    .push_back({instruction_index,a,c,values[entry]});
            }
        }
        instructions.push_back(instruction); if(has) offset+=shape_product(value.at("path_shape").get<std::vector<int>>());
        ++instruction_index;
    }
    weight_size_=offset; internal_weights=toKokkosView("e3 tensor weights",tensor_values(data.at("weight"))); output_mask=toKokkosView("e3 tensor mask",tensor_values(data.at("output_mask")));
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP) \
    || defined(KOKKOS_ENABLE_SYCL) || defined(KOKKOS_ENABLE_OPENMPTARGET)
    mh1_fast_path=false;
#else
    mh1_fast_path=official_layout&&internal_weights.empty();
#endif
    if(mh1_fast_path) {
        mh1_instruction_count_=instructions.size();
        set_kokkos_view(
            mh1_instruction_data,mh1_instruction_data_host,
            mh1_instruction_count_,7);
        mh1_path_weights=toKokkosView(
            "e3 mh1 path weights",mh1_path_weights_host);
        mh1_component_offsets=toKokkosView(
            "e3 mh1 component offsets",mh1_component_offsets_host);
        set_kokkos_view(
            mh1_sparse_indices,mh1_sparse_indices_host,
            mh1_sparse_values_host.size(),3);
        mh1_sparse_values=toKokkosView(
            "e3 mh1 sparse values",mh1_sparse_values_host);

        std::vector<int> harmonic_offsets(input_2_dimension_+1,0);
        std::vector<int> harmonic_terms;
        std::vector<double> harmonic_values;
        for(int component=0;component<input_2_dimension_;++component) {
            for(const auto& term:mh1_harmonic_terms_host[component]) {
                harmonic_terms.insert(
                    harmonic_terms.end(),{term.instruction,term.a,term.c});
                harmonic_values.push_back(term.value);
            }
            harmonic_offsets[component+1]=harmonic_values.size();
        }
        mh1_harmonic_offsets=toKokkosView(
            "e3 mh1 harmonic offsets",harmonic_offsets);
        set_kokkos_view(
            mh1_harmonic_terms,harmonic_terms,harmonic_values.size(),3);
        mh1_harmonic_values=toKokkosView(
            "e3 mh1 harmonic values",harmonic_values);
    }
}

void E3TensorProductKokkos::evaluate(Kokkos::View<const double**,Kokkos::LayoutRight> input_1,Kokkos::View<const double**,Kokkos::LayoutRight> input_2,Kokkos::View<const double**,Kokkos::LayoutRight> dynamic_weights,Kokkos::View<double**,Kokkos::LayoutRight> output) const
{
    Kokkos::deep_copy(output,0.0); auto mask=output_mask; auto fixed=internal_weights;
    if(mh1_fast_path) {
        auto plan=mh1_instruction_data;
        auto paths=mh1_path_weights;
        auto indices=mh1_sparse_indices;
        auto values=mh1_sparse_values;
        auto component_offsets=mh1_component_offsets;
        const int instruction_count=mh1_instruction_count_;
        Kokkos::parallel_for(
            "e3 tensor product mh1 fused",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                {0,0},{static_cast<int>(input_1.extent(0)),128}),
            KOKKOS_LAMBDA(int sample,int channel) {
                for(int instruction=0;instruction<instruction_count;
                    ++instruction) {
                    const int input_1_offset=plan(instruction,0);
                    const int input_2_offset=plan(instruction,1);
                    const int output_offset=plan(instruction,2);
                    const int input_width=plan(instruction,3);
                    const int output_width=plan(instruction,4);
                    const int weight_offset=plan(instruction,5);
                    const int component_base=plan(instruction,6);
                    const double scale=paths(instruction)
                        *dynamic_weights(sample,weight_offset+channel);
                    for(int component=0;component<output_width;++component) {
                        double result=0.0;
                        for(int entry=component_offsets(component_base+component);
                            entry<component_offsets(
                                component_base+component+1);++entry)
                            result+=values(entry)
                                *input_1(sample,input_1_offset
                                    +channel*input_width+indices(entry,0))
                                *input_2(sample,input_2_offset+indices(entry,1));
                        const int output_index=output_offset
                            +channel*output_width+component;
                        output(sample,output_index)+=
                            scale*result*mask(output_index);
                    }
                }
            });
        return;
    }
    for (const auto instruction:instructions) { auto wigner=instruction.wigner;
        Kokkos::parallel_for("e3 tensor product",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{static_cast<int>(input_1.extent(0)),instruction.output_multiplicity,instruction.output_width}),KOKKOS_LAMBDA(int sample,int u,int c) {
            double result=0.0; for(int v=0;v<instruction.multiplicity_2;++v) { if(instruction.uuu&&u!=v) continue; const int wi=instruction.weight_offset+(instruction.uuu?u:u*instruction.multiplicity_2+v); const double weight=instruction.has_weight?(dynamic_weights.extent(1)?dynamic_weights(sample,wi):fixed(wi)):1.0;
                for(int a=0;a<instruction.width_1;++a) for(int b=0;b<instruction.width_2;++b) result+=instruction.path_weight*weight*wigner((a*instruction.width_2+b)*instruction.output_width+c)*input_1(sample,instruction.input_1_offset+u*instruction.width_1+a)*input_2(sample,instruction.input_2_offset+v*instruction.width_2+b);
            } const int index=instruction.output_offset+u*instruction.output_width+c; output(sample,index)+=result*mask(index);
        });
    }
}

void E3TensorProductKokkos::reverse(Kokkos::View<const double**,Kokkos::LayoutRight> input_1,Kokkos::View<const double**,Kokkos::LayoutRight> input_2,Kokkos::View<const double**,Kokkos::LayoutRight> dynamic_weights,Kokkos::View<const double**,Kokkos::LayoutRight> output_adjoint,Kokkos::View<double**,Kokkos::LayoutRight> input_1_adjoint,Kokkos::View<double**,Kokkos::LayoutRight> input_2_adjoint,Kokkos::View<double**,Kokkos::LayoutRight> weights_adjoint) const
{
    Kokkos::deep_copy(input_1_adjoint,0.0); Kokkos::deep_copy(input_2_adjoint,0.0); Kokkos::deep_copy(weights_adjoint,0.0); auto mask=output_mask; auto fixed=internal_weights;
    if(mh1_fast_path) {
        auto plan=mh1_instruction_data;
        auto paths=mh1_path_weights;
        auto indices=mh1_sparse_indices;
        auto values=mh1_sparse_values;
        auto component_offsets=mh1_component_offsets;
        auto harmonic_offsets=mh1_harmonic_offsets;
        auto harmonic_terms=mh1_harmonic_terms;
        auto harmonic_values=mh1_harmonic_values;
        const int instruction_count=mh1_instruction_count_;
        Kokkos::parallel_for(
            "e3 tensor reverse mh1 fused channels",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                {0,0},{static_cast<int>(input_1.extent(0)),128}),
            KOKKOS_LAMBDA(int sample,int channel) {
                for(int instruction=0;instruction<instruction_count;
                    ++instruction) {
                    const int input_1_offset=plan(instruction,0);
                    const int input_2_offset=plan(instruction,1);
                    const int output_offset=plan(instruction,2);
                    const int input_width=plan(instruction,3);
                    const int output_width=plan(instruction,4);
                    const int weight_offset=plan(instruction,5);
                    const int component_base=plan(instruction,6);
                    const int first_entry=component_offsets(component_base);
                    const int last_entry=component_offsets(
                        component_base+output_width);
                    const double weight=dynamic_weights(
                        sample,weight_offset+channel);
                    double weight_value=0.0;
                    for(int entry=first_entry;entry<last_entry;++entry) {
                        const int a=indices(entry,0);
                        const int b=indices(entry,1);
                        const int c=indices(entry,2);
                        const int output_index=output_offset
                            +channel*output_width+c;
                        const double common=paths(instruction)*values(entry)
                            *mask(output_index)*output_adjoint(sample,output_index);
                        const double first=input_1(sample,input_1_offset
                            +channel*input_width+a);
                        const double second=input_2(sample,input_2_offset+b);
                        input_1_adjoint(sample,input_1_offset
                            +channel*input_width+a)+=common*weight*second;
                        weight_value+=common*first*second;
                    }
                    weights_adjoint(sample,weight_offset+channel)=weight_value;
                }
            });
        Kokkos::parallel_for(
            "e3 tensor reverse mh1 fused harmonics",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                {0,0},{static_cast<int>(input_1.extent(0)),input_2_dimension_}),
            KOKKOS_LAMBDA(int sample,int component) {
                double result=0.0;
                for(int channel=0;channel<128;++channel)
                    for(int term=harmonic_offsets(component);
                        term<harmonic_offsets(component+1);++term) {
                        const int instruction=harmonic_terms(term,0);
                        const int a=harmonic_terms(term,1);
                        const int c=harmonic_terms(term,2);
                        const int input_1_offset=plan(instruction,0);
                        const int output_offset=plan(instruction,2);
                        const int input_width=plan(instruction,3);
                        const int output_width=plan(instruction,4);
                        const int weight_offset=plan(instruction,5);
                        const int output_index=output_offset
                            +channel*output_width+c;
                        const double common=paths(instruction)
                            *harmonic_values(term)*mask(output_index)
                            *output_adjoint(sample,output_index);
                        result+=common
                            *dynamic_weights(sample,weight_offset+channel)
                            *input_1(sample,input_1_offset
                                +channel*input_width+a);
                    }
                input_2_adjoint(sample,component)=result;
            });
        return;
    }
    for(const auto instruction:instructions) { auto wigner=instruction.wigner;
        Kokkos::parallel_for("e3 tensor reverse",input_1.extent(0),KOKKOS_LAMBDA(int sample) {
            for(int u=0;u<instruction.multiplicity_1;++u) for(int v=0;v<instruction.multiplicity_2;++v) { if(instruction.uuu&&u!=v) continue; const int wi=instruction.weight_offset+(instruction.uuu?u:u*instruction.multiplicity_2+v); const double weight=instruction.has_weight?(dynamic_weights.extent(1)?dynamic_weights(sample,wi):fixed(wi)):1.0;
                for(int a=0;a<instruction.width_1;++a) for(int b=0;b<instruction.width_2;++b) for(int c=0;c<instruction.output_width;++c) { const int oi=instruction.output_offset+u*instruction.output_width+c; const double common=instruction.path_weight*wigner((a*instruction.width_2+b)*instruction.output_width+c)*mask(oi)*output_adjoint(sample,oi); input_1_adjoint(sample,instruction.input_1_offset+u*instruction.width_1+a)+=common*weight*input_2(sample,instruction.input_2_offset+v*instruction.width_2+b); input_2_adjoint(sample,instruction.input_2_offset+v*instruction.width_2+b)+=common*weight*input_1(sample,instruction.input_1_offset+u*instruction.width_1+a); if(instruction.has_weight) weights_adjoint(sample,wi)+=common*input_1(sample,instruction.input_1_offset+u*instruction.width_1+a)*input_2(sample,instruction.input_2_offset+v*instruction.width_2+b); }
            }
        });
    }
}
