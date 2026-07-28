#include "e3nn_kokkos.hpp"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#include "KokkosBlas.hpp"
#include "tools_kokkos.hpp"

namespace {
template<typename Precision>
std::vector<Precision> tensor_values(const nlohmann::json& value) { return value.at("values").get<std::vector<Precision>>(); }
int shape_product(const std::vector<int>& shape) { return std::accumulate(shape.begin(),shape.end(),1,std::multiplies<int>()); }

int mh1_cuda_team_size(int multiplicity)
{
    if(multiplicity<=0) return 0;
    constexpr int warp_size=32;
    constexpr int maximum_team_size=128;
    const int capped=std::min(multiplicity,maximum_team_size);
    return std::max(
        warp_size,((capped+warp_size-1)/warp_size)*warp_size);
}

class ProfileRegion {
public:
    explicit ProfileRegion(const char* name) { Kokkos::Profiling::pushRegion(name); }
    ~ProfileRegion() { Kokkos::Profiling::popRegion(); }
};

}

template<typename Precision>
E3LinearKokkosT<Precision>::E3LinearKokkosT(const nlohmann::json& data)
{
    E3Linear validated(data);
    Irreps input(data.at("irreps_in").get<std::string>()), output(data.at("irreps_out").get<std::string>());
    input_dimension_=input.dimension(); output_dimension_=output.dimension(); int offset=0;
    for (const auto& value:data.at("instructions")) {
        const auto& in=input.blocks.at(value.at("i_in").get<int>()); const auto& out=output.blocks.at(value.at("i_out").get<int>());
        const auto shape=value.at("path_shape").get<std::vector<int>>();
        instructions.push_back({in.offset,out.offset,in.multiplicity,out.multiplicity,2*in.l+1,offset,value.at("path_weight").get<Precision>()});
        offset+=shape_product(shape);
    }
    weights=toKokkosView("e3 linear weights",tensor_values<Precision>(data.at("weight")));
    bias=toKokkosView("e3 linear bias",tensor_values<Precision>(data.at("bias")));
    output_mask=toKokkosView("e3 linear output mask",tensor_values<Precision>(data.at("output_mask")));
    if (weights.size()!=offset) throw std::invalid_argument("Kokkos e3 linear weight size is invalid.");
}

template<typename Precision>
void E3LinearKokkosT<Precision>::set_backend(const std::string& backend)
{
    if(backend=="auto") backend_=Backend::automatic;
    else if(backend=="scalar") backend_=Backend::scalar;
    else if(backend=="packed_gemm") backend_=Backend::packed_gemm;
    else throw std::invalid_argument(
        "Kokkos E3 linear backend must be auto, scalar, or packed_gemm.");
}

template<typename Precision>
std::string E3LinearKokkosT<Precision>::backend() const
{
    if(backend_==Backend::scalar) return "scalar";
    if(backend_==Backend::packed_gemm) return "packed_gemm";
    return "auto";
}

template<typename Precision>
bool E3LinearKokkosT<Precision>::use_scalar_backend(std::size_t samples) const
{
    if(backend_==Backend::scalar) return true;
    if(backend_==Backend::packed_gemm) return false;
#ifdef KOKKOS_ENABLE_CUDA
    return true;
#else
    return samples<=4;
#endif
}

template<typename Precision>
std::string E3LinearKokkosT<Precision>::selected_backend(
    std::size_t samples) const
{
    return use_scalar_backend(samples) ? "scalar" : "packed_gemm";
}

template<typename Precision>
std::size_t E3LinearKokkosT<Precision>::workspace_bytes() const
{
    return workspace_->bytes();
}

template<typename Precision>
void E3LinearKokkosT<Precision>::evaluate(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,Kokkos::View<Precision**,Kokkos::LayoutRight> output) const
{
    if(input.extent(1)!=static_cast<std::size_t>(input_dimension_)
        ||output.extent(0)!=input.extent(0)
        ||output.extent(1)!=static_cast<std::size_t>(output_dimension_))
        throw std::invalid_argument("Kokkos e3 linear batch dimensions are inconsistent.");
    ProfileRegion profile("symmetrix/e3_linear/forward");
    ordered_kokkos_deep_copy(output,Precision(0)); auto all_weights=weights;
    const bool use_scalar_path=use_scalar_backend(input.extent(0));
    if(use_scalar_path) {
        for(const auto instruction:instructions)
            Kokkos::parallel_for(
                "e3 linear small batch",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                    {0,0,0},{static_cast<int>(input.extent(0)),
                        instruction.output_multiplicity,instruction.width}),
                KOKKOS_LAMBDA(int sample,int target,int component) {
                    Precision value=Precision(0);
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
                +(all_bias.size()?all_bias(column):Precision(0)))*mask(column);
        });
        return;
    }
    for (const auto& instruction:instructions) {
        const int samples=input.extent(0);
        const int columns=samples*instruction.width;
        if(workspace_->packed_input.extent(0)
                <static_cast<std::size_t>(instruction.input_multiplicity)
            ||workspace_->packed_input.extent(1)<static_cast<std::size_t>(columns))
            Kokkos::realloc(
                workspace_->packed_input,
                std::max<std::size_t>(workspace_->packed_input.extent(0),
                    instruction.input_multiplicity),
                std::max<std::size_t>(workspace_->packed_input.extent(1),columns));
        if(workspace_->packed_output.extent(0)
                <static_cast<std::size_t>(instruction.output_multiplicity)
            ||workspace_->packed_output.extent(1)<static_cast<std::size_t>(columns))
            Kokkos::realloc(
                workspace_->packed_output,
                std::max<std::size_t>(workspace_->packed_output.extent(0),
                    instruction.output_multiplicity),
                std::max<std::size_t>(workspace_->packed_output.extent(1),columns));
        auto packed_input=Kokkos::subview(
            workspace_->packed_input,
            std::make_pair(0,instruction.input_multiplicity),
            std::make_pair(0,columns));
        auto packed_output=Kokkos::subview(
            workspace_->packed_output,
            std::make_pair(0,instruction.output_multiplicity),
            std::make_pair(0,columns));
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
            const Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
        weight_matrix weight(
            all_weights.data()+instruction.weight_offset,
            instruction.input_multiplicity,instruction.output_multiplicity);
        KokkosBlas::gemm(
            "T","N",instruction.path_weight,weight,packed_input,Precision(0),packed_output);
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
    Kokkos::parallel_for("e3 linear mask",output.size(),KOKKOS_LAMBDA(int flat) { const int sample=flat/output.extent(1),column=flat%output.extent(1); output(sample,column)=(output(sample,column)+(all_bias.size()?all_bias(column):Precision(0)))*mask(column); });
}

template<typename Precision>
void E3LinearKokkosT<Precision>::reverse(Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint) const
{
    if(output_adjoint.extent(1)!=static_cast<std::size_t>(output_dimension_)
        ||input_adjoint.extent(0)!=output_adjoint.extent(0)
        ||input_adjoint.extent(1)!=static_cast<std::size_t>(input_dimension_))
        throw std::invalid_argument("Kokkos e3 linear reverse batch dimensions are inconsistent.");
    ProfileRegion profile("symmetrix/e3_linear/reverse");
    ordered_kokkos_deep_copy(input_adjoint,Precision(0)); auto all_weights=weights; auto mask=output_mask;
    const bool use_scalar_path=use_scalar_backend(output_adjoint.extent(0));
    if(use_scalar_path) {
        for(const auto instruction:instructions)
            Kokkos::parallel_for(
                "e3 reverse linear small batch",
                Kokkos::MDRangePolicy<Kokkos::Rank<3>>(
                    {0,0,0},{static_cast<int>(output_adjoint.extent(0)),
                        instruction.input_multiplicity,instruction.width}),
                KOKKOS_LAMBDA(int sample,int source,int component) {
                    Precision value=Precision(0);
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
        if(workspace_->packed_input.extent(0)
                <static_cast<std::size_t>(instruction.input_multiplicity)
            ||workspace_->packed_input.extent(1)<static_cast<std::size_t>(columns))
            Kokkos::realloc(
                workspace_->packed_input,
                std::max<std::size_t>(workspace_->packed_input.extent(0),
                    instruction.input_multiplicity),
                std::max<std::size_t>(workspace_->packed_input.extent(1),columns));
        if(workspace_->packed_output.extent(0)
                <static_cast<std::size_t>(instruction.output_multiplicity)
            ||workspace_->packed_output.extent(1)<static_cast<std::size_t>(columns))
            Kokkos::realloc(
                workspace_->packed_output,
                std::max<std::size_t>(workspace_->packed_output.extent(0),
                    instruction.output_multiplicity),
                std::max<std::size_t>(workspace_->packed_output.extent(1),columns));
        auto packed_input=Kokkos::subview(
            workspace_->packed_input,
            std::make_pair(0,instruction.input_multiplicity),
            std::make_pair(0,columns));
        auto packed_output=Kokkos::subview(
            workspace_->packed_output,
            std::make_pair(0,instruction.output_multiplicity),
            std::make_pair(0,columns));
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
            const Precision**,Kokkos::LayoutRight,Kokkos::MemoryUnmanaged>;
        weight_matrix weight(
            all_weights.data()+instruction.weight_offset,
            instruction.input_multiplicity,instruction.output_multiplicity);
        KokkosBlas::gemm(
            "N","N",instruction.path_weight,weight,packed_output,Precision(0),packed_input);
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

template<typename Precision>
E3TensorProductKokkosT<Precision>::E3TensorProductKokkosT(const nlohmann::json& data)
{
    E3TensorProduct validated(data);
    Irreps in1(data.at("irreps_in1").get<std::string>()),in2(data.at("irreps_in2").get<std::string>()),out(data.at("irreps_out").get<std::string>());
    input_1_dimension_=in1.dimension(); input_2_dimension_=in2.dimension(); output_dimension_=out.dimension(); int offset=0;
    bool official_layout=!validated.instructions.empty();
    std::vector<int> mh1_instruction_data_host;
    std::vector<Precision> mh1_path_weights_host;
    std::vector<int> mh1_component_offsets_host;
    std::vector<int> mh1_sparse_indices_host;
    std::vector<Precision> mh1_sparse_values_host;
    struct HarmonicTerm { int instruction,a,c; Precision value; };
    std::vector<std::vector<HarmonicTerm>> mh1_harmonic_terms_host(
        input_2_dimension_);
    int instruction_index=0;
    for (const auto& value:data.at("instructions")) {
        const auto& a=in1.blocks.at(value.at("i_in1").get<int>()); const auto& b=in2.blocks.at(value.at("i_in2").get<int>()); const auto& c=out.blocks.at(value.at("i_out").get<int>());
        const bool has=value.at("has_weight").get<bool>(); const auto mode=value.at("connection_mode").get<std::string>();
        if (mode!="uvu"&&mode!="uuu") throw std::invalid_argument("Unsupported Kokkos tensor-product mode: "+mode);
        Instruction instruction{a.offset,b.offset,c.offset,a.multiplicity,b.multiplicity,c.multiplicity,2*a.l+1,2*b.l+1,2*c.l+1,offset,has,mode=="uuu",value.at("path_weight").get<Precision>(),toKokkosView("e3 wigner",tensor_values<Precision>(value.at("wigner_3j")))};
        const auto path_shape=value.at("path_shape").get<std::vector<int>>();
        const int instruction_multiplicity=a.multiplicity;
        official_layout=official_layout&&mode=="uvu"&&has
            &&instruction_multiplicity>0&&b.multiplicity==1
            &&c.multiplicity==instruction_multiplicity
            &&(mh1_multiplicity_==0||mh1_multiplicity_==instruction_multiplicity)
            &&instruction.width_1<=3
            &&path_shape==std::vector<int>({instruction_multiplicity,1});
        if(official_layout) {
            mh1_multiplicity_=instruction_multiplicity;
            const auto& entries=validated.instructions.at(instruction_index).nonzero_wigner;
            std::vector<int> component_offsets(instruction.output_width+1,0);
            std::vector<int> indices;
            std::vector<Precision> values;
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
    weight_size_=offset; internal_weights=toKokkosView("e3 tensor weights",tensor_values<Precision>(data.at("weight"))); output_mask=toKokkosView("e3 tensor mask",tensor_values<Precision>(data.at("output_mask")));
    mh1_fast_path=official_layout&&internal_weights.empty();
    if(mh1_fast_path) {
        std::vector<int> input_component_data;
        mh1_direct_node_layout_=true;
        for(const auto& block:in1.blocks) {
            if(block.multiplicity!=mh1_multiplicity_) {
                mh1_direct_node_layout_=false;
                break;
            }
            const int width=2*block.l+1;
            for(int component=0;component<width;++component) {
                input_component_data.push_back(block.offset+component);
                input_component_data.push_back(width);
            }
        }
        mh1_input_1_angular_dimension_=input_component_data.size()/2;
        mh1_direct_node_layout_=mh1_direct_node_layout_
            &&mh1_input_1_angular_dimension_>0
            &&mh1_input_1_angular_dimension_
                <=max_direct_node_input_components_;
        if(mh1_direct_node_layout_)
            set_kokkos_view(
                mh1_input_component_data,input_component_data,
                mh1_input_1_angular_dimension_,2);
        mh1_cuda_channel_team_size_=mh1_cuda_team_size(mh1_multiplicity_);
        mh1_cuda_harmonic_team_size_=mh1_cuda_team_size(mh1_multiplicity_);
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
        std::vector<Precision> harmonic_values;
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

template<typename Precision>
std::string E3TensorProductKokkosT<Precision>::execution_backend() const
{
    if(!mh1_fast_path) return "generic_kokkos";
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Precision,float>
        &&std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>)
        return "official_cuda_team";
#endif
    return "official_kokkos_mdrange";
}

template<typename Precision>
int E3TensorProductKokkosT<Precision>::channel_team_size() const
{
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Precision,float>
        &&std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>)
        return mh1_fast_path ? mh1_cuda_channel_team_size_ : 0;
#endif
    return 0;
}

template<typename Precision>
int E3TensorProductKokkosT<Precision>::harmonic_team_size() const
{
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Precision,float>
        &&std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>)
        return mh1_fast_path ? mh1_cuda_harmonic_team_size_ : 0;
#endif
    return 0;
}

template<typename Precision>
bool E3TensorProductKokkosT<Precision>::supports_direct_node_reverse() const
{
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Precision,float>
        &&std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>)
        return mh1_fast_path&&mh1_direct_node_layout_;
#endif
    return false;
}

template<typename Precision>
void E3TensorProductKokkosT<Precision>::evaluate(Kokkos::View<const Precision**,Kokkos::LayoutRight> input_1,Kokkos::View<const Precision**,Kokkos::LayoutRight> input_2,Kokkos::View<const Precision**,Kokkos::LayoutRight> dynamic_weights,Kokkos::View<Precision**,Kokkos::LayoutRight> output) const
{
    ProfileRegion profile(mh1_fast_path
        ? "symmetrix/mh1/tensor_product/forward"
        : "symmetrix/generic/tensor_product/forward");
    ordered_kokkos_deep_copy(output,Precision(0)); auto mask=output_mask; auto fixed=internal_weights;
    if(mh1_fast_path) {
        auto plan=mh1_instruction_data;
        auto paths=mh1_path_weights;
        auto indices=mh1_sparse_indices;
        auto values=mh1_sparse_values;
        auto component_offsets=mh1_component_offsets;
        const int instruction_count=mh1_instruction_count_;
        const int multiplicity=mh1_multiplicity_;
#ifdef KOKKOS_ENABLE_CUDA
        if constexpr(std::is_same_v<Precision,float>
            &&std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>) {
        const int channel_team_size=mh1_cuda_channel_team_size_;
        using channel_policy=Kokkos::TeamPolicy<>;
        Kokkos::parallel_for(
            "e3 tensor product mh1 team channels",
            channel_policy(input_1.extent(0),channel_team_size),
            KOKKOS_LAMBDA(const typename channel_policy::member_type& team) {
                const int sample=team.league_rank();
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team,multiplicity),
                    [&](const int channel) {
                for(int instruction=0;instruction<instruction_count;
                    ++instruction) {
                    const int input_1_offset=plan(instruction,0);
                    const int input_2_offset=plan(instruction,1);
                    const int output_offset=plan(instruction,2);
                    const int input_width=plan(instruction,3);
                    const int output_width=plan(instruction,4);
                    const int weight_offset=plan(instruction,5);
                    const int component_base=plan(instruction,6);
                    const Precision scale=paths(instruction)
                        *dynamic_weights(sample,weight_offset+channel);
                    for(int component=0;component<output_width;++component) {
                        Precision result=Precision(0);
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
            });
        return;
        }
#endif
        Kokkos::parallel_for(
            "e3 tensor product mh1 fused",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                {0,0},{static_cast<int>(input_1.extent(0)),multiplicity}),
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
                    const Precision scale=paths(instruction)
                        *dynamic_weights(sample,weight_offset+channel);
                    for(int component=0;component<output_width;++component) {
                        Precision result=Precision(0);
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
            Precision result=Precision(0); for(int v=0;v<instruction.multiplicity_2;++v) { if(instruction.uuu&&u!=v) continue; const int wi=instruction.weight_offset+(instruction.uuu?u:u*instruction.multiplicity_2+v); const Precision weight=instruction.has_weight?(dynamic_weights.extent(1)?dynamic_weights(sample,wi):fixed(wi)):Precision(1);
                for(int a=0;a<instruction.width_1;++a) for(int b=0;b<instruction.width_2;++b) result+=instruction.path_weight*weight*wigner((a*instruction.width_2+b)*instruction.output_width+c)*input_1(sample,instruction.input_1_offset+u*instruction.width_1+a)*input_2(sample,instruction.input_2_offset+v*instruction.width_2+b);
            } const int index=instruction.output_offset+u*instruction.output_width+c; output(sample,index)+=result*mask(index);
        });
    }
}

template<typename Precision>
void E3TensorProductKokkosT<Precision>::reverse(Kokkos::View<const Precision**,Kokkos::LayoutRight> input_1,Kokkos::View<const Precision**,Kokkos::LayoutRight> input_2,Kokkos::View<const Precision**,Kokkos::LayoutRight> dynamic_weights,Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,Kokkos::View<Precision**,Kokkos::LayoutRight> input_1_adjoint,Kokkos::View<Precision**,Kokkos::LayoutRight> input_2_adjoint,Kokkos::View<Precision**,Kokkos::LayoutRight> weights_adjoint) const
{
    ProfileRegion profile(mh1_fast_path
        ? "symmetrix/mh1/tensor_product/reverse"
        : "symmetrix/generic/tensor_product/reverse");
    ordered_kokkos_deep_copy(input_1_adjoint,Precision(0));
    ordered_kokkos_deep_copy(input_2_adjoint,Precision(0));
    ordered_kokkos_deep_copy(weights_adjoint,Precision(0));
    auto mask=output_mask; auto fixed=internal_weights;
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
        const int multiplicity=mh1_multiplicity_;
#ifdef KOKKOS_ENABLE_CUDA
        if constexpr(std::is_same_v<Precision,float>
            &&std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>) {
        const int channel_team_size=mh1_cuda_channel_team_size_;
        const int harmonic_team_size=mh1_cuda_harmonic_team_size_;
        using channel_policy=Kokkos::TeamPolicy<>;
        Kokkos::parallel_for(
            "e3 tensor reverse mh1 team channels",
            channel_policy(input_1.extent(0),channel_team_size),
            KOKKOS_LAMBDA(const typename channel_policy::member_type& team) {
                const int sample=team.league_rank();
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team,multiplicity),
                    [&](const int channel) {
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
                        const Precision weight=dynamic_weights(
                            sample,weight_offset+channel);
                        Precision weight_value=Precision(0);
                        for(int entry=first_entry;entry<last_entry;++entry) {
                            const int a=indices(entry,0);
                            const int b=indices(entry,1);
                            const int c=indices(entry,2);
                            const int output_index=output_offset
                                +channel*output_width+c;
                            const Precision common=paths(instruction)*values(entry)
                                *mask(output_index)*output_adjoint(sample,output_index);
                            const Precision first=input_1(sample,input_1_offset
                                +channel*input_width+a);
                            const Precision second=input_2(sample,input_2_offset+b);
                            input_1_adjoint(sample,input_1_offset
                                +channel*input_width+a)+=common*weight*second;
                            weight_value+=common*first*second;
                        }
                        weights_adjoint(sample,weight_offset+channel)=weight_value;
                }
                });
            });
        using harmonic_policy=Kokkos::TeamPolicy<>;
        const int harmonic_dimension=input_2_dimension_;
        Kokkos::parallel_for(
            "e3 tensor reverse mh1 team harmonics",
            harmonic_policy(
                input_1.extent(0)*harmonic_dimension,harmonic_team_size),
            KOKKOS_LAMBDA(const typename harmonic_policy::member_type& team) {
                const int sample=team.league_rank()/harmonic_dimension;
                const int component=team.league_rank()%harmonic_dimension;
                Precision result=Precision(0);
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team,multiplicity),
                    [&](const int channel,Precision& channel_result) {
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
                        const Precision common=paths(instruction)
                            *harmonic_values(term)*mask(output_index)
                            *output_adjoint(sample,output_index);
                        channel_result+=common
                            *dynamic_weights(sample,weight_offset+channel)
                            *input_1(sample,input_1_offset
                                +channel*input_width+a);
                    }},result);
                Kokkos::single(Kokkos::PerTeam(team),[&]() {
                    input_2_adjoint(sample,component)=result;
                });
            });
        return;
        }
#endif
        Kokkos::parallel_for(
            "e3 tensor reverse mh1 fused channels",
            Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
                {0,0},{static_cast<int>(input_1.extent(0)),multiplicity}),
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
                    const Precision weight=dynamic_weights(
                        sample,weight_offset+channel);
                    Precision weight_value=Precision(0);
                    for(int entry=first_entry;entry<last_entry;++entry) {
                        const int a=indices(entry,0);
                        const int b=indices(entry,1);
                        const int c=indices(entry,2);
                        const int output_index=output_offset
                            +channel*output_width+c;
                        const Precision common=paths(instruction)*values(entry)
                            *mask(output_index)*output_adjoint(sample,output_index);
                        const Precision first=input_1(sample,input_1_offset
                            +channel*input_width+a);
                        const Precision second=input_2(sample,input_2_offset+b);
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
                Precision result=Precision(0);
                for(int channel=0;channel<multiplicity;++channel)
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
                        const Precision common=paths(instruction)
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
            for(int u=0;u<instruction.multiplicity_1;++u) for(int v=0;v<instruction.multiplicity_2;++v) { if(instruction.uuu&&u!=v) continue; const int wi=instruction.weight_offset+(instruction.uuu?u:u*instruction.multiplicity_2+v); const Precision weight=instruction.has_weight?(dynamic_weights.extent(1)?dynamic_weights(sample,wi):fixed(wi)):Precision(1);
                for(int a=0;a<instruction.width_1;++a) for(int b=0;b<instruction.width_2;++b) for(int c=0;c<instruction.output_width;++c) { const int oi=instruction.output_offset+u*instruction.output_width+c; const Precision common=instruction.path_weight*wigner((a*instruction.width_2+b)*instruction.output_width+c)*mask(oi)*output_adjoint(sample,oi); input_1_adjoint(sample,instruction.input_1_offset+u*instruction.width_1+a)+=common*weight*input_2(sample,instruction.input_2_offset+v*instruction.width_2+b); input_2_adjoint(sample,instruction.input_2_offset+v*instruction.width_2+b)+=common*weight*input_1(sample,instruction.input_1_offset+u*instruction.width_1+a); if(instruction.has_weight) weights_adjoint(sample,wi)+=common*input_1(sample,instruction.input_1_offset+u*instruction.width_1+a)*input_2(sample,instruction.input_2_offset+v*instruction.width_2+b); }
            }
        });
    }
}

template<typename Precision>
bool E3TensorProductKokkosT<Precision>::try_reverse_from_nodes(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> source_node_values,
    Kokkos::View<const int*> source_indices,
    int first_edge,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> edge_input_2,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> edge_weights,
    Kokkos::View<const Precision**,Kokkos::LayoutRight>
        target_node_output_adjoint,
    Kokkos::View<const int*> target_indices,
    Kokkos::View<Precision**,Kokkos::LayoutRight> source_node_input_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> edge_input_2_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> edge_weights_adjoint) const
{
    if(!supports_direct_node_reverse()) return false;
#ifdef KOKKOS_ENABLE_CUDA
    if constexpr(std::is_same_v<Precision,float>
        &&std::is_same_v<Kokkos::DefaultExecutionSpace,Kokkos::Cuda>) {
        const std::size_t samples=edge_input_2.extent(0);
        const std::size_t last_edge=first_edge<0
            ?std::size_t(0):static_cast<std::size_t>(first_edge)+samples;
        if(first_edge<0||source_indices.extent(0)<last_edge
            ||target_indices.extent(0)<last_edge
            ||source_node_values.extent(1)
                !=static_cast<std::size_t>(input_1_dimension_)
            ||target_node_output_adjoint.extent(1)
                !=static_cast<std::size_t>(output_dimension_)
            ||source_node_input_adjoint.extent(0)
                !=source_node_values.extent(0)
            ||source_node_input_adjoint.extent(1)
                !=source_node_values.extent(1)
            ||edge_input_2.extent(1)
                !=static_cast<std::size_t>(input_2_dimension_)
            ||edge_weights.extent(0)!=samples
            ||edge_weights.extent(1)
                !=static_cast<std::size_t>(weight_size_)
            ||edge_input_2_adjoint.extent(0)!=samples
            ||edge_input_2_adjoint.extent(1)!=edge_input_2.extent(1)
            ||edge_weights_adjoint.extent(0)!=samples
            ||edge_weights_adjoint.extent(1)!=edge_weights.extent(1))
            throw std::invalid_argument(
                "Kokkos direct-node tensor-product reverse dimensions are inconsistent.");

        ProfileRegion profile("symmetrix/mh1/tensor_product/reverse_direct_nodes");
        auto plan=mh1_instruction_data;
        auto paths=mh1_path_weights;
        auto indices=mh1_sparse_indices;
        auto values=mh1_sparse_values;
        auto component_offsets=mh1_component_offsets;
        auto harmonic_offsets=mh1_harmonic_offsets;
        auto harmonic_terms=mh1_harmonic_terms;
        auto harmonic_values=mh1_harmonic_values;
        auto input_components=mh1_input_component_data;
        auto mask=output_mask;
        const int instruction_count=mh1_instruction_count_;
        const int multiplicity=mh1_multiplicity_;
        const int angular_dimension=mh1_input_1_angular_dimension_;

        using channel_policy=Kokkos::TeamPolicy<>;
        Kokkos::parallel_for(
            "e3 tensor reverse mh1 direct node channels",
            channel_policy(samples,mh1_cuda_channel_team_size_),
            KOKKOS_LAMBDA(const typename channel_policy::member_type& team) {
                const int sample=team.league_rank();
                const int edge=first_edge+sample;
                const int source=source_indices(edge);
                const int target=target_indices(edge);
                Kokkos::parallel_for(
                    Kokkos::TeamThreadRange(team,multiplicity),
                    [&](const int channel) {
                    Precision input_values[max_direct_node_input_components_];
                    for(int component=0;component<angular_dimension;++component)
                        input_values[component]=Precision(0);
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
                        const Precision weight=edge_weights(
                            sample,weight_offset+channel);
                        Precision weight_value=Precision(0);
                        for(int entry=first_entry;entry<last_entry;++entry) {
                            const int a=indices(entry,0);
                            const int b=indices(entry,1);
                            const int c=indices(entry,2);
                            const int output_index=output_offset
                                +channel*output_width+c;
                            const Precision common=paths(instruction)*values(entry)
                                *mask(output_index)
                                *target_node_output_adjoint(target,output_index);
                            const Precision first=source_node_values(
                                source,input_1_offset+channel*input_width+a);
                            const Precision second=edge_input_2(
                                sample,input_2_offset+b);
                            input_values[input_1_offset/multiplicity+a]
                                +=common*weight*second;
                            weight_value+=common*first*second;
                        }
                        edge_weights_adjoint(
                            sample,weight_offset+channel)=weight_value;
                    }
                    for(int component=0;component<angular_dimension;++component) {
                        const Precision value=input_values[component];
                        if(value!=Precision(0))
                            Kokkos::atomic_add(
                                &source_node_input_adjoint(
                                    source,input_components(component,0)
                                        +channel*input_components(component,1)),
                                value);
                    }
                });
            });

        using harmonic_policy=Kokkos::TeamPolicy<>;
        const int harmonic_dimension=input_2_dimension_;
        Kokkos::parallel_for(
            "e3 tensor reverse mh1 direct node harmonics",
            harmonic_policy(
                samples*harmonic_dimension,mh1_cuda_harmonic_team_size_),
            KOKKOS_LAMBDA(const typename harmonic_policy::member_type& team) {
                const int sample=team.league_rank()/harmonic_dimension;
                const int component=team.league_rank()%harmonic_dimension;
                const int edge=first_edge+sample;
                const int source=source_indices(edge);
                const int target=target_indices(edge);
                Precision result=Precision(0);
                Kokkos::parallel_reduce(
                    Kokkos::TeamThreadRange(team,multiplicity),
                    [&](const int channel,Precision& channel_result) {
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
                        const Precision common=paths(instruction)
                            *harmonic_values(term)*mask(output_index)
                            *target_node_output_adjoint(target,output_index);
                        channel_result+=common
                            *edge_weights(sample,weight_offset+channel)
                            *source_node_values(
                                source,input_1_offset+channel*input_width+a);
                    }},result);
                Kokkos::single(Kokkos::PerTeam(team),[&]() {
                    edge_input_2_adjoint(sample,component)=result;
                });
            });
        return true;
    }
#endif
    return false;
}

template class E3LinearKokkosT<float>;
template class E3LinearKokkosT<double>;
template class E3TensorProductKokkosT<float>;
template class E3TensorProductKokkosT<double>;
