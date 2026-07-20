#include "e3nn_kokkos.hpp"

#include <numeric>
#include <stdexcept>

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
    Kokkos::deep_copy(output,0.0); auto all_weights=weights;
    for (const auto instruction:instructions) {
        Kokkos::parallel_for("e3 linear",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{static_cast<int>(input.extent(0)),instruction.output_multiplicity,instruction.width}),
            KOKKOS_LAMBDA(int sample,int target,int component) {
                double value=0.0; for (int source=0;source<instruction.input_multiplicity;++source)
                    value+=instruction.path_weight*all_weights(instruction.weight_offset+source*instruction.output_multiplicity+target)*input(sample,instruction.input_offset+source*instruction.width+component);
                output(sample,instruction.output_offset+target*instruction.width+component)+=value;
            });
    }
    auto mask=output_mask; auto all_bias=bias;
    Kokkos::parallel_for("e3 linear mask",output.size(),KOKKOS_LAMBDA(int flat) { const int sample=flat/output.extent(1),column=flat%output.extent(1); output(sample,column)=(output(sample,column)+(all_bias.size()?all_bias(column):0.0))*mask(column); });
}

void E3LinearKokkos::reverse(Kokkos::View<const double**,Kokkos::LayoutRight> output_adjoint,Kokkos::View<double**,Kokkos::LayoutRight> input_adjoint) const
{
    Kokkos::deep_copy(input_adjoint,0.0); auto all_weights=weights; auto mask=output_mask;
    for (const auto instruction:instructions)
        Kokkos::parallel_for("e3 reverse linear",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{static_cast<int>(output_adjoint.extent(0)),instruction.input_multiplicity,instruction.width}),
            KOKKOS_LAMBDA(int sample,int source,int component) {
                double value=0.0; for (int target=0;target<instruction.output_multiplicity;++target) {
                    const int out=instruction.output_offset+target*instruction.width+component;
                    value+=instruction.path_weight*all_weights(instruction.weight_offset+source*instruction.output_multiplicity+target)*mask(out)*output_adjoint(sample,out);
                } input_adjoint(sample,instruction.input_offset+source*instruction.width+component)+=value;
            });
}

E3TensorProductKokkos::E3TensorProductKokkos(const nlohmann::json& data)
{
    E3TensorProduct validated(data);
    Irreps in1(data.at("irreps_in1").get<std::string>()),in2(data.at("irreps_in2").get<std::string>()),out(data.at("irreps_out").get<std::string>());
    input_1_dimension_=in1.dimension(); input_2_dimension_=in2.dimension(); output_dimension_=out.dimension(); int offset=0;
    for (const auto& value:data.at("instructions")) {
        const auto& a=in1.blocks.at(value.at("i_in1").get<int>()); const auto& b=in2.blocks.at(value.at("i_in2").get<int>()); const auto& c=out.blocks.at(value.at("i_out").get<int>());
        const bool has=value.at("has_weight").get<bool>(); const auto mode=value.at("connection_mode").get<std::string>();
        if (mode!="uvu"&&mode!="uuu") throw std::invalid_argument("Unsupported Kokkos tensor-product mode: "+mode);
        Instruction instruction{a.offset,b.offset,c.offset,a.multiplicity,b.multiplicity,c.multiplicity,2*a.l+1,2*b.l+1,2*c.l+1,offset,has,mode=="uuu",value.at("path_weight").get<double>(),toKokkosView("e3 wigner",tensor_values(value.at("wigner_3j")))};
        instructions.push_back(instruction); if(has) offset+=shape_product(value.at("path_shape").get<std::vector<int>>());
    }
    weight_size_=offset; internal_weights=toKokkosView("e3 tensor weights",tensor_values(data.at("weight"))); output_mask=toKokkosView("e3 tensor mask",tensor_values(data.at("output_mask")));
}

void E3TensorProductKokkos::evaluate(Kokkos::View<const double**,Kokkos::LayoutRight> input_1,Kokkos::View<const double**,Kokkos::LayoutRight> input_2,Kokkos::View<const double**,Kokkos::LayoutRight> dynamic_weights,Kokkos::View<double**,Kokkos::LayoutRight> output) const
{
    Kokkos::deep_copy(output,0.0); auto mask=output_mask; auto fixed=internal_weights;
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
    for(const auto instruction:instructions) { auto wigner=instruction.wigner;
        Kokkos::parallel_for("e3 tensor reverse",input_1.extent(0),KOKKOS_LAMBDA(int sample) {
            for(int u=0;u<instruction.multiplicity_1;++u) for(int v=0;v<instruction.multiplicity_2;++v) { if(instruction.uuu&&u!=v) continue; const int wi=instruction.weight_offset+(instruction.uuu?u:u*instruction.multiplicity_2+v); const double weight=instruction.has_weight?(dynamic_weights.extent(1)?dynamic_weights(sample,wi):fixed(wi)):1.0;
                for(int a=0;a<instruction.width_1;++a) for(int b=0;b<instruction.width_2;++b) for(int c=0;c<instruction.output_width;++c) { const int oi=instruction.output_offset+u*instruction.output_width+c; const double common=instruction.path_weight*wigner((a*instruction.width_2+b)*instruction.output_width+c)*mask(oi)*output_adjoint(sample,oi); input_1_adjoint(sample,instruction.input_1_offset+u*instruction.width_1+a)+=common*weight*input_2(sample,instruction.input_2_offset+v*instruction.width_2+b); input_2_adjoint(sample,instruction.input_2_offset+v*instruction.width_2+b)+=common*weight*input_1(sample,instruction.input_1_offset+u*instruction.width_1+a); if(instruction.has_weight) weights_adjoint(sample,wi)+=common*input_1(sample,instruction.input_1_offset+u*instruction.width_1+a)*input_2(sample,instruction.input_2_offset+v*instruction.width_2+b); }
            }
        });
    }
}
