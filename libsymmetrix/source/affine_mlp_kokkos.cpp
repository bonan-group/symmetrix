#include "affine_mlp_kokkos.hpp"
#include "affine_mlp.hpp"

#include <stdexcept>
#include <utility>
#include <vector>

#include "KokkosBlas.hpp"
#include "tools_kokkos.hpp"

namespace {
template<class Destination,class Source>
void affine_deep_copy(Destination destination,Source source)
{
#ifdef KOKKOS_ENABLE_CUDA
    Kokkos::deep_copy(Kokkos::DefaultExecutionSpace{},destination,source);
#else
    Kokkos::deep_copy(destination,source);
#endif
}

template<typename Precision>
std::vector<Precision> tensor_values(const nlohmann::json& value)
{
    return value.at("values").get<std::vector<Precision>>();
}
}

template<typename Precision>
AffineMLPKokkosT<Precision>::AffineMLPKokkosT(const nlohmann::json& definition)
{
    AffineMLP validated(definition);
    const int count = definition.at("layers").size();
    types = Kokkos::View<int*,Kokkos::SharedSpace>("affine mlp types",count);
    input_sizes = Kokkos::View<int*,Kokkos::SharedSpace>("affine mlp inputs",count);
    output_sizes = Kokkos::View<int*,Kokkos::SharedSpace>("affine mlp outputs",count);
    eps = Kokkos::View<Precision*,Kokkos::SharedSpace>("affine mlp eps",count);
    weights = decltype(weights)(Kokkos::view_alloc("affine mlp weights",Kokkos::SequentialHostInit),count);
    biases = decltype(biases)(Kokkos::view_alloc("affine mlp biases",Kokkos::SequentialHostInit),count);
    values = decltype(values)(Kokkos::view_alloc("affine mlp values",Kokkos::SequentialHostInit),count+1);
    adjoints = decltype(adjoints)(Kokkos::view_alloc("affine mlp adjoints",Kokkos::SequentialHostInit),count+1);
    value_storage = decltype(value_storage)(
        Kokkos::view_alloc("affine mlp value storage",Kokkos::SequentialHostInit),count+1);
    adjoint_storage = decltype(adjoint_storage)(
        Kokkos::view_alloc("affine mlp adjoint storage",Kokkos::SequentialHostInit),count+1);
    int previous = -1;
    for (int index=0; index<count; ++index) {
        const auto& layer = definition.at("layers").at(index);
        const auto type = layer.at("type").get<std::string>();
        if (type == "linear") {
            types(index) = Linear;
            const auto shape = layer.at("weight").at("shape").get<std::vector<int>>();
            if (shape.size()!=2) throw std::invalid_argument("Kokkos affine linear weight must be rank two.");
            input_sizes(index)=shape[1]; output_sizes(index)=shape[0];
            weights(index)=toKokkosView("affine mlp weight",tensor_values<Precision>(layer.at("weight")),shape[0],shape[1]);
            biases(index)=toKokkosView("affine mlp bias",tensor_values<Precision>(layer.at("bias")));
        } else if (type == "layer_norm") {
            types(index)=LayerNorm;
            input_sizes(index)=layer.at("normalized_shape").at(0).get<int>();
            output_sizes(index)=input_sizes(index); eps(index)=layer.at("eps").get<Precision>();
            const auto gamma=tensor_values<Precision>(layer.at("weight"));
            weights(index)=toKokkosView("affine layer norm gamma",gamma,1,gamma.size());
            biases(index)=toKokkosView("affine layer norm beta",tensor_values<Precision>(layer.at("bias")));
        } else if (type == "silu") {
            if (previous<0) throw std::invalid_argument("Kokkos affine SiLU cannot be the first layer.");
            types(index)=SiLU; input_sizes(index)=previous; output_sizes(index)=previous;
        } else throw std::invalid_argument("Unsupported Kokkos affine MLP layer: "+type);
        if (previous>=0 && input_sizes(index)!=previous)
            throw std::invalid_argument("Kokkos affine MLP layer dimensions are inconsistent.");
        previous=output_sizes(index);
    }
}

template<typename Precision>
int AffineMLPKokkosT<Precision>::input_size() const { return input_sizes(0); }
template<typename Precision>
int AffineMLPKokkosT<Precision>::output_size() const { return output_sizes(output_sizes.size()-1); }
template<typename Precision>
bool AffineMLPKokkosT<Precision>::supports_conditioned_input(int dynamic_input_size) const
{
    return types.size()>0&&types(0)==Linear&&dynamic_input_size>0
        &&dynamic_input_size<input_size();
}

template<typename Precision>
int AffineMLPKokkosT<Precision>::workspace_rows() const
{
    int rows=0;
    for(int index=0;index<value_storage.extent_int(0);++index) {
        rows=std::max(rows,value_storage(index).extent_int(0));
        rows=std::max(rows,adjoint_storage(index).extent_int(0));
    }
    return rows;
}

template<typename Precision>
std::size_t AffineMLPKokkosT<Precision>::workspace_bytes() const
{
    std::size_t bytes=0;
    for(int index=0;index<value_storage.extent_int(0);++index)
        bytes+=sizeof(Precision)*(value_storage(index).size()+adjoint_storage(index).size());
    return bytes;
}

template<typename Precision>
void AffineMLPKokkosT<Precision>::clear_workspace()
{
    for(int index=0;index<value_storage.extent_int(0);++index) {
        values(index)={};
        adjoints(index)={};
        value_storage(index)={};
        adjoint_storage(index)={};
    }
    tape_batch_size=-1;
    tape_input_size=-1;
}

template<typename Precision>
void AffineMLPKokkosT<Precision>::prepare(int batch_size,int active_input_size)
{
    if (value_storage(0).extent(0)<batch_size
        ||value_storage(0).extent(1)!=active_input_size)
        Kokkos::realloc(value_storage(0),batch_size,active_input_size);
    if (adjoint_storage(0).extent(0)<batch_size
        ||adjoint_storage(0).extent(1)!=active_input_size)
        Kokkos::realloc(adjoint_storage(0),batch_size,active_input_size);
    values(0)=Kokkos::subview(
        value_storage(0),std::make_pair(0,batch_size),Kokkos::ALL);
    adjoints(0)=Kokkos::subview(
        adjoint_storage(0),std::make_pair(0,batch_size),Kokkos::ALL);
    for (int layer=0; layer<types.size(); ++layer) {
        if (value_storage(layer+1).extent(0)<batch_size
            ||value_storage(layer+1).extent(1)!=output_sizes(layer))
            Kokkos::realloc(value_storage(layer+1),batch_size,output_sizes(layer));
        if (adjoint_storage(layer+1).extent(0)<batch_size
            ||adjoint_storage(layer+1).extent(1)!=output_sizes(layer))
            Kokkos::realloc(adjoint_storage(layer+1),batch_size,output_sizes(layer));
        values(layer+1)=Kokkos::subview(
            value_storage(layer+1),std::make_pair(0,batch_size),Kokkos::ALL);
        adjoints(layer+1)=Kokkos::subview(
            adjoint_storage(layer+1),std::make_pair(0,batch_size),Kokkos::ALL);
    }
}

template<typename Precision>
void AffineMLPKokkosT<Precision>::prepare_conditioned_weight(int active_input_size)
{
    if(conditioned_input_size==active_input_size) return;
    if(!supports_conditioned_input(active_input_size))
        throw std::invalid_argument("Kokkos affine MLP conditioned weight dimensions are inconsistent.");
    Kokkos::realloc(
        Kokkos::WithoutInitializing,conditioned_weight,output_sizes(0),active_input_size);
    auto full_weight=weights(0);
    auto compact_weight=conditioned_weight;
    Kokkos::parallel_for(
        "prepare conditioned affine weight",
        Kokkos::MDRangePolicy<Kokkos::Rank<2>>(
            {0,0},{output_sizes(0),active_input_size}),
        KOKKOS_LAMBDA(int row,int column) {
            compact_weight(row,column)=full_weight(row,column);
        });
    conditioned_input_size=active_input_size;
}

template<typename Precision>
void AffineMLPKokkosT<Precision>::forward(Kokkos::View<const Precision**,Kokkos::LayoutRight> input)
{
    if(input.extent(1)!=static_cast<std::size_t>(input_size()))
        throw std::invalid_argument("Kokkos affine MLP input dimensions are inconsistent.");
    forward_impl(input,{});
}

template<typename Precision>
void AffineMLPKokkosT<Precision>::forward_impl(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> row_contributions)
{
    const bool conditioned=row_contributions.data()!=nullptr;
    if(conditioned&&(!supports_conditioned_input(input.extent(1))
        ||row_contributions.extent(0)!=input.extent(0)
        ||row_contributions.extent(1)!=static_cast<std::size_t>(output_sizes(0))))
        throw std::invalid_argument("Kokkos affine MLP conditioned dimensions are inconsistent.");
    prepare(input.extent(0),input.extent(1));
    if(conditioned) prepare_conditioned_weight(input.extent(1));
    affine_deep_copy(values(0),input);
    for (int layer=0; layer<types.size();) {
        if(types(layer)==LayerNorm&&layer+1<types.size()&&types(layer+1)==SiLU) {
            auto source=values(layer);
            auto target=values(layer+2);
            auto gamma=weights(layer);
            auto beta=biases(layer);
            const Precision epsilon=eps(layer);
            Kokkos::parallel_for(
                "affine fused layer norm silu",target.extent(0),
                KOKKOS_LAMBDA(int sample) {
                    Precision mean=Precision(0);
                    for(int column=0;column<source.extent(1);++column)
                        mean+=source(sample,column);
                    mean/=source.extent(1);
                    Precision variance=Precision(0);
                    for(int column=0;column<source.extent(1);++column) {
                        const Precision delta=source(sample,column)-mean;
                        variance+=delta*delta;
                    }
                    variance/=source.extent(1);
                    const Precision inverse=Precision(1)/Kokkos::sqrt(variance+epsilon);
                    for(int column=0;column<source.extent(1);++column) {
                        const Precision value=(source(sample,column)-mean)*inverse
                            *gamma(0,column)+beta(column);
                        target(sample,column)=value/(Precision(1)+Kokkos::exp(-value));
                    }
                });
            layer+=2;
            continue;
        }
        auto source=values(layer); auto target=values(layer+1);
        if (types(layer)==Linear) {
            auto weight=weights(layer); auto bias=biases(layer);
            const bool use_gemm=source.extent(0)>4;
            if(use_gemm) {
                if(conditioned&&layer==0)
                    KokkosBlas::gemm(
                        "N","T",Precision(1),source,conditioned_weight,Precision(0),target);
                else
                    KokkosBlas::gemm("N","T",Precision(1),source,weight,Precision(0),target);
                Kokkos::parallel_for(
                    "affine linear epilogue",target.size(),KOKKOS_LAMBDA(int flat) {
                        const int sample=flat/target.extent(1);
                        const int row=flat%target.extent(1);
                        target(sample,row)+=bias(row)
                            +(conditioned&&layer==0?row_contributions(sample,row):Precision(0));
                    });
            } else {
                Kokkos::parallel_for("affine linear",Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{target.extent(0),target.extent(1)}),
                    KOKKOS_LAMBDA(int sample,int row) {
                        Precision value=bias(row);
                        for (int column=0; column<source.extent(1); ++column) value+=weight(row,column)*source(sample,column);
                        if(conditioned&&layer==0) value+=row_contributions(sample,row);
                        target(sample,row)=value;
                    });
            }
        } else if (types(layer)==LayerNorm) {
            auto gamma=weights(layer); auto beta=biases(layer); const Precision epsilon=eps(layer);
            Kokkos::parallel_for("affine layer norm",target.extent(0),KOKKOS_LAMBDA(int sample) {
                Precision mean=Precision(0); for (int column=0; column<source.extent(1); ++column) mean+=source(sample,column); mean/=source.extent(1);
                Precision variance=Precision(0); for (int column=0; column<source.extent(1); ++column) { const Precision delta=source(sample,column)-mean; variance+=delta*delta; } variance/=source.extent(1);
                const Precision inverse=Precision(1)/Kokkos::sqrt(variance+epsilon);
                for (int column=0; column<source.extent(1); ++column) target(sample,column)=(source(sample,column)-mean)*inverse*gamma(0,column)+beta(column);
            });
        } else Kokkos::parallel_for("affine silu",target.size(),KOKKOS_LAMBDA(int flat) {
            const int sample=flat/target.extent(1), column=flat%target.extent(1); const Precision value=source(sample,column); target(sample,column)=value/(Precision(1)+Kokkos::exp(-value));
        });
        ++layer;
    }
    tape_batch_size=input.extent(0);
    tape_input_size=input.extent(1);
}

template<typename Precision>
void AffineMLPKokkosT<Precision>::evaluate(Kokkos::View<const Precision**,Kokkos::LayoutRight> input,Kokkos::View<Precision**,Kokkos::LayoutRight> output)
{
    forward(input); affine_deep_copy(output,values(types.size()));
}

template<typename Precision>
void AffineMLPKokkosT<Precision>::evaluate_conditioned(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> row_contributions,
    Kokkos::View<Precision**,Kokkos::LayoutRight> output)
{
    forward_impl(input,row_contributions);
    affine_deep_copy(output,values(types.size()));
}

template<typename Precision>
void AffineMLPKokkosT<Precision>::reverse(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> input,
    Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint)
{
    forward(input);
    reverse_from_tape(output_adjoint,input_adjoint);
}

template<typename Precision>
void AffineMLPKokkosT<Precision>::reverse_from_tape(
    Kokkos::View<const Precision**,Kokkos::LayoutRight> output_adjoint,
    Kokkos::View<Precision**,Kokkos::LayoutRight> input_adjoint)
{
    if(tape_batch_size<0
        ||output_adjoint.extent(0)!=static_cast<std::size_t>(tape_batch_size)
        ||output_adjoint.extent(1)!=static_cast<std::size_t>(output_size())
        ||input_adjoint.extent(0)!=static_cast<std::size_t>(tape_batch_size)
        ||input_adjoint.extent(1)!=static_cast<std::size_t>(tape_input_size))
        throw std::invalid_argument("Kokkos affine MLP reverse tape dimensions are inconsistent.");
    affine_deep_copy(adjoints(types.size()),output_adjoint);
    for (int layer=types.size()-1; layer>=0;) {
        if(types(layer)==SiLU&&layer>0&&types(layer-1)==LayerNorm) {
            auto source=values(layer-1);
            auto source_adj=adjoints(layer-1);
            auto target_adj=adjoints(layer+1);
            auto gamma=weights(layer-1);
            auto beta=biases(layer-1);
            const Precision epsilon=eps(layer-1);
            Kokkos::parallel_for(
                "affine fused reverse layer norm silu",source.extent(0),
                KOKKOS_LAMBDA(int sample) {
                    const int width=source.extent(1);
                    Precision mean=Precision(0);
                    for(int column=0;column<width;++column)
                        mean+=source(sample,column);
                    mean/=width;
                    Precision variance=Precision(0);
                    for(int column=0;column<width;++column) {
                        const Precision delta=source(sample,column)-mean;
                        variance+=delta*delta;
                    }
                    variance/=width;
                    const Precision inverse=Precision(1)/Kokkos::sqrt(variance+epsilon);
                    Precision sum=Precision(0);
                    Precision sum_normalized=Precision(0);
                    for(int column=0;column<width;++column) {
                        const Precision normalized=(source(sample,column)-mean)*inverse;
                        const Precision value=normalized*gamma(0,column)+beta(column);
                        const Precision probability=Precision(1)/(Precision(1)+Kokkos::exp(-value));
                        const Precision scaled=target_adj(sample,column)
                            *(probability+value*probability*(Precision(1)-probability))
                            *gamma(0,column);
                        sum+=scaled;
                        sum_normalized+=scaled*normalized;
                    }
                    for(int column=0;column<width;++column) {
                        const Precision normalized=(source(sample,column)-mean)*inverse;
                        const Precision value=normalized*gamma(0,column)+beta(column);
                        const Precision probability=Precision(1)/(Precision(1)+Kokkos::exp(-value));
                        const Precision scaled=target_adj(sample,column)
                            *(probability+value*probability*(Precision(1)-probability))
                            *gamma(0,column);
                        source_adj(sample,column)=inverse
                            *(width*scaled-sum-normalized*sum_normalized)/width;
                    }
                });
            layer-=2;
            continue;
        }
        auto source=values(layer); auto source_adj=adjoints(layer); auto target_adj=adjoints(layer+1);
        if (types(layer)==Linear) {
            auto weight=weights(layer);
            if(source_adj.extent(0)>4) {
                if(layer==0&&tape_input_size<input_size())
                    KokkosBlas::gemm(
                        "N","N",Precision(1),target_adj,conditioned_weight,Precision(0),source_adj);
                else
                    KokkosBlas::gemm("N","N",Precision(1),target_adj,weight,Precision(0),source_adj);
            } else {
                Kokkos::parallel_for("affine reverse linear",source_adj.size(),KOKKOS_LAMBDA(int flat) {
                    const int sample=flat/source_adj.extent(1), column=flat%source_adj.extent(1); Precision value=Precision(0);
                    for (int row=0; row<target_adj.extent(1); ++row) value+=weight(row,column)*target_adj(sample,row); source_adj(sample,column)=value;
                });
            }
        } else if (types(layer)==LayerNorm) {
            affine_deep_copy(source_adj,Precision(0));
            auto gamma=weights(layer); const Precision epsilon=eps(layer);
            Kokkos::parallel_for("affine reverse layer norm",source.extent(0),KOKKOS_LAMBDA(int sample) {
                const int width=source.extent(1); Precision mean=Precision(0); for (int i=0;i<width;++i) mean+=source(sample,i); mean/=width;
                Precision variance=Precision(0); for (int i=0;i<width;++i) { const Precision delta=source(sample,i)-mean; variance+=delta*delta; } variance/=width;
                const Precision inverse=Precision(1)/Kokkos::sqrt(variance+epsilon); Precision sum=Precision(0),sum_normalized=Precision(0);
                for (int i=0;i<width;++i) { const Precision scaled=target_adj(sample,i)*gamma(0,i); sum+=scaled; sum_normalized+=scaled*(source(sample,i)-mean)*inverse; }
                for (int i=0;i<width;++i) { const Precision scaled=target_adj(sample,i)*gamma(0,i); const Precision normalized=(source(sample,i)-mean)*inverse; source_adj(sample,i)=inverse*(width*scaled-sum-normalized*sum_normalized)/width; }
            });
        } else {
            affine_deep_copy(source_adj,Precision(0));
            Kokkos::parallel_for("affine reverse silu",source_adj.size(),KOKKOS_LAMBDA(int flat) {
                const int sample=flat/source_adj.extent(1), column=flat%source_adj.extent(1); const Precision value=source(sample,column); const Precision probability=Precision(1)/(Precision(1)+Kokkos::exp(-value)); source_adj(sample,column)=target_adj(sample,column)*(probability+value*probability*(Precision(1)-probability));
            });
        }
        --layer;
    }
    affine_deep_copy(input_adjoint,adjoints(0));
}

template class AffineMLPKokkosT<float>;
template class AffineMLPKokkosT<double>;
