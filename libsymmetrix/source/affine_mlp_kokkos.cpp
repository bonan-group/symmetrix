#include "affine_mlp_kokkos.hpp"
#include "affine_mlp.hpp"

#include <stdexcept>
#include <vector>

#include "tools_kokkos.hpp"

namespace {
std::vector<double> tensor_values(const nlohmann::json& value)
{
    return value.at("values").get<std::vector<double>>();
}
}

AffineMLPKokkos::AffineMLPKokkos(const nlohmann::json& definition)
{
    AffineMLP validated(definition);
    const int count = definition.at("layers").size();
    types = Kokkos::View<int*,Kokkos::SharedSpace>("affine mlp types",count);
    input_sizes = Kokkos::View<int*,Kokkos::SharedSpace>("affine mlp inputs",count);
    output_sizes = Kokkos::View<int*,Kokkos::SharedSpace>("affine mlp outputs",count);
    eps = Kokkos::View<double*,Kokkos::SharedSpace>("affine mlp eps",count);
    weights = decltype(weights)(Kokkos::view_alloc("affine mlp weights",Kokkos::SequentialHostInit),count);
    biases = decltype(biases)(Kokkos::view_alloc("affine mlp biases",Kokkos::SequentialHostInit),count);
    values = decltype(values)(Kokkos::view_alloc("affine mlp values",Kokkos::SequentialHostInit),count+1);
    adjoints = decltype(adjoints)(Kokkos::view_alloc("affine mlp adjoints",Kokkos::SequentialHostInit),count+1);
    int previous = -1;
    for (int index=0; index<count; ++index) {
        const auto& layer = definition.at("layers").at(index);
        const auto type = layer.at("type").get<std::string>();
        if (type == "linear") {
            types(index) = Linear;
            const auto shape = layer.at("weight").at("shape").get<std::vector<int>>();
            if (shape.size()!=2) throw std::invalid_argument("Kokkos affine linear weight must be rank two.");
            input_sizes(index)=shape[1]; output_sizes(index)=shape[0];
            weights(index)=toKokkosView("affine mlp weight",tensor_values(layer.at("weight")),shape[0],shape[1]);
            biases(index)=toKokkosView("affine mlp bias",tensor_values(layer.at("bias")));
        } else if (type == "layer_norm") {
            types(index)=LayerNorm;
            input_sizes(index)=layer.at("normalized_shape").at(0).get<int>();
            output_sizes(index)=input_sizes(index); eps(index)=layer.at("eps").get<double>();
            const auto gamma=tensor_values(layer.at("weight"));
            weights(index)=toKokkosView("affine layer norm gamma",gamma,1,gamma.size());
            biases(index)=toKokkosView("affine layer norm beta",tensor_values(layer.at("bias")));
        } else if (type == "silu") {
            if (previous<0) throw std::invalid_argument("Kokkos affine SiLU cannot be the first layer.");
            types(index)=SiLU; input_sizes(index)=previous; output_sizes(index)=previous;
        } else throw std::invalid_argument("Unsupported Kokkos affine MLP layer: "+type);
        if (previous>=0 && input_sizes(index)!=previous)
            throw std::invalid_argument("Kokkos affine MLP layer dimensions are inconsistent.");
        previous=output_sizes(index);
    }
}

int AffineMLPKokkos::input_size() const { return input_sizes(0); }
int AffineMLPKokkos::output_size() const { return output_sizes(output_sizes.size()-1); }

void AffineMLPKokkos::prepare(int batch_size)
{
    if (values(0).extent(0)!=batch_size || values(0).extent(1)!=input_size())
        Kokkos::realloc(values(0),batch_size,input_size());
    if (adjoints(0).extent(0)!=batch_size || adjoints(0).extent(1)!=input_size())
        Kokkos::realloc(adjoints(0),batch_size,input_size());
    for (int layer=0; layer<types.size(); ++layer) {
        if (values(layer+1).extent(0)!=batch_size || values(layer+1).extent(1)!=output_sizes(layer))
            Kokkos::realloc(values(layer+1),batch_size,output_sizes(layer));
        if (adjoints(layer+1).extent(0)!=batch_size || adjoints(layer+1).extent(1)!=output_sizes(layer))
            Kokkos::realloc(adjoints(layer+1),batch_size,output_sizes(layer));
    }
}

void AffineMLPKokkos::forward(Kokkos::View<const double**,Kokkos::LayoutRight> input)
{
    prepare(input.extent(0));
    Kokkos::deep_copy(values(0),input);
    for (int layer=0; layer<types.size(); ++layer) {
        auto source=values(layer); auto target=values(layer+1);
        if (types(layer)==Linear) {
            auto weight=weights(layer); auto bias=biases(layer);
            Kokkos::parallel_for("affine linear",Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0,0},{target.extent(0),target.extent(1)}),
                KOKKOS_LAMBDA(int sample,int row) {
                    double value=bias(row);
                    for (int column=0; column<source.extent(1); ++column) value+=weight(row,column)*source(sample,column);
                    target(sample,row)=value;
                });
        } else if (types(layer)==LayerNorm) {
            auto gamma=weights(layer); auto beta=biases(layer); const double epsilon=eps(layer);
            Kokkos::parallel_for("affine layer norm",target.extent(0),KOKKOS_LAMBDA(int sample) {
                double mean=0.0; for (int column=0; column<source.extent(1); ++column) mean+=source(sample,column); mean/=source.extent(1);
                double variance=0.0; for (int column=0; column<source.extent(1); ++column) { const double delta=source(sample,column)-mean; variance+=delta*delta; } variance/=source.extent(1);
                const double inverse=1.0/Kokkos::sqrt(variance+epsilon);
                for (int column=0; column<source.extent(1); ++column) target(sample,column)=(source(sample,column)-mean)*inverse*gamma(0,column)+beta(column);
            });
        } else Kokkos::parallel_for("affine silu",target.size(),KOKKOS_LAMBDA(int flat) {
            const int sample=flat/target.extent(1), column=flat%target.extent(1); const double value=source(sample,column); target(sample,column)=value/(1.0+Kokkos::exp(-value));
        });
    }
}

void AffineMLPKokkos::evaluate(Kokkos::View<const double**,Kokkos::LayoutRight> input,Kokkos::View<double**,Kokkos::LayoutRight> output)
{
    forward(input); Kokkos::deep_copy(output,values(types.size()));
}

void AffineMLPKokkos::reverse(
    Kokkos::View<const double**,Kokkos::LayoutRight> input,
    Kokkos::View<const double**,Kokkos::LayoutRight> output_adjoint,
    Kokkos::View<double**,Kokkos::LayoutRight> input_adjoint)
{
    forward(input); Kokkos::deep_copy(adjoints(types.size()),output_adjoint);
    for (int layer=types.size()-1; layer>=0; --layer) {
        auto source=values(layer); auto source_adj=adjoints(layer); auto target_adj=adjoints(layer+1); Kokkos::deep_copy(source_adj,0.0);
        if (types(layer)==Linear) {
            auto weight=weights(layer);
            Kokkos::parallel_for("affine reverse linear",source_adj.size(),KOKKOS_LAMBDA(int flat) {
                const int sample=flat/source_adj.extent(1), column=flat%source_adj.extent(1); double value=0.0;
                for (int row=0; row<target_adj.extent(1); ++row) value+=weight(row,column)*target_adj(sample,row); source_adj(sample,column)=value;
            });
        } else if (types(layer)==LayerNorm) {
            auto gamma=weights(layer); const double epsilon=eps(layer);
            Kokkos::parallel_for("affine reverse layer norm",source.extent(0),KOKKOS_LAMBDA(int sample) {
                const int width=source.extent(1); double mean=0.0; for (int i=0;i<width;++i) mean+=source(sample,i); mean/=width;
                double variance=0.0; for (int i=0;i<width;++i) { const double delta=source(sample,i)-mean; variance+=delta*delta; } variance/=width;
                const double inverse=1.0/Kokkos::sqrt(variance+epsilon); double sum=0.0,sum_normalized=0.0;
                for (int i=0;i<width;++i) { const double scaled=target_adj(sample,i)*gamma(0,i); sum+=scaled; sum_normalized+=scaled*(source(sample,i)-mean)*inverse; }
                for (int i=0;i<width;++i) { const double scaled=target_adj(sample,i)*gamma(0,i); const double normalized=(source(sample,i)-mean)*inverse; source_adj(sample,i)=inverse*(width*scaled-sum-normalized*sum_normalized)/width; }
            });
        } else Kokkos::parallel_for("affine reverse silu",source_adj.size(),KOKKOS_LAMBDA(int flat) {
            const int sample=flat/source_adj.extent(1), column=flat%source_adj.extent(1); const double value=source(sample,column); const double probability=1.0/(1.0+Kokkos::exp(-value)); source_adj(sample,column)=target_adj(sample,column)*(probability+value*probability*(1.0-probability));
        });
    }
    Kokkos::deep_copy(input_adjoint,adjoints(0));
}
