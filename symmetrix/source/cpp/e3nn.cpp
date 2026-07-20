#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "affine_mlp.hpp"
#include "e3nn.hpp"
#include "e3nn_product.hpp"

#ifdef SYMMETRIX_KOKKOS
#include "e3nn_kokkos.hpp"
#include "e3nn_product_kokkos.hpp"
#include "tools_kokkos.hpp"
#endif

namespace py = pybind11;

void bind_e3nn(py::module_& module)
{
    py::class_<AffineMLP>(module, "AffineMLP")
        .def(py::init([](const std::string& definition) {
            return AffineMLP(nlohmann::json::parse(definition));
        }))
        .def_property_readonly("input_size", &AffineMLP::input_size)
        .def_property_readonly("output_size", &AffineMLP::output_size)
        .def("supports_conditioned_input", &AffineMLP::supports_conditioned_input)
        .def("first_layer_contribution", &AffineMLP::first_layer_contribution)
        .def("evaluate", &AffineMLP::evaluate)
        .def("evaluate_conditioned", &AffineMLP::evaluate_conditioned)
        .def("reverse_conditioned", &AffineMLP::evaluate_gradient_conditioned)
        .def("reverse", &AffineMLP::evaluate_gradient);

    py::class_<E3Linear>(module, "E3Linear")
        .def(py::init([](const std::string& definition) {
            return E3Linear(nlohmann::json::parse(definition));
        }))
        .def_property_readonly("input_dimension", &E3Linear::input_dimension)
        .def_property_readonly("output_dimension", &E3Linear::output_dimension)
        .def("evaluate", &E3Linear::evaluate)
        .def("reverse", [](const E3Linear& self, const std::vector<double>& output_adjoint) {
            std::vector<double> input_adjoint;
            self.reverse(output_adjoint, input_adjoint);
            return input_adjoint;
        });

    py::class_<E3TensorProduct>(module, "E3TensorProduct")
        .def(py::init([](const std::string& definition) {
            return E3TensorProduct(nlohmann::json::parse(definition));
        }))
        .def_property_readonly("input_1_dimension", &E3TensorProduct::input_1_dimension)
        .def_property_readonly("input_2_dimension", &E3TensorProduct::input_2_dimension)
        .def_property_readonly("output_dimension", &E3TensorProduct::output_dimension)
        .def("evaluate", &E3TensorProduct::evaluate,
             py::arg("input_1"), py::arg("input_2"), py::arg("weights") = std::vector<double>{})
        .def("reverse", [](
            const E3TensorProduct& self,
            const std::vector<double>& input_1,
            const std::vector<double>& input_2,
            const std::vector<double>& weights,
            const std::vector<double>& output_adjoint) {
            std::vector<double> input_1_adjoint;
            std::vector<double> input_2_adjoint;
            std::vector<double> weights_adjoint;
            self.reverse(input_1, input_2, weights, output_adjoint,
                         input_1_adjoint, input_2_adjoint, weights_adjoint);
            return py::make_tuple(input_1_adjoint, input_2_adjoint, weights_adjoint);
        });

    py::class_<E3ProductBasis>(module, "E3ProductBasis")
        .def(py::init([](const std::string& definition) {
            return E3ProductBasis(nlohmann::json::parse(definition));
        }))
        .def_property_readonly("input_dimension", &E3ProductBasis::input_dimension)
        .def_property_readonly("output_dimension", &E3ProductBasis::output_dimension)
        .def_property_readonly("uses_compiled_plan", &E3ProductBasis::uses_compiled_plan)
        .def_property_readonly("compiled_term_count", &E3ProductBasis::compiled_term_count)
        .def("evaluate", &E3ProductBasis::evaluate)
        .def("reverse", [](
            const E3ProductBasis& self,
            const std::vector<double>& node_features,
            int element,
            const std::vector<double>& output_adjoint) {
            std::vector<double> node_features_adjoint;
            std::vector<double> skip_connection_adjoint;
            self.reverse(node_features, element, output_adjoint,
                         node_features_adjoint, skip_connection_adjoint);
            return py::make_tuple(node_features_adjoint, skip_connection_adjoint);
        });

#ifdef SYMMETRIX_KOKKOS
    py::class_<E3TensorProductKokkos>(module, "E3TensorProductKokkos")
        .def(py::init([](const std::string& definition) {
            return E3TensorProductKokkos(nlohmann::json::parse(definition));
        }))
        .def("evaluate", [](
            const E3TensorProductKokkos& self,
            const std::vector<double>& input_1,
            const std::vector<double>& input_2,
            const std::vector<double>& weights) {
            Kokkos::View<double**,Kokkos::LayoutRight> input_1_view;
            Kokkos::View<double**,Kokkos::LayoutRight> input_2_view;
            Kokkos::View<double**,Kokkos::LayoutRight> weights_view;
            set_kokkos_view(input_1_view, input_1, 1, self.input_1_dimension());
            set_kokkos_view(input_2_view, input_2, 1, self.input_2_dimension());
            set_kokkos_view(weights_view, weights, 1, self.weight_size());
            Kokkos::View<double**,Kokkos::LayoutRight> output(
                "bound e3 tensor output", 1, self.output_dimension());
            self.evaluate(input_1_view, input_2_view, weights_view, output);
            return view2vector(output);
        })
        .def("reverse", [](
            const E3TensorProductKokkos& self,
            const std::vector<double>& input_1,
            const std::vector<double>& input_2,
            const std::vector<double>& weights,
            const std::vector<double>& output_adjoint) {
            Kokkos::View<double**,Kokkos::LayoutRight> input_1_view;
            Kokkos::View<double**,Kokkos::LayoutRight> input_2_view;
            Kokkos::View<double**,Kokkos::LayoutRight> weights_view;
            Kokkos::View<double**,Kokkos::LayoutRight> output_adjoint_view;
            set_kokkos_view(input_1_view, input_1, 1, self.input_1_dimension());
            set_kokkos_view(input_2_view, input_2, 1, self.input_2_dimension());
            set_kokkos_view(weights_view, weights, 1, self.weight_size());
            set_kokkos_view(
                output_adjoint_view, output_adjoint, 1, self.output_dimension());
            Kokkos::View<double**,Kokkos::LayoutRight> input_1_adjoint(
                "bound e3 tensor input 1 adjoint", 1, self.input_1_dimension());
            Kokkos::View<double**,Kokkos::LayoutRight> input_2_adjoint(
                "bound e3 tensor input 2 adjoint", 1, self.input_2_dimension());
            Kokkos::View<double**,Kokkos::LayoutRight> weights_adjoint(
                "bound e3 tensor weights adjoint", 1, self.weight_size());
            self.reverse(
                input_1_view, input_2_view, weights_view, output_adjoint_view,
                input_1_adjoint, input_2_adjoint, weights_adjoint);
            return py::make_tuple(
                view2vector(input_1_adjoint),
                view2vector(input_2_adjoint),
                view2vector(weights_adjoint));
        });

    py::class_<E3ProductBasisKokkos>(module, "E3ProductBasisKokkos")
        .def(py::init([](const std::string& definition) {
            return E3ProductBasisKokkos(nlohmann::json::parse(definition));
        }));
#endif
}
