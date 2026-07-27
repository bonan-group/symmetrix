#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mace_nonlinear_kokkos.hpp"
#include "utilities_kokkos.hpp"

namespace py=pybind11;
using IntArray=py::array_t<int,py::array::c_style|py::array::forcecast>;
using DoubleArray=py::array_t<double,py::array::c_style|py::array::forcecast>;

template<typename Evaluator>
void bind_mace_nonlinear_kokkos_evaluator(
    py::module_& module,const char* name)
{
    py::class_<Evaluator>(module,name)
        .def(py::init<std::string>())
        .def_readonly("r_cut",&Evaluator::r_cut)
        .def_readonly("has_field_coupling",&Evaluator::has_field_coupling)
        .def_property_readonly("scalar_size_bytes",[](const Evaluator&) {
            return sizeof(typename Evaluator::precision_type);
        })
        .def_property_readonly("uses_mh1_fast_path",&Evaluator::uses_mh1_fast_path)
        .def_property_readonly("supports_streamed_edges",&Evaluator::supports_streamed_edges)
        .def_property_readonly("streamed_edges_mode",&Evaluator::streamed_edges_mode)
        .def("set_streamed_edges",&Evaluator::set_streamed_edges)
        .def_property_readonly("edge_workspace_rows",&Evaluator::edge_workspace_rows)
        .def_property_readonly("edge_workspace_bytes",&Evaluator::edge_workspace_bytes)
        .def("fence",&Evaluator::fence)
        .def("set_e3_linear_backend",&Evaluator::set_e3_linear_backend)
        .def_property_readonly("e3_linear_backend",&Evaluator::e3_linear_backend)
        .def_property_readonly("tensor_product_backend",&Evaluator::tensor_product_backend)
        .def_property_readonly("linear_workspace_bytes",&Evaluator::linear_workspace_bytes)
        .def_property_readonly("tensor_workspace_bytes",&Evaluator::tensor_workspace_bytes)
        .def_property_readonly("precision_workspace_bytes",&Evaluator::precision_workspace_bytes)
        .def_property_readonly("atomic_numbers",[](Evaluator& self){return self.atomic_numbers_host;})
        .def_property_readonly("atomic_energies",[](Evaluator& self){return view2vector(self.atomic_energies);})
        .def_property_readonly("node_energies",[](Evaluator& self){return view2vector(self.node_energies);})
        .def_property_readonly("node_forces",[](Evaluator& self){return view2vector(self.node_forces);})
        .def("compute_node_energies_forces",[](Evaluator& self,int num_nodes,IntArray node_types,IntArray num_neigh,IntArray neigh_indices,IntArray neigh_types,DoubleArray xyz,DoubleArray distances){self.compute_node_energies_forces(num_nodes,create_kokkos_view("nonlinear node types",node_types),create_kokkos_view("nonlinear num neigh",num_neigh),create_kokkos_view("nonlinear neigh indices",neigh_indices),create_kokkos_view("nonlinear neigh types",neigh_types),create_kokkos_view("nonlinear xyz",xyz),create_kokkos_view("nonlinear distances",distances));});
}

void bind_mace_nonlinear_kokkos(py::module_& module)
{
    bind_mace_nonlinear_kokkos_evaluator<MaceNonlinearKokkos>(
        module,"MACENonlinearKokkos");
    bind_mace_nonlinear_kokkos_evaluator<MaceNonlinearFloatKokkos>(
        module,"MACENonlinearKokkosFloat");
}
