#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mace_nonlinear_kokkos.hpp"
#include "utilities_kokkos.hpp"

namespace py=pybind11;
using IntArray=py::array_t<int,py::array::c_style|py::array::forcecast>;
using DoubleArray=py::array_t<double,py::array::c_style|py::array::forcecast>;

void bind_mace_nonlinear_kokkos(py::module_& module)
{
    py::class_<MaceNonlinearKokkos>(module,"MACENonlinearKokkos")
        .def(py::init<std::string>())
        .def_readonly("r_cut",&MaceNonlinearKokkos::r_cut)
        .def_readonly("has_field_coupling",&MaceNonlinearKokkos::has_field_coupling)
        .def_property_readonly("atomic_numbers",[](MaceNonlinearKokkos& self){return self.atomic_numbers_host;})
        .def_property_readonly("atomic_energies",[](MaceNonlinearKokkos& self){return view2vector(self.atomic_energies);})
        .def_property_readonly("node_energies",[](MaceNonlinearKokkos& self){return view2vector(self.node_energies);})
        .def_property_readonly("node_forces",[](MaceNonlinearKokkos& self){return view2vector(self.node_forces);})
        .def("compute_node_energies_forces",[](MaceNonlinearKokkos& self,int num_nodes,IntArray node_types,IntArray num_neigh,IntArray neigh_indices,IntArray neigh_types,DoubleArray xyz,DoubleArray distances){self.compute_node_energies_forces(num_nodes,create_kokkos_view("nonlinear node types",node_types),create_kokkos_view("nonlinear num neigh",num_neigh),create_kokkos_view("nonlinear neigh indices",neigh_indices),create_kokkos_view("nonlinear neigh types",neigh_types),create_kokkos_view("nonlinear xyz",xyz),create_kokkos_view("nonlinear distances",distances));});
}
