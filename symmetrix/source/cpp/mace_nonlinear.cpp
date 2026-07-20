#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mace_nonlinear.hpp"

namespace py = pybind11;
using IntArray = py::array_t<int, py::array::c_style | py::array::forcecast>;
using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;

void bind_mace_nonlinear(py::module_& module)
{
    py::class_<MaceNonlinear>(module, "MACENonlinear")
        .def(py::init<std::string>())
        .def_readonly("atomic_numbers", &MaceNonlinear::atomic_numbers)
        .def_readonly("atomic_energies", &MaceNonlinear::atomic_energies)
        .def_readonly("r_cut", &MaceNonlinear::r_cut)
        .def_readonly("has_field_coupling", &MaceNonlinear::has_field_coupling)
        .def_readonly("node_energies", &MaceNonlinear::node_energies)
        .def_readonly("node_forces", &MaceNonlinear::node_forces)
        .def("compute_node_energies_forces", [](
            MaceNonlinear& self, int num_nodes, IntArray node_types,
            IntArray num_neigh, IntArray neigh_indices, IntArray neigh_types,
            DoubleArray xyz, DoubleArray distances) {
            self.compute_node_energies_forces(
                num_nodes,
                {node_types.data(), static_cast<std::size_t>(node_types.size())},
                {num_neigh.data(), static_cast<std::size_t>(num_neigh.size())},
                {neigh_indices.data(), static_cast<std::size_t>(neigh_indices.size())},
                {neigh_types.data(), static_cast<std::size_t>(neigh_types.size())},
                {xyz.data(), static_cast<std::size_t>(xyz.size())},
                {distances.data(), static_cast<std::size_t>(distances.size())});
        });
}
