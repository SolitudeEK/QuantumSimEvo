#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include "../QuantumSimEvo/Circuit.h" // Path to your actual header

namespace py = pybind11;

PYBIND11_MODULE(QuantumSimEvo, m) {
    m.doc() = "Python wrapper for QuantumSimEvo";

    py::class_<Circuit>(m, "Circuit")
        .def(py::init<size_t>(), py::arg("num_qubits"))
        .def("hadamard", &Circuit::hadamard, py::arg("q"))
        .def("pauli_x", &Circuit::pauliX, py::arg("q"))
        .def("pauli_y", &Circuit::pauliY, py::arg("q"))
        .def("pauli_z", &Circuit::pauliZ, py::arg("q"))
		.def("phase", &Circuit::phase, py::arg("theta"), py::arg("q"))
        .def("rotate_x", &Circuit::rotateX, py::arg("theta"), py::arg("q"))
        .def("rotate_y", &Circuit::rotateX, py::arg("theta"), py::arg("q"))
        .def("rotate_z", &Circuit::rotateZ, py::arg("theta"), py::arg("q"))
		.def("cnot", &Circuit::cnot, py::arg("control"), py::arg("target"))
        .def("execute", &Circuit::execute, py::arg("print_steps") = false)
		//.def("measure", &Circuit::measure)
		.def("print_state", &Circuit::printState)
        .def("reset", &Circuit::reset);
}