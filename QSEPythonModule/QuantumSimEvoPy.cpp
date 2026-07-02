#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/stl.h>
#include "../QuantumSimEvo/Circuit.h" // Path to your actual header

namespace py = pybind11;

PYBIND11_MODULE(QuantumSimEvo, m) {
    m.doc() = "Python wrapper for QuantumSimEvo";

    py::class_<Circuit>(m, "Circuit")
        .def(py::init<size_t, bool>(),
            py::arg("num_qubits"), py::arg("simulate_pauli_noise") = false)

        // Single-qubit gates
        .def("hadamard", &Circuit::hadamard, py::arg("q"))
        .def("pauli_x", &Circuit::pauliX, py::arg("q"))
        .def("pauli_y", &Circuit::pauliY, py::arg("q"))
        .def("pauli_z", &Circuit::pauliZ, py::arg("q"))
        .def("phase", &Circuit::phase, py::arg("theta"), py::arg("q"))
        .def("rotate_x", &Circuit::rotateX, py::arg("theta"), py::arg("q"))
        .def("rotate_y", &Circuit::rotateY, py::arg("theta"), py::arg("q"))
        .def("rotate_z", &Circuit::rotateZ, py::arg("theta"), py::arg("q"))

        // Multi-qubit gates
        .def("cnot", &Circuit::cnot, py::arg("control"), py::arg("target"))
        .def("cphase", &Circuit::cphase,
            py::arg("theta"), py::arg("control"), py::arg("target"))
        .def("mcphase", &Circuit::mcphase,
            py::arg("theta"), py::arg("controls"), py::arg("target"))
        .def("swap", &Circuit::swap, py::arg("q0"), py::arg("q1"))

        // Execution & measurement
        .def("execute", &Circuit::execute, py::arg("print_steps") = false)
        .def("measure", &Circuit::measure)
        .def("sample", &Circuit::sample, py::arg("num_shots"))
        .def("reset", &Circuit::reset)

        // Utilities
        .def("print_state", &Circuit::printState);
}
