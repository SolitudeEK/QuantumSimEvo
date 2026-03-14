#include "CircuitUnitaryOperation.h"
#include <omp.h>
#include <complex>
#include "Structurs.h"
#include <vector>
#include <chrono>
#include <iostream>

void CircuitUnitaryOperation::applyGate(StateVector& sv, const Gate2x2& gate, size_t q) {
    std::vector<std::complex<double>>& amplitudes = sv.data();
    size_t N = sv.qubits();
    size_t size = sv.size();

    size_t sectionSize = 1ULL << q;

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < (size >> 1); ++i) {
        // Bit manipulation to find the pair of indices (i0, i1) 
        // affected by the gate on qubit q
        size_t i0 = ((i >> q) << (q + 1)) | (i & (sectionSize - 1));
        size_t i1 = i0 | sectionSize;
 
        std::complex<double> low = amplitudes[i0];
        std::complex<double> high = amplitudes[i1];

        amplitudes[i0] = gate.data[0][0] * low + gate.data[0][1] * high;
        amplitudes[i1] = gate.data[1][0] * low + gate.data[1][1] * high;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "Gate applied in: " << ms.count() << " ms \n"; //Debug timing
}

void CircuitUnitaryOperation::applyCNOT(StateVector& sv, size_t control, size_t target) {
    std::vector<std::complex<double>>& states = sv.data();
    size_t size = sv.size();
    size_t ctrlMask = 1ULL << control;
    size_t tgtMask = 1ULL << target;

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < size; ++i) {
        // Apply X to target ONLY if control bit is 1 
        // and we haven't processed this pair yet (i < (i ^ tgtMask))
        if ((i & ctrlMask) && (i < (i ^ tgtMask))) {
            std::swap(states[i], states[i ^ tgtMask]);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "CNOT applied in: " << ms.count() << " ms \n"; //Debug timing
}

void CircuitUnitaryOperation::applyPauliX(StateVector& sv, size_t q) {
    std::vector<std::complex<double>>& states = sv.data();
    size_t size = sv.size();
    size_t mask = 1ULL << q;

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < size; ++i) {
        if (!(i & mask)) {
            std::swap(states[i], states[i | mask]);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "PauliX applied in: " << ms.count() << " ms \n"; //Debug timing
}

void CircuitUnitaryOperation::applyHadamard(StateVector& sv, size_t targetQubit) {
    std::vector<Complex>& state = sv.data();
    size_t stateSize = sv.size();

    //Precompute the scalar constant (1/sqrt(2))
    const double s = 0.7071067811865475244;

    size_t step = 1ULL << targetQubit;

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < stateSize; i += 2 * step) {
        for (size_t j = i; j < i + step; ++j) {
            size_t idx_zero = j;
            size_t idx_one = j + step;

            Complex v0 = state[idx_zero];
            Complex v1 = state[idx_one];

            state[idx_zero] = s * (v0 + v1);
            state[idx_one] = s * (v0 - v1);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "Hadamard applied in: " << ms.count() << " ms \n"; //Debug timing
}

void CircuitUnitaryOperation::applyPhase(StateVector& sv, size_t q, double theta) {
    // Phase gates only modify the |1> state, so we only touch half the vector
    std::vector<std::complex<double>>& states = sv.data();
    size_t size = sv.size();
    size_t mask = 1ULL << q;
    std::complex<double> phase = std::exp(std::complex<double>(0, theta));

    auto start = std::chrono::high_resolution_clock::now();

     #pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < size; ++i) {
        if (i & mask) {
            states[i] *= phase;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "Phase applied in: " << ms.count() << " ms \n"; //Debug timing
}

void CircuitUnitaryOperation::applyPauliY(StateVector& sv, size_t q) {
    std::vector<std::complex<double>>& states = sv.data();
    size_t size = sv.size();
    size_t mask = 1ULL << q;
    const Complex I(0.0, 1.0);
   
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < size; ++i) {
        if (!(i & mask)) {
            uint64_t idx0 = i;
            uint64_t idx1 = i | mask;

            Complex val0 = states[idx0];
            Complex val1 = states[idx1];

            states[idx0] = -I * val1;
            states[idx1] = I * val0;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "PauliY applied in: " << ms.count() << " ms \n"; //Debug timing
}

void CircuitUnitaryOperation::applyPauliZ(StateVector& sv, size_t q) {
	std::vector<std::complex<double>>& states = sv.data();
	size_t size = sv.size();
	size_t mask = 1ULL << q;

	auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for( std::ptrdiff_t i = 0; i < size; ++i) {
        if (i & mask)
			states[i] = -states[i]; // negation faster than multiplication by -1.0
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
	std::cout << "PauliZ applied in: " << ms.count() << " ms \n"; //Debug timing
}

void CircuitUnitaryOperation::applyRotateX(StateVector& sv, size_t target, double theta) {
    std::vector<Complex>& states = sv.data();
    size_t size = sv.size();
    uint64_t mask = 1ULL << target;

    double c = std::cos(theta / 2.0);
    double s = std::sin(theta / 2.0);
    const Complex miS(0, -s); // -i * sin(theta/2)

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < size; ++i) {
        if (!(i & mask)) {
            uint64_t idx0 = i;
            uint64_t idx1 = i | mask;

            Complex v0 = states[idx0];
            Complex v1 = states[idx1];

            states[idx0] = c * v0 + miS * v1;
            states[idx1] = miS * v0 + c * v1;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "RotateX applied in: " << ms.count() << " ms \n"; //Debug timing
}

void CircuitUnitaryOperation::applyRotateY(StateVector& sv, size_t target, double theta) {
    std::vector<Complex>& states = sv.data();
    size_t size = sv.size();
    uint64_t mask = 1ULL << target;

    auto start = std::chrono::high_resolution_clock::now();

    double c = std::cos(theta / 2.0);
    double s = std::sin(theta / 2.0);

    #pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < size; ++i) {
        if (!(i & mask)) {
            uint64_t idx0 = i;
            uint64_t idx1 = i | mask;

            Complex v0 = states[idx0];
            Complex v1 = states[idx1];

            states[idx0] = c * v0 - s * v1;
            states[idx1] = s * v0 + c * v1;
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "RotateY applied in: " << ms.count() << " ms \n"; //Debug timing
}

void CircuitUnitaryOperation::applyRotateZ(StateVector& sv, size_t target, double theta) {
    std::vector<Complex>& states = sv.data();
    size_t size = sv.size();
    uint64_t mask = 1ULL << target;

    // Precompute phase factors outside the loop
    Complex phase0 = std::exp(Complex(0, -theta / 2.0));
    Complex phase1 = std::exp(Complex(0, theta / 2.0));

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < size; ++i) {
        if (i & mask)
            states[i] *= phase1;
        else
            states[i] *= phase0;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "RotateZ applied in: " << ms.count() << " ms \n"; //Debug timing
}