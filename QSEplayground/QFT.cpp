#include "QFT.h"

#include <cmath>
#include <iostream>
#include <map>

static const double PI = 3.14159265358979323846;

void QFT::createQFT(Circuit& circuit, const std::vector<size_t>& qubits)
{
    int n = static_cast<int>(qubits.size());
    for (int i = 0; i < n; ++i) {
        circuit.hadamard(qubits[i]);
        for (int j = i + 1; j < n; ++j) {
            double theta = PI / std::pow(2.0, j - i);
            circuit.cphase(theta, qubits[j], qubits[i]);
        }
    }
    for (int i = 0; i < n / 2; ++i) {
        circuit.swap(qubits[i], qubits[n - 1 - i]);
    }
}

void QFT::createIQFT(Circuit& circuit, const std::vector<size_t>& qubits)
{
    int n = static_cast<int>(qubits.size());
    for (int i = 0; i < n / 2; ++i) {
        circuit.swap(qubits[i], qubits[n - 1 - i]);
    }
    for (int i = n - 1; i >= 0; --i) {
        for (int j = n - 1; j > i; --j) {
            double theta = -PI / std::pow(2.0, j - i);
            circuit.cphase(theta, qubits[j], qubits[i]);
        }
        circuit.hadamard(qubits[i]);
    }
}

void QFT::runQFT(int numQubits, int shots)
{
    std::cout << "\n========================================\n";
    std::cout << "  QUANTUM FOURIER TRANSFORM (" << numQubits << " qubits)\n";
    std::cout << "========================================\n";

    Circuit circuit(numQubits);

    circuit.pauliX(0);

    std::vector<size_t> qubits;
    for (int i = 0; i < numQubits; ++i)
        qubits.push_back(static_cast<size_t>(i));

    createQFT(circuit, qubits);

    circuit.execute(true);

    std::cout << "\n--- State after QFT ---\n";
    circuit.printState();

    auto results = circuit.sample(shots);

    std::map<size_t, int> counts;
    for (size_t r : results)
        counts[r]++;

    std::cout << "\n--- Measurement Results (" << shots << " shots) ---\n";
    for (auto const& [state, count] : counts) {
        std::cout << "|" << state << "> : " << count
                  << " (" << (100.0 * count / shots) << "%)\n";
    }
}

void QFT::runIQFT(int numQubits, int shots)
{
    std::cout << "\n========================================\n";
    std::cout << "  QUANTUM INVERSE FOURIER TRANSFORM (" << numQubits << " qubits)\n";
    std::cout << "========================================\n";

    Circuit circuit(numQubits);

    circuit.pauliX(0);

    std::vector<size_t> qubits;
    for (int i = 0; i < numQubits; ++i)
        qubits.push_back(static_cast<size_t>(i));

    createIQFT(circuit, qubits);

    circuit.execute(true);

    std::cout << "\n--- State after IQFT ---\n";
    circuit.printState();

    auto results = circuit.sample(shots);

    std::map<size_t, int> counts;
    for (size_t r : results)
        counts[r]++;

    std::cout << "\n--- Measurement Results (" << shots << " shots) ---\n";
    for (auto const& [state, count] : counts) {
        std::cout << "|" << state << "> : " << count
            << " (" << (100.0 * count / shots) << "%)\n";
    }
}
