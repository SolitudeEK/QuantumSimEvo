#include "Grover.h"
#include "../QuantumSimEvo/Circuit.h"
#include <iostream>
#include <map>
#include <vector>
#include <cmath>
#include <algorithm>
#include <bitset>

static const double PI = 3.14159265358979323846;

void Grover::runGrover(int target, int shots)
{
    // --- Estimate resources ---
    // Minimum qubits to represent the target index
    int nQubits = 1;
    while ((1 << nQubits) <= target) ++nQubits;

    int searchSpace = 1 << nQubits;
    int iterations = static_cast<int>(std::round((PI / 4.0) * std::sqrt(static_cast<double>(searchSpace))));
    if (iterations < 1) iterations = 1;

    std::cout << "=== Grover's Search ===\n";
    std::cout << "  Search space  : " << searchSpace << " items\n";
    std::cout << "  Qubits needed : " << nQubits << "\n";
    std::cout << "  Target state  : |" << target << "> ("
              << std::bitset<32>(target).to_string().substr(32 - nQubits) << " binary)\n";
    std::cout << "  Iterations    : " << iterations << "\n\n";

    // --- Build and execute the circuit ---
    Circuit circuit(nQubits);
    createGroverCircuit(circuit, nQubits, target, iterations);
    circuit.execute(true);

    // --- Sample without collapsing ---
    auto results = circuit.sample(shots);

    std::map<size_t, int> counts;
    for (size_t r : results) counts[r]++;

    std::vector<std::pair<int, size_t>> sorted;
    sorted.reserve(counts.size());
    for (auto& [val, cnt] : counts) sorted.push_back({ cnt, val });
    std::sort(sorted.rbegin(), sorted.rend());

    std::cout << "Results (" << shots << " shots):\n";
    int display = std::min(static_cast<int>(sorted.size()), 8);
    for (int i = 0; i < display; ++i) {
        int cnt = sorted[i].first;
        size_t val = sorted[i].second;
        std::cout << "  |" << val << "> ("
                  << std::bitset<32>(val).to_string().substr(32 - nQubits) << "): "
                  << cnt << " / " << shots
                  << " (" << std::round(1000.0 * cnt / shots) / 10.0 << "%)"
                  << (val == static_cast<size_t>(target) ? " <-- target" : "") << "\n";
    }

    // --- Final collapsed measurement ---
    size_t measured = circuit.measure();
    std::cout << "\nCollapsed measurement: |" << measured << "> ("
              << std::bitset<32>(measured).to_string().substr(32 - nQubits) << ")\n";
    std::cout << (measured == static_cast<size_t>(target) ? "  SUCCESS\n" : "  MISS (try more iterations)\n");
}

void Grover::createGroverCircuit(Circuit& circuit, int nQubits, int target, int iterations)
{
    // Uniform superposition
    for (int i = 0; i < nQubits; ++i)
        circuit.hadamard(i);

    for (int step = 0; step < iterations; ++step) {
        groverOracle(circuit, nQubits, target);
        groverDiffusion(circuit, nQubits);
    }
}

void Grover::groverOracle(Circuit& circuit, int nQubits, int target)
{
    for (int i = 0; i < nQubits; ++i)
        if (!((target >> i) & 1)) circuit.pauliX(i);

    std::vector<size_t> controls;
    controls.reserve(nQubits - 1);
    for (int i = 0; i < nQubits - 1; ++i) controls.push_back(i);
    circuit.mcphase(PI, controls, nQubits - 1);

    for (int i = 0; i < nQubits; ++i)
        if (!((target >> i) & 1)) circuit.pauliX(i);
}

void Grover::groverDiffusion(Circuit& circuit, int nQubits)
{
    for (int i = 0; i < nQubits; ++i) circuit.hadamard(i);

    for (int i = 0; i < nQubits; ++i) circuit.pauliX(i);

    std::vector<size_t> controls;
    controls.reserve(nQubits - 1);
    for (int i = 0; i < nQubits - 1; ++i) controls.push_back(i);
    circuit.mcphase(PI, controls, nQubits - 1);

    for (int i = 0; i < nQubits; ++i) circuit.pauliX(i);

    for (int i = 0; i < nQubits; ++i) circuit.hadamard(i);
}
