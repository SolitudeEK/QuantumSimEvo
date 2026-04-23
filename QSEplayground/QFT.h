#pragma once

#include <vector>
#include "../QuantumSimEvo/Circuit.h"

class QFT
{
public:
    static void createQFT(Circuit& circuit, const std::vector<size_t>& qubits);
    static void createIQFT(Circuit& circuit, const std::vector<size_t>& qubits);
    static void runQFT(int numQubits, int shots = 1000);
    static void runIQFT(int numQubits, int shots = 1000);
};
