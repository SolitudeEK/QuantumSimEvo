#ifndef Shor_H
#define Shor_H
#include <vector>
#include "../QuantumSimEvo/Circuit.h"

class Shor
{
public:
    // Run Shor's factoring algorithm for N with co-prime guess a.
    // Uses 4n+2 qubits where n = ceil(log2(N)).
    static void runShor(int N, int a);

private:
    // QFT helpers (operate on the qubits listed in `reg`, in order)
    static void applyQFT(std::vector<size_t>& reg, Circuit& qc);
    static void applyIQFT(std::vector<size_t>& reg, Circuit& qc);

    // Constant adders in the QFT (phase) basis. `a` may be negative;
    // it is reduced modulo 2^|reg| before phases are applied.
    static void addConstantQFT(std::vector<size_t>& reg, int a, Circuit& qc);
    static void cAddConstantQFT(size_t ctrl, std::vector<size_t>& reg, int a, Circuit& qc);
    static void ccAddConstantQFT(size_t c1, size_t c2, std::vector<size_t>& reg, int a, Circuit& qc);

    // CCX (Toffoli) and Fredkin (controlled SWAP), built from H + MCPhase + CNOT
    static void toffoli(size_t c1, size_t c2, size_t t, Circuit& qc);
    static void cSwap(size_t ctrl, size_t a, size_t b, Circuit& qc);

    // Beauregard doubly-controlled modular adder.
    // breg has n+1 qubits, ancilla is one qubit (returned to |0>).
    static void ccModAdd(size_t c1, size_t c2,
                         std::vector<size_t>& breg, size_t ancilla,
                         int a, int N, Circuit& qc);

    // Singly-controlled modular multiplier: when ctrl=1, |x> -> |a*x mod N>.
    static void cModMult(size_t ctrl,
                         std::vector<size_t>& xreg,
                         std::vector<size_t>& breg,
                         size_t ancilla,
                         int a, int N, Circuit& qc);

    // Top-level QPE driver. Returns the (y, count) histogram of phase-register
    // samples, sorted by count descending. y is the integer with phase_reg[0]
    // as MSB, so the estimated phase is y / 2^(2n).
    static std::vector<std::pair<size_t, int>> shorsAlgorithm4n2(int a, int N, Circuit& qc);

    // Number-theoretic helpers
    static int modExp(int base, int exp, int mod);
    static int modInverse(int a, int N);
    static int gcd(int a, int b);

    // Phase post-processing
    static double binaryFractionToDouble(const std::vector<int>& bits);
    static int getDenominator(double phase, int max_den);
};
#endif
