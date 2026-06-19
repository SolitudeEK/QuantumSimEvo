#pragma once
#include <utility>
#include <vector>

// ============================================================================
// Travelling Salesman Problem via the native (city x time-step) QUBO, solved
// with a QAOA-style circuit on the QuantumSimEvo state-vector simulator.
//
// Binary variable x_{v,t} = 1 means "city v is visited at step t".
// With n cities this uses n*n qubits (qubit index = getIndex(v, t, n)).
//
// QUBO cost (closed tour, step n wraps to step 0):
//   H = A * sum_v ( 1 - sum_t x_{v,t} )^2          // each city visited once
//     + A * sum_t ( 1 - sum_v x_{v,t} )^2          // one city per step
//     + B * sum_t sum_{u!=v} w_{uv} x_{u,t} x_{v,t+1}   // route length
// where A = penalty (large) and B = weight.
//
// The circuit prepares |+...+>, applies one QAOA cost+mixer layer, samples the
// final state, and we post-select the lowest-cost feasible bitstring. The
// result is validated against the brute-force optimal permutation tour.
// ============================================================================
class TVS {
public:
    // Qubit index for "city `city` at step `step`" (n cities).
    static size_t getIndex(size_t city, size_t step, size_t n);

    // Ising coefficients (h_i Z_i + J_ij Z_i Z_j) of the TSP QUBO. Built once so
    // the circuit and quboCost cannot drift apart. Returns the per-qubit field h;
    // J is filled as a map keyed by (i*qubits + j) with i < j.
    static std::vector<double> buildIsing(size_t n, double penalty, double weight,
                                          const std::vector<std::vector<double>>& distances,
                                          std::vector<std::pair<size_t, size_t>>& zzPairs,
                                          std::vector<double>& zzCoeffs,
                                          bool rowConstraint = true,
                                          bool colConstraint = true);

    // Build + run the p-layer QAOA circuit; returns `shots` raw measurement
    // samples. gammas/betas must each have `p` entries (one per layer).
    static std::vector<size_t> createCircuit(size_t n, double penalty, double weight,
                                             const std::vector<double>& gammas,
                                             const std::vector<double>& betas,
                                             const std::vector<std::vector<double>>& distances,
                                             int shots = 10000,
                                             bool printSteps = false);

    // Decode one measurement (qubit q = getIndex(v,t,n)) into the visiting
    // order [city@step0, city@step1, ...]. Empty if the assignment is not a
    // valid permutation (some step has != 1 city, or a city is repeated).
    static std::vector<int> decodeSample(size_t sample, size_t n);

    // Energy of an assignment under the QUBO cost above. Lets you "see" what
    // the QUBO scores a given bitstring, feasible or not.
    static double quboCost(const std::vector<int>& x, size_t n,
                           const std::vector<std::vector<double>>& dist,
                           double penalty, double weight);

    // Length of a closed tour visiting `order` and returning to the start.
    static double tourLength(const std::vector<std::vector<double>>& dist,
                             const std::vector<int>& order);

    // Brute-force optimal closed tour (fixes city 0 as start). For validation.
    static std::vector<int> bruteForceOptimal(const std::vector<std::vector<double>>& dist,
                                              double& bestLen);

    // End-to-end demo: run the circuit, decode + score samples, report the best
    // tour found, and compare it with the brute-force optimum.
    static void runTVS(const std::vector<std::vector<double>>& dist,
                       double penalty = -1.0, double weight = 1.0,
                       double gamma = 0.8, double beta = 0.4, int shots = 20000,
                       int layers = 4);
};
