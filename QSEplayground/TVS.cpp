#include "TVS.h"
#include "../QuantumSimEvo/Circuit.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <unordered_map>

size_t TVS::getIndex(size_t city, size_t step, size_t n)
{
    return city * n + step;
}

// ============================================================================
// Build the Ising form of the TSP QUBO.
//
// QUBO cost is sum_i c_i x_i + sum_{i<j} Q_ij x_i x_j. Substituting the
// standard map x = (1 - Z)/2 gives the Ising Hamiltonian
//   H = const + sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j
// with the contributions:
//   linear c_i x_i      -> h_i  -= c_i / 2
//   quad   Q_ij x_i x_j -> h_i  -= Q_ij / 4 , h_j -= Q_ij / 4 , J_ij += Q_ij / 4
//
// Returns h (per qubit). The ZZ terms are returned as parallel arrays
// (zzPairs[k], zzCoeffs[k] = J for that pair). Building both the linear and
// quadratic parts here is what step 1 of the review was about: the previous
// circuit applied only the ZZ terms and silently dropped every single-qubit Z
// term, so the phase separator did not match the real cost.
// ============================================================================
std::vector<double> TVS::buildIsing(size_t n, double penalty, double weight,
                                    const std::vector<std::vector<double>>& distances,
                                    std::vector<std::pair<size_t, size_t>>& zzPairs,
                                    std::vector<double>& zzCoeffs,
                                    bool rowConstraint, bool colConstraint)
{
    const size_t qubits = n * n;
    std::vector<double> h(qubits, 0.0);

    // Accumulate J on a dense map first so duplicate pairs merge cleanly.
    std::unordered_map<size_t, double> J;
    auto addZZ = [&](size_t a, size_t b, double coeff) {
        if (a > b) std::swap(a, b);
        J[a * qubits + b] += coeff;
    };
    auto addLinear = [&](size_t i, double c) { h[i] -= c / 2.0; };
    auto addQuad = [&](size_t i, size_t j, double q) {
        h[i] -= q / 4.0;
        h[j] -= q / 4.0;
        addZZ(i, j, q / 4.0);
    };

    // Constraint A*(1 - S)^2 = A*[1 - sum_i x_i + 2*sum_{i<j} x_i x_j] over a
    // group: linear coeff -A per member, quad coeff +2A per pair.
    auto addConstraint = [&](const std::vector<size_t>& group) {
        for (size_t a = 0; a < group.size(); ++a) {
            addLinear(group[a], -penalty);
            for (size_t b = a + 1; b < group.size(); ++b)
                addQuad(group[a], group[b], 2.0 * penalty);
        }
    };

    // Each city visited exactly once (one constraint per row v). May be skipped
    // when the row one-hot structure is enforced by the mixer instead.
    if (rowConstraint)
        for (size_t v = 0; v < n; ++v) {
            std::vector<size_t> row(n);
            for (size_t t = 0; t < n; ++t) row[t] = getIndex(v, t, n);
            addConstraint(row);
        }
    // Exactly one city per step (one constraint per column t).
    if (colConstraint)
        for (size_t t = 0; t < n; ++t) {
            std::vector<size_t> col(n);
            for (size_t v = 0; v < n; ++v) col[v] = getIndex(v, t, n);
            addConstraint(col);
        }
    // Route cost B * w_{uv} x_{u,t} x_{v,t+1} (cyclic).
    for (size_t u = 0; u < n; ++u)
        for (size_t v = 0; v < n; ++v) {
            if (u == v) continue;
            double q = weight * distances[u][v];
            for (size_t t = 0; t < n; ++t) {
                size_t next_t = (t + 1) % n;
                addQuad(getIndex(u, t, n), getIndex(v, next_t, n), q);
            }
        }

    zzPairs.clear();
    zzCoeffs.clear();
    zzPairs.reserve(J.size());
    zzCoeffs.reserve(J.size());
    for (const auto& kv : J) {
        if (std::abs(kv.second) < 1e-15) continue;
        zzPairs.emplace_back(kv.first / qubits, kv.first % qubits);
        zzCoeffs.push_back(kv.second);
    }
    return h;
}

// ============================================================================
// p-layer Quantum Alternating Operator Ansatz (constraint-preserving QAOA).
//
//   * Initial state: a feasible permutation (the identity tour, city v at step
//     v), so we START inside the feasible subspace instead of an equal
//     superposition over all 2^(n*n) bitstrings.
//   * Mixer: an XY ring mixer WITHIN each city-row. The row qubits are
//     contiguous (getIndex(v,t,n) = v*n + t), and e^{-i beta (X_i X_j + Y_i Y_j)}
//     only HOPS the single excitation around the ring, so each city stays
//     assigned to exactly one step. The "each city visited once" constraint can
//     therefore never be violated -- it is structural, not a penalty.
//   * Phase separator: route cost + the COLUMN penalty only (one city per step).
//     The row penalty is dropped because the mixer enforces it.
//
// This shrinks the reachable space from 2^(n*n) to n^n and keeps amplitude on
// (or very near) real tours, which is what raises the feasible-sample rate.
// ============================================================================
std::vector<size_t> TVS::createCircuit(size_t n, double penalty, double weight,
                                       const std::vector<double>& gammas,
                                       const std::vector<double>& betas,
                                       const std::vector<std::vector<double>>& distances,
                                       int shots, bool printSteps)
{
    const double PI = 3.14159265358979323846;
    const size_t qubits = n * n;
    Circuit qc(qubits);

    // Phase separator: route cost + column penalty only (row is structural).
    std::vector<std::pair<size_t, size_t>> zzPairs;
    std::vector<double> zzCoeffs;
    std::vector<double> h = buildIsing(n, penalty, weight, distances, zzPairs, zzCoeffs,
                                       /*rowConstraint=*/false, /*colConstraint=*/true);

    // Feasible initial state: identity tour, city v visited at step v.
    for (size_t v = 0; v < n; ++v)
        qc.pauliX(getIndex(v, v, n));

    // e^{-i beta (X_a X_b + Y_a Y_b)} on a pair. XX and YY commute, so apply
    // Rxx(2 beta) then Ryy(2 beta); each is a basis change around the standard
    // Rzz = CNOT, RZ, CNOT core (which realizes e^{-i theta/2 Z_a Z_b}).
    auto xyInteraction = [&](size_t a, size_t b, double beta) {
        // Rxx(2 beta): conjugate Rzz by Hadamards.
        qc.hadamard(a); qc.hadamard(b);
        qc.cnot(a, b); qc.rotateZ(2 * beta, b); qc.cnot(a, b);
        qc.hadamard(a); qc.hadamard(b);
        // Ryy(2 beta): conjugate Rzz by RX(+-pi/2).
        qc.rotateX(PI / 2, a); qc.rotateX(PI / 2, b);
        qc.cnot(a, b); qc.rotateZ(2 * beta, b); qc.cnot(a, b);
        qc.rotateX(-PI / 2, a); qc.rotateX(-PI / 2, b);
    };

    const size_t p = std::min(gammas.size(), betas.size());
    for (size_t layer = 0; layer < p; ++layer) {
        const double gamma = gammas[layer];
        const double beta = betas[layer];

        // Phase separator: ZZ phases e^{-i gamma J_ij Z_i Z_j} ...
        for (size_t k = 0; k < zzPairs.size(); ++k) {
            size_t q1 = zzPairs[k].first, q2 = zzPairs[k].second;
            qc.cnot(q1, q2);
            qc.rotateZ(2 * gamma * zzCoeffs[k], q2);
            qc.cnot(q1, q2);
        }
        // ... and single-qubit Z phases e^{-i gamma h_q Z_q}.
        for (size_t q = 0; q < qubits; ++q)
            qc.rotateZ(2 * gamma * h[q], q);

        // XY ring mixer inside each city-row (preserves one excitation per row).
        for (size_t v = 0; v < n; ++v)
            for (size_t t = 0; t < n; ++t) {
                size_t a = getIndex(v, t, n);
                size_t b = getIndex(v, (t + 1) % n, n);
                xyInteraction(a, b, beta);
            }
    }

    qc.execute(printSteps);
    return qc.sample(shots);
}

// ============================================================================
// Decode a measurement into a visiting order, or {} if infeasible.
// ============================================================================
std::vector<int> TVS::decodeSample(size_t sample, size_t n)
{
    std::vector<int> order(n, -1);
    std::vector<char> cityUsed(n, 0);

    for (size_t t = 0; t < n; ++t) {
        int chosen = -1;
        for (size_t v = 0; v < n; ++v) {
            if ((sample >> getIndex(v, t, n)) & 1ULL) {
                if (chosen != -1) return {};      // two cities at this step
                chosen = (int)v;
            }
        }
        if (chosen == -1 || cityUsed[chosen]) return {};  // empty step or repeat
        cityUsed[chosen] = 1;
        order[t] = chosen;
    }
    return order;
}

// ============================================================================
// QUBO cost of an assignment (x[getIndex(v,t,n)] in {0,1}).
// ============================================================================
double TVS::quboCost(const std::vector<int>& x, size_t n,
                     const std::vector<std::vector<double>>& dist,
                     double penalty, double weight)
{
    double cost = 0.0;

    // Each city visited exactly once across steps.
    for (size_t v = 0; v < n; ++v) {
        int s = 0;
        for (size_t t = 0; t < n; ++t) s += x[getIndex(v, t, n)];
        cost += penalty * (1 - s) * (1 - s);
    }
    // Exactly one city per step.
    for (size_t t = 0; t < n; ++t) {
        int s = 0;
        for (size_t v = 0; v < n; ++v) s += x[getIndex(v, t, n)];
        cost += penalty * (1 - s) * (1 - s);
    }
    // Route length (cyclic).
    for (size_t t = 0; t < n; ++t) {
        size_t next_t = (t + 1) % n;
        for (size_t u = 0; u < n; ++u)
            for (size_t v = 0; v < n; ++v) {
                if (u == v) continue;
                if (x[getIndex(u, t, n)] && x[getIndex(v, next_t, n)])
                    cost += weight * dist[u][v];
            }
    }
    return cost;
}

// ============================================================================
// Classical helpers for validation
// ============================================================================
double TVS::tourLength(const std::vector<std::vector<double>>& dist,
                       const std::vector<int>& order)
{
    double len = 0.0;
    const size_t n = order.size();
    for (size_t i = 0; i < n; ++i)
        len += dist[order[i]][order[(i + 1) % n]];   // closed tour
    return len;
}

std::vector<int> TVS::bruteForceOptimal(const std::vector<std::vector<double>>& dist,
                                        double& bestLen)
{
    const int n = (int)dist.size();
    std::vector<int> perm(n - 1);
    std::iota(perm.begin(), perm.end(), 1);          // fix city 0 as the start
    std::vector<int> best;
    bestLen = std::numeric_limits<double>::infinity();

    do {
        std::vector<int> order;
        order.push_back(0);
        order.insert(order.end(), perm.begin(), perm.end());
        double len = tourLength(dist, order);
        if (len < bestLen) { bestLen = len; best = order; }
    } while (std::next_permutation(perm.begin(), perm.end()));

    return best;
}

// ============================================================================
// Driver
// ============================================================================
void TVS::runTVS(const std::vector<std::vector<double>>& dist,
                 double penalty, double weight, double gamma, double beta, int shots,
                 int layers)
{
    const size_t n = dist.size();

    std::cout << "\n========================================\n";
    std::cout << "  TSP via QAOA + XY-mixer ansatz (sim)     \n";
    std::cout << "========================================\n";
    std::cout << "Cities: " << n << "   Qubits (n*n): " << n * n << "\n";

    if (n < 3) { std::cout << "Need at least 3 cities.\n"; return; }

    // Auto penalty: just large enough that breaking a constraint always costs
    // more than any route saving, but small enough that gamma*A phases don't
    // wrap many times around 2*pi (which washes out the route-cost signal).
    // A single edge contributes at most weight*maxDist, and a tour has n edges,
    // so weight*maxDist*n upper-bounds the gain from any feasibility violation.
    if (penalty < 0.0) {
        double maxDist = 0.0;
        for (size_t a = 0; a < n; ++a)
            for (size_t b = 0; b < n; ++b) maxDist = std::max(maxDist, dist[a][b]);
        penalty = weight * maxDist * (double)n + 1.0;
    }

    if (layers < 1) layers = 1;
    // Per-layer schedule: ramp gamma up and beta down across the p layers, the
    // standard annealing-inspired default. Tune the gamma/beta seeds (or replace
    // this with an outer optimizer) to push success rates higher.
    std::vector<double> gammas(layers), betas(layers);
    for (int k = 0; k < layers; ++k) {
        double frac = (layers == 1) ? 1.0 : (double)(k + 1) / (double)layers;
        gammas[k] = gamma * frac;
        betas[k]  = beta * (1.0 - frac) + beta * 0.25;   // keep some mixing on the last layer
    }

    std::cout << "Penalty A: " << penalty << "   Weight B: " << weight
              << "   layers p: " << layers
              << "   gamma seed: " << gamma << "   beta seed: " << beta << "\n";

    std::cout << "\n[Running QAOA circuit and sampling " << shots << " shots...]\n";
    std::vector<size_t> samples = createCircuit(n, penalty, weight, gammas, betas, dist, shots, true);

    // Tally unique samples and pick the lowest-cost FEASIBLE bitstring.
    std::unordered_map<size_t, int> hist;
    for (size_t s : samples) hist[s]++;

    std::vector<int> bestOrder;            // lowest route length among feasible
    double bestLen = std::numeric_limits<double>::infinity();
    double bestQubo = std::numeric_limits<double>::infinity();
    int bestCount = 0;

    std::vector<int> mostOrder;            // most frequently sampled feasible tour
    double mostLen = 0.0, mostQubo = 0.0;
    int mostCount = -1;

    std::vector<int> x(n * n);

    for (const auto& kv : hist) {
        for (size_t q = 0; q < n * n; ++q) x[q] = (int)((kv.first >> q) & 1ULL);
        double cost = quboCost(x, n, dist, penalty, weight);
        std::vector<int> order = decodeSample(kv.first, n);
        if (order.empty()) continue;

        double len = tourLength(dist, order);
        if (len < bestLen) {
            bestLen = len; bestOrder = order; bestQubo = cost; bestCount = kv.second;
        }
        if (kv.second > mostCount) {
            mostCount = kv.second; mostOrder = order; mostLen = len; mostQubo = cost;
        }
    }

    std::cout << "\n--- QUBO result (best feasible sample) ---\n";
    if (bestOrder.empty()) {
        std::cout << "No feasible tour was sampled (raise penalty, shots, or tune "
                     "gamma/beta).\n";
    } else {
        std::cout << "Tour: ";
        for (size_t i = 0; i < bestOrder.size(); ++i) std::cout << bestOrder[i] << " -> ";
        std::cout << bestOrder[0] << "\n";
        std::cout << "QUBO energy: " << bestQubo << "   (route length = " << bestLen << ")\n";
        std::cout << "Sampled " << bestCount << " / " << shots << " times\n";

        std::cout << "\n--- Most probable feasible tour ---\n";
        std::cout << "Tour: ";
        for (size_t i = 0; i < mostOrder.size(); ++i) std::cout << mostOrder[i] << " -> ";
        std::cout << mostOrder[0] << "\n";
        std::cout << "QUBO energy: " << mostQubo << "   (route length = " << mostLen << ")\n";
        std::cout << "Sampled " << mostCount << " / " << shots << " times ("
                  << (100.0 * mostCount / shots) << "% of shots)\n";
    }

    // Brute-force validation.
    double optLen = 0.0;
    std::vector<int> opt = bruteForceOptimal(dist, optLen);
    std::cout << "\n--- Brute-force optimum ---\n";
    std::cout << "Tour: ";
    for (size_t i = 0; i < opt.size(); ++i) std::cout << opt[i] << " -> ";
    std::cout << opt[0] << "\nRoute length: " << optLen << "\n";

    if (!bestOrder.empty()) {
        std::cout << "\n" << (std::abs(bestLen - optLen) < 1e-9
                              ? "SUCCESS: QAOA sample matches the optimal tour."
                              : "Best sampled tour is feasible but sub-optimal.")
                  << "\n";
    }
}
