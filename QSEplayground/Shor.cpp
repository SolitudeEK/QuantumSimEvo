#include "Shor.h"
#include "../QuantumSimEvo/Circuit.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <unordered_map>

const static double PI = 3.14159265358979323846;

// ============================================================================
// QFT / inverse QFT (no bit-reversal swap)
// ============================================================================
void Shor::applyQFT(std::vector<size_t>& reg, Circuit& qc)
{
    int n = (int)reg.size();
    for (int i = 0; i < n; ++i) {
        qc.hadamard(reg[i]);
        for (int j = i + 1; j < n; ++j) {
            double theta = PI / std::pow(2.0, j - i);
            qc.cphase(theta, reg[j], reg[i]);
        }
    }
}

void Shor::applyIQFT(std::vector<size_t>& reg, Circuit& qc)
{
    int n = (int)reg.size();
    for (int i = n - 1; i >= 0; --i) {
        for (int j = n - 1; j > i; --j) {
            double theta = -PI / std::pow(2.0, j - i);
            qc.cphase(theta, reg[j], reg[i]);
        }
        qc.hadamard(reg[i]);
    }
}

// ============================================================================
// Constant adders in the QFT basis.
// Convention (matches applyQFT in this file): reg[0] holds the MSB of the
// register's integer value, so applyQFT puts phase 2*pi*v / 2^(m-i) on reg[i].
// To add 'a', we apply the same phase pattern with v <- a (mod 2^m), which
// reduces to angle_i = sum_{k=0..m-i-1} a_k * pi / 2^(m-i-1-k) on reg[i].
// 'a' may be negative; it is reduced mod 2^m before the phases are computed.
// ============================================================================
static int reduceMod2k(int a, int m)
{
    long long mod = 1LL << m;
    long long r = ((long long)a % mod + mod) % mod;
    return (int)r;
}

static double phaseAngleForBit(int v, int m, int i)
{
    double angle = 0.0;
    for (int k = 0; k < m - i; ++k) {
        if ((v >> k) & 1) angle += PI / std::pow(2.0, m - i - 1 - k);
    }
    return angle;
}

void Shor::addConstantQFT(std::vector<size_t>& reg, int a, Circuit& qc)
{
    int m = (int)reg.size();
    int v = reduceMod2k(a, m);
    for (int i = 0; i < m; ++i) {
        double angle = phaseAngleForBit(v, m, i);
        if (angle != 0.0) qc.phase(angle, reg[i]);
    }
}

void Shor::cAddConstantQFT(size_t ctrl, std::vector<size_t>& reg, int a, Circuit& qc)
{
    int m = (int)reg.size();
    int v = reduceMod2k(a, m);
    for (int i = 0; i < m; ++i) {
        double angle = phaseAngleForBit(v, m, i);
        if (angle != 0.0) qc.cphase(angle, ctrl, reg[i]);
    }
}

void Shor::ccAddConstantQFT(size_t c1, size_t c2, std::vector<size_t>& reg, int a, Circuit& qc)
{
    int m = (int)reg.size();
    int v = reduceMod2k(a, m);
    std::vector<size_t> ctrls = { c1, c2 };
    for (int i = 0; i < m; ++i) {
        double angle = phaseAngleForBit(v, m, i);
        if (angle != 0.0) qc.mcphase(angle, ctrls, reg[i]);
    }
}

// ============================================================================
// Toffoli and controlled-SWAP, built from primitives the Circuit exposes.
// ============================================================================
void Shor::toffoli(size_t c1, size_t c2, size_t t, Circuit& qc)
{
    // CCX = H_t * CCZ * H_t  and  CCZ = MCPhase(pi)
    qc.hadamard(t);
    qc.mcphase(PI, { c1, c2 }, t);
    qc.hadamard(t);
}

void Shor::cSwap(size_t ctrl, size_t a, size_t b, Circuit& qc)
{
    // CSWAP(ctrl, a, b) = CNOT(b,a); Toffoli(ctrl,a,b); CNOT(b,a)
    qc.cnot(b, a);
    toffoli(ctrl, a, b, qc);
    qc.cnot(b, a);
}

// ============================================================================
// Beauregard doubly-controlled modular adder.
//   In:  |c1, c2, b, anc=0>  with  0 <= b < N  and  0 <= a < N
//   Out: |c1, c2, (b + a) mod N, 0>   if  c1 = c2 = 1
//        |c1, c2, b, 0>               otherwise
// breg is held in the QFT basis on entry and exit.
// ============================================================================
void Shor::ccModAdd(size_t c1, size_t c2,
                    std::vector<size_t>& breg, size_t ancilla,
                    int a, int N, Circuit& qc)
{
    // breg uses MSB-at-front convention (matches applyQFT), so the overflow
    // / sign qubit is breg[0], not breg.back().
    size_t msb = breg.front();

    // 1. b += a (controlled on c1 AND c2)
    ccAddConstantQFT(c1, c2, breg, a, qc);
    // 2. b -= N (unconditional)
    addConstantQFT(breg, -N, qc);
    // 3-5. copy MSB (sign bit) into ancilla
    applyIQFT(breg, qc);
    qc.cnot(msb, ancilla);
    applyQFT(breg, qc);
    // 6. b += N controlled on ancilla (restore on underflow)
    cAddConstantQFT(ancilla, breg, N, qc);
    // 7. b -= a  (controlled on c1 AND c2). Now MSB encodes whether
    //    we are in the "wrap" branch, in a way the ancilla can be uncomputed.
    ccAddConstantQFT(c1, c2, breg, -a, qc);
    // 8-12. uncompute ancilla using the (now inverted) MSB
    applyIQFT(breg, qc);
    qc.pauliX(msb);
    qc.cnot(msb, ancilla);
    qc.pauliX(msb);
    applyQFT(breg, qc);
    // 13. b += a (final). Net effect is (b + a) mod N when c1=c2=1.
    ccAddConstantQFT(c1, c2, breg, a, qc);
}

// ============================================================================
// Singly-controlled modular multiplier U_a:  |x> -> |a*x mod N>  if ctrl=1.
// xreg has n qubits (LSB at xreg[0]); breg has n+1 qubits, ancilla 1 qubit;
// breg and ancilla start and end in |0>.
// ============================================================================
void Shor::cModMult(size_t ctrl,
                    std::vector<size_t>& xreg,
                    std::vector<size_t>& breg,
                    size_t ancilla,
                    int a, int N, Circuit& qc)
{
    int n = (int)xreg.size();

    // Phase 1: b += a*x (mod N), controlled on ctrl
    applyQFT(breg, qc);
    int factor = ((a % N) + N) % N;
    for (int i = 0; i < n; ++i) {
        ccModAdd(ctrl, xreg[i], breg, ancilla, factor, N, qc);
        factor = (int)(((long long)factor * 2) % N);
    }
    applyIQFT(breg, qc);

    // Phase 2: controlled-SWAP of x and the lower n bits of b.
    // xreg uses LSB-at-front (xreg[i] = bit i of x); breg uses MSB-at-front
    // with the overflow bit at breg[0], so bit i of b lives at breg[n - i].
    for (int i = 0; i < n; ++i) {
        cSwap(ctrl, xreg[i], breg[n - i], qc);
    }

    // Phase 3: subtract a^{-1} * x' (where x' = a*x mod N now lives in xreg).
    // Result: breg returns to |0>, xreg holds a*x mod N.
    int a_inv = modInverse(((a % N) + N) % N, N);
    applyQFT(breg, qc);
    int factor_inv = a_inv;
    for (int i = 0; i < n; ++i) {
        int sub = (N - factor_inv) % N;     // adding (N - z) mod N == subtracting z mod N
        ccModAdd(ctrl, xreg[i], breg, ancilla, sub, N, qc);
        factor_inv = (int)(((long long)factor_inv * 2) % N);
    }
    applyIQFT(breg, qc);
}

// ============================================================================
// QPE driver: 2n-qubit phase register, full inverse QFT readout.
// Returns the 2n estimated phase bits in MSB-first order.
// ============================================================================
std::vector<std::pair<size_t, int>> Shor::shorsAlgorithm4n2(int a, int N, Circuit& qc)
{
    int n = (int)std::ceil(std::log2((double)N));
    if (n < 1) n = 1;
    int m = 2 * n;

    std::vector<size_t> phase_reg;
    for (int i = 0; i < m; ++i) phase_reg.push_back((size_t)i);

    std::vector<size_t> x_reg;
    for (int i = 0; i < n; ++i) x_reg.push_back((size_t)(m + i));

    std::vector<size_t> b_reg;
    for (int i = 0; i < n + 1; ++i) b_reg.push_back((size_t)(m + n + i));

    size_t ancilla = (size_t)(m + n + (n + 1));   // index 4n+1, the only ancilla used.

    // 1. Hadamard on the phase register
    for (size_t q : phase_reg) qc.hadamard(q);

    // 2. Initialize work register x to |1>  (LSB at x_reg[0])
    qc.pauliX(x_reg[0]);

    // 3. Controlled U_a^{2^p}. The applyQFT convention here treats phase_reg[0]
    //    as the MSB of the recovered integer, so the highest power must be
    //    controlled from phase_reg[0] and the lowest from phase_reg[m-1].
    int power = ((a % N) + N) % N;
    for (int k = m - 1; k >= 0; --k) {
        cModMult(phase_reg[k], x_reg, b_reg, ancilla, power, N, qc);
        power = (int)(((long long)power * power) % N);
    }

    // 4. Inverse QFT on the phase register
    applyIQFT(phase_reg, qc);

    // 5. Run the circuit once, then draw many samples from the resulting
    //    state without collapsing. The sample whose phase-register bits
    //    appear most often is most likely to be a good s/r approximation.
    qc.execute(true);
    const int num_shots = 1000;
    std::vector<size_t> shots = qc.sample(num_shots);

    std::unordered_map<size_t, int> hist;
    for (size_t s : shots) {
        size_t y = 0;
        for (int i = 0; i < m; ++i) {
            size_t qbit = (s >> phase_reg[i]) & 1ULL;
            y |= qbit << (m - 1 - i);   // phase_reg[0] is MSB of y
        }
        hist[y]++;
    }

    std::vector<std::pair<size_t, int>> ranked(hist.begin(), hist.end());
    std::sort(ranked.begin(), ranked.end(),
        [](const std::pair<size_t, int>& x, const std::pair<size_t, int>& y) {
            return x.second > y.second;
        });
    return ranked;
}

// ============================================================================
// Number-theoretic helpers
// ============================================================================
int Shor::gcd(int a, int b)
{
    while (b != 0) {
        int t = b;
        b = a % b;
        a = t;
    }
    return a < 0 ? -a : a;
}

int Shor::modExp(int base, int exp, int mod)
{
    long long res = 1;
    long long b = ((long long)base % mod + mod) % mod;
    while (exp > 0) {
        if (exp & 1) res = (res * b) % mod;
        exp >>= 1;
        b = (b * b) % mod;
    }
    return (int)res;
}

int Shor::modInverse(int a, int N)
{
    // Iterative extended Euclidean. Caller must ensure gcd(a, N) = 1.
    int g_old = a, g_cur = N;
    int x_old = 1, x_cur = 0;
    while (g_cur != 0) {
        int q = g_old / g_cur;
        int t = g_old - q * g_cur; g_old = g_cur; g_cur = t;
        t = x_old - q * x_cur;     x_old = x_cur; x_cur = t;
    }
    return ((x_old % N) + N) % N;
}

// ============================================================================
// Phase post-processing
// ============================================================================
double Shor::binaryFractionToDouble(const std::vector<int>& bits)
{
    double value = 0.0;
    for (size_t i = 0; i < bits.size(); ++i) {
        if (bits[i] == 1) value += 1.0 / std::pow(2.0, (double)(i + 1));
    }
    return value;
}

int Shor::getDenominator(double phase, int max_den)
{
    // Continued-fractions: return the largest convergent denominator <= max_den.
    if (phase <= 0.0 || phase >= 1.0) return 1;
    long long h_prev = 0, h_cur = 1;
    long long k_prev = 1, k_cur = 0;
    double x = phase;
    for (int iter = 0; iter < 64; ++iter) {
        long long ai = (long long)std::floor(x);
        long long h_next = ai * h_cur + h_prev;
        long long k_next = ai * k_cur + k_prev;
        if (k_next > (long long)max_den) break;
        h_prev = h_cur; h_cur = h_next;
        k_prev = k_cur; k_cur = k_next;
        double frac = x - (double)ai;
        if (std::abs(frac) < 1e-12) break;
        x = 1.0 / frac;
    }
    return (int)k_cur == 0 ? 1 : (int)k_cur;
}

// ============================================================================
// Entry point
// ============================================================================
void Shor::runShor(int N, int a)
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "      SHOR'S ALGORITHM FACTORIZATION      " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Target N = " << N << ", Base Guess a = " << a << std::endl;

    // Trivial classical pre-checks
    if (N < 2) { std::cout << "N must be >= 2." << std::endl; return; }
    if (N % 2 == 0) {
        std::cout << "\nN is even. Classical factor: 2 (and " << N / 2 << ")." << std::endl;
        return;
    }
    if (a < 2 || a >= N) {
        std::cout << "\n'a' must satisfy 2 <= a < N." << std::endl;
        return;
    }
    int initial_gcd = gcd(a, N);
    if (initial_gcd > 1) {
        std::cout << "\nLucky guess: gcd(a, N) = " << initial_gcd << " is a non-trivial factor." << std::endl;
        std::cout << "No quantum circuit required." << std::endl;
        return;
    }

    int n = (int)std::ceil(std::log2((double)N));
    int required_qubits = 4 * n + 2;

    std::cout << "Bit size of N (n) = " << n << " bits" << std::endl;
    std::cout << "Qubits required (4n + 2) = " << required_qubits << " qubits" << std::endl;

    if (required_qubits > 30) {
        std::cout << "\n[ERROR] Simulator memory limit exceeded (30 qubits)." << std::endl;
        std::cout << "The 4n+2 architecture supports at most n = 7 (N <= 127)." << std::endl;
        return;
    }

    std::cout << "\n[Running quantum phase estimation...]" << std::endl;
    Circuit qc(required_qubits);

    std::vector<std::pair<size_t, int>> ranked = shorsAlgorithm4n2(a, N, qc);
    int m = 2 * n;
    long long denom = 1LL << m;

    std::cout << "\n--- Quantum Results ---" << std::endl;
    std::cout << "Top sampled phase-register values (y / 2^" << m << " = phase):" << std::endl;
    int top_to_show = std::min<int>(8, (int)ranked.size());
    for (int i = 0; i < top_to_show; ++i) {
        std::cout << "  y = " << ranked[i].first
                  << "  phase = " << ((double)ranked[i].first / (double)denom)
                  << "  count = " << ranked[i].second << std::endl;
    }

    // Walk down the histogram, trying each candidate phase. Period r is the
    // continued-fraction denominator of y/2^m capped by N. Accept the first
    // (r, half_power) that yields a non-trivial factor of N.
    int max_candidates = std::min<int>(32, (int)ranked.size());
    for (int idx = 0; idx < max_candidates; ++idx) {
        size_t y = ranked[idx].first;
        if (y == 0) continue;                       // s=0 gives no info
        double phase = (double)y / (double)denom;

        int r = getDenominator(phase, N);
        if (r <= 1) continue;

        // If r is odd but a^r == 1 only for an even multiple, double once.
        if (r % 2 != 0) {
            if (2 * r <= N && modExp(a, 2 * r, N) == 1) r *= 2;
            else continue;
        }

        // Verify r is the (or a multiple of) the actual order.
        if (modExp(a, r, N) != 1) continue;

        int half_power = modExp(a, r / 2, N);
        if (half_power == 1 || half_power == N - 1) continue;

        int factor1 = gcd(half_power - 1, N);
        int factor2 = gcd(half_power + 1, N);
        if (factor1 <= 1 || factor1 >= N || factor2 <= 1 || factor2 >= N) continue;

        std::cout << "\nUsing candidate y = " << y << ", phase = " << phase
                  << ", recovered r = " << r << std::endl;
        std::cout << "\n========================================" << std::endl;
        std::cout << "                SUCCESS!                " << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Factors of " << N << ": " << factor1 << " and " << factor2 << std::endl;
        if ((long long)factor1 * factor2 == N) {
            std::cout << "(Verified: " << factor1 << " * " << factor2 << " = " << N << ")" << std::endl;
        }
        return;
    }

    std::cout << "\n[FAILED] No sampled phase yielded a non-trivial factor."
              << " Try a different 'a' or rerun." << std::endl;
}
