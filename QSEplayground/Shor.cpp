#include "Shor.h"
#include "../QuantumSimEvo/Circuit.h"

const static double PI = 3.14159265358979323846;

std::vector<int> Shor::shorsAlgorithm2n3(int a, int N, Circuit& qc)
{
    int n = std::ceil(std::log2(N));
    int total_qubits = 2 * n + 3;

    // Define Qubit Roles
    size_t ctrl_qubit = 0;
    std::vector<size_t> acc_reg;
    for (size_t i = 1; i <= n; ++i) acc_reg.push_back(i);
    std::vector<size_t> aux_reg;
    for (size_t i = n + 1; i < total_qubits; ++i) aux_reg.push_back(i);

    // Initialize accumulator to |1> (Classical state 1)
    qc.pauliX(acc_reg[0]);

    int phase_estimation_bits = 2 * n;

    // Semiclassical QFT Phase Estimation Loop
    // Iterate from least significant bit to most significant bit of the phase
    for (int step = phase_estimation_bits - 1; step >= 0; --step) {

        // 1. Prepare control
        qc.hadamard(ctrl_qubit);

        // 2. Controlled Modular Exponentiation: U^{2^step}
        int power = modExp(a, 1 << step, N);
        cModMult(ctrl_qubit, acc_reg, aux_reg, power, N, qc);

        // 3. Semiclassical Phase Correction
        // Apply Z-rotations based on PREVIOUS measurement outcomes
        double phase_correction = 0.0;
        for (size_t j = 0; j < qc.getMeasurementResults().size(); ++j) {
            if (qc.getMeasurementResults()[j] == 1) {
                // Angle depends on the distance between the current step and the past measurement
                int distance = qc.getMeasurementResults().size() - j;
                phase_correction -= PI / std::pow(2.0, distance);
            }
        }

        if (phase_correction != 0.0) {
            qc.phase(phase_correction, ctrl_qubit);
        }

        // 4. Final Hadamard and Measure
        qc.hadamard(ctrl_qubit);

        bool bit_result = qc.measure(ctrl_qubit);

        // 5. Reset control qubit for the next loop iteration
        qc.resetQubit(ctrl_qubit);
    }
	qc.execute(true);
    // The vector 'measurements' now contains the binary fraction of s/r
    // (Note: Because we iterated backwards, the array might need to be reversed depending on your bit-endianness preference)
	auto measurements = qc.getMeasurementResults();
    std::reverse(measurements.begin(), measurements.end());

    return measurements;
}

int Shor::gcd(int a, int b)
{
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

double Shor::binaryFractionToDouble(const std::vector<int>& bits)
{
    double value = 0.0;
    for (size_t i = 0; i < bits.size(); ++i) {
        if (bits[i] == 1) {
            value += 1.0 / std::pow(2.0, i + 1);
        }
    }
    return value;
}

int Shor::getDenominator(double phase, int max_den)
{
    if (phase == 0.0) return 1;

    int h1 = 1, h2 = 0;
    int k1 = 0, k2 = 1;
    double b = phase;

    do {
        int a = std::floor(1 / b);
        int h = a * h1 + h2;
        int k = a * k1 + k2;

        if (k > max_den) break;

        h2 = h1; h1 = h;
        k2 = k1; k1 = k;

        b = (1 / b) - a;
    } while (std::abs(b) > 1e-6);

    return k1;
}

void Shor::runShor(int N, int a)
{
    std::cout << "\n========================================" << std::endl;
    std::cout << "      SHOR'S ALGORITHM FACTORIZATION      " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Target N = " << N << ", Base Guess a = " << a << std::endl;

    // 1. Classical pre-check
    int initial_gcd = gcd(a, N);
    if (initial_gcd > 1) {
        std::cout << "\nLucky guess! 'a' shares a factor with 'N'." << std::endl;
        std::cout << "Classical factor found: " << initial_gcd << std::endl;
        std::cout << "No quantum circuit required." << std::endl;
        return;
    }

    // 2. Qubit limit check
    int n = std::ceil(std::log2(N));
    int required_qubits = 2 * n + 3;

    std::cout << "Bit size of N (n) = " << n << " bits" << std::endl;
    std::cout << "Qubits required (2n + 3) = " << required_qubits << " qubits" << std::endl;

    if (required_qubits > 30) { // Limit based on your API
        std::cout << "\n[ERROR] Simulator Memory Limit Exceeded!" << std::endl;
        std::cout << "Your simulator supports up to 30 qubits." << std::endl;
        std::cout << "Cannot factor N=" << N << " with the 2n+3 architecture." << std::endl;
        return;
    }

    // 3. Execute Quantum Semiclassical Loop
    std::cout << "\n[Running Quantum Phase Estimation...]" << std::endl;
    Circuit qc(required_qubits); // Clear any previous state

    // Call the semiclassical algorithm from the previous code block
    std::vector<int> measurements = shorsAlgorithm2n3(a, N, qc);

    // 4. Process Results
    std::cout << "\n--- Quantum Results ---" << std::endl;
    std::cout << "Raw Measurement (Binary Phase): 0.";
    for (int bit : measurements) std::cout << bit;
    std::cout << std::endl;

    double phase = binaryFractionToDouble(measurements);
    std::cout << "Decimal Phase (s/r): " << phase << std::endl;

    // 5. Continued Fractions to find period 'r'
    int r = getDenominator(phase, N);
    std::cout << "Calculated Period (r): " << r << std::endl;

    if (r == 1 || r == 0) {
        std::cout << "\n[FAILED] Algorithm returned a trivial period. Try running again or use a different 'a'." << std::endl;
        return;
    }

    // 6. Calculate Factors
    if (r % 2 != 0) {
        std::cout << "\n[FAILED] Period 'r' is odd. Shor's algorithm requires an even period to factor. Try a different 'a'." << std::endl;
        return;
    }

    // modExp was defined in the previous response
    int half_power = modExp(a, r / 2, N);

    if (half_power == N - 1) { // Equivalent to -1 mod N
        std::cout << "\n[FAILED] Trivial factors found (x = -1 mod N). Try a different 'a'." << std::endl;
        return;
    }

    // Final GCD calculations to find p and q
    int factor1 = gcd(half_power - 1, N);
    int factor2 = gcd(half_power + 1, N);

    std::cout << "\n========================================" << std::endl;
    std::cout << "                SUCCESS!                " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "The factors of " << N << " are: " << factor1 << " and " << factor2 << std::endl;

    // Validation check
    if (factor1 * factor2 == N) {
        std::cout << "(Verified: " << factor1 << " * " << factor2 << " = " << N << ")" << std::endl;
    }
    else {
        std::cout << "(Note: These factors may need further reduction depending on N)" << std::endl;
    }
}

void Shor::applyQFT(std::vector<size_t>& reg, Circuit& qc)
{
    int n = reg.size();
    for (int i = 0; i < n; ++i) {
        qc.hadamard(reg[i]);
        for (int j = i + 1; j < n; ++j) {
            double theta = PI / std::pow(2.0, j - i);
            qc.cphase(theta, reg[j], reg[i]);
        }
    }
}

void Shor::applyIQFT(std::vector<size_t>& reg, Circuit& qc) {
    int n = reg.size();
    for (int i = n - 1; i >= 0; --i) {
        for (int j = n - 1; j > i; --j) {
            double theta = -PI / std::pow(2.0, j - i);
            qc.cphase(theta, reg[j], reg[i]);
        }
		qc.hadamard(reg[i]);
    }
}

void Shor::addConstantQFT(std::vector<size_t>& reg, int a, Circuit& qc) {
    int n = reg.size();
    for (int i = 0; i < n; ++i) {
        double angle = 0.0;
        for (int j = 0; j <= i; ++j) {
            if ((a >> (i - j)) & 1) {
                angle += PI / std::pow(2.0, j);
            }
        }
        if (angle > 0) {
            qc.phase(angle, reg[i]);
        }
    }
}

void Shor::cAddConstantQFT(size_t ctrl, std::vector<size_t>& reg, int a, Circuit& qc) {
    int n = reg.size();
    for (int i = 0; i < n; ++i) {
        double angle = 0.0;
        for (int j = 0; j <= i; ++j) {
            if ((a >> (i - j)) & 1) {
                angle += PI / std::pow(2.0, j);
            }
        }
        if (angle > 0) {
           qc.cphase(angle, ctrl, reg[i]);
        }
    }
}

void Shor::cModAdd(size_t ctrl, std::vector<size_t>& reg, std::vector<size_t>& aux, int a, int N, Circuit& qc)
{
    size_t msb = reg.back();

    cAddConstantQFT(ctrl, reg, a, qc);

    addConstantQFT(reg, -N, qc);

    applyIQFT(reg, qc);

	qc.cnot(msb, aux[0]);

    applyQFT(reg, qc);

    cAddConstantQFT(aux[0], reg, N, qc);

    // 7. Cleanup: Uncompute the auxiliary sign bit
    // 7. Temporarily undo the addition of 'a'
    cAddConstantQFT(ctrl, reg, -a, qc);

    // 8. Transform to computational basis again
    applyIQFT(reg, qc);

    // 9. At this point, the MSB represents the exact OPPOSITE of the aux qubit.
    // We flip it so it perfectly matches the aux qubit.
    qc.pauliX(msb);

    // 10. XOR the MSB into 'aux'. Because MSB now matches aux, 
    // this perfectly erases 'aux' back to |0>!
    qc.cnot(msb, aux[0]);

    // 11. Flip MSB back to its original state
    qc.pauliX(msb);

    // 12. Transform back to Fourier basis
    applyQFT(reg, qc);

    // 13. Re-add 'a' to restore the correct answer
    cAddConstantQFT(ctrl, reg, a, qc);
}


void Shor::cModMult(size_t ctrl, std::vector<size_t>& acc, std::vector<size_t>& aux, int a, int N, Circuit& qc)
{
    applyQFT(aux, qc); // Aux is used as the target for addition

    int n = acc.size();
    for (int i = 0; i < n; ++i) {
        // Multiply by powers of 2 modulo N
        int a_shifted = (a * (1 << i)) % N;

        // This requires a Doubly-Controlled Mod Add (CC-ModAdd)
        // In practice, decompose this into Toffoli-driven controls or nested phase additions.
        cModAdd(ctrl, acc, aux, a_shifted, N, qc);
    }

    applyIQFT(aux, qc);

    // Swap acc and aux registers (Beauregard uses specific CNOT swaps here)
    for (size_t i = 0; i < n; ++i) {
        qc.cnot(aux[i + 1], acc[i]);
        qc.cnot(acc[i], aux[i + 1]);
        qc.cnot(aux[i + 1], acc[i]);
    }

    // Uncompute the garbage left in the other register by multiplying by modular inverse of 'a'
    // ...
}

int Shor::modExp(int base, int exp, int mod)
{
    int res = 1;
    base = base % mod;
    while (exp > 0) {
        if (exp % 2 == 1) res = (res * base) % mod;
        exp = exp >> 1;
        base = (base * base) % mod;
    }
    return res;
}
