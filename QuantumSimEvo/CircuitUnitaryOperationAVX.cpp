#include "CircuitUnitaryOperationAVX.h";
#include <chrono>
#include <iostream>
#include <immintrin.h>

void CircuitUnitaryOperationAVX::applyHadamard(StateVector& sv, size_t targetQubit)
{
    auto& state = sv.data();
    const size_t size = sv.size();

    const double invSqrt2 = 0.7071067811865475244;
    const __m256d vec_s = _mm256_set1_pd(invSqrt2);

    auto start = std::chrono::high_resolution_clock::now();

    if (targetQubit >= 1) {

        const size_t mask = 1ULL << targetQubit;
        size_t num_sections = size >> (targetQubit + 1);

#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t s = 0; s < num_sections; ++s) {
            size_t base = s << (targetQubit + 1);
            for (size_t k = 0; k < mask; k += 2) {
                size_t i = base + k;
                size_t j = i + mask;

                __m256d v0 = _mm256_loadu_pd((double*)&state[i]);
                __m256d v1 = _mm256_loadu_pd((double*)&state[j]);

                __m256d sum = _mm256_add_pd(v0, v1);
                __m256d diff = _mm256_sub_pd(v0, v1);

                _mm256_storeu_pd((double*)&state[i], _mm256_mul_pd(sum, vec_s));
                _mm256_storeu_pd((double*)&state[j], _mm256_mul_pd(diff, vec_s));
            }
        }
    }
    else {
#pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i = 0; i < size; i += 2) {
            __m256d v = _mm256_loadu_pd((double*)&state[i]);

            // v_low  = [R0, I0, R0, I0]
            // v_high = [R1, I1, R1, I1]
            __m256d v_low = _mm256_permute2f128_pd(v, v, 0x00);
            __m256d v_high = _mm256_permute2f128_pd(v, v, 0x11);

            __m256d res_sum = _mm256_add_pd(v_low, v_high); // [R0+R1, I0+I1, R0+R1, I0+I1]
            __m256d res_diff = _mm256_sub_pd(v_low, v_high); // [R0-R1, I0-I1, R0-R1, I0-I1]

            // Combine them: [ (R0+R1)s, (I0+I1)s, (R0-R1)s, (I0-I1)s ]
            __m256d result = _mm256_blend_pd(res_sum, res_diff, 0x0C); // 0x0C binary: 1100
            _mm256_storeu_pd((double*)&state[i], _mm256_mul_pd(result, vec_s));
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "Hadamard applied in: " << ms.count() << " ms\n";
}

void CircuitUnitaryOperationAVX::applyPauliY(StateVector& sv, size_t q) {
    std::vector<std::complex<double>>& states = sv.data();
    size_t size = states.size();

    auto start = std::chrono::high_resolution_clock::now();

    // Reinterpret as flat double array: [R0, I0, R1, I1, ...]
    double* data_ptr = reinterpret_cast<double*>(states.data());

    if (q == 0) {
        // q == 0: Pairs are in the same AVX register [R0, I0, R1, I1]
        // We want to transform this to: [I1, -R1, -I0, R0]

        // Sign mask: [1.0, -1.0, -1.0, 1.0]
        __m256d sign_mask = _mm256_setr_pd(1.0, -1.0, -1.0, 1.0);

#pragma omp parallel for
        for (std::ptrdiff_t i = 0; i < size; i += 2) {
            __m256d v = _mm256_loadu_pd(&data_ptr[i * 2]);

            // 1. Swap the two complex numbers: [R1, I1, R0, I0]
            __m256d v_swap_128 = _mm256_permute2f128_pd(v, v, 0x01);

            // 2. Swap Real and Imaginary parts: [I1, R1, I0, R0]
            __m256d v_swap_64 = _mm256_permute_pd(v_swap_128, 0x05);

            // 3. Apply signs: [I1, -R1, -I0, R0]
            __m256d v_out = _mm256_mul_pd(v_swap_64, sign_mask);

            _mm256_storeu_pd(&data_ptr[i * 2], v_out);
        }
    }
    else {
        // q >= 1: Pairs are separated in memory
        size_t half_step = 1ULL << q;
        size_t mask_low = half_step - 1;

        // Sign masks for separated registers
        // For psi0_new (comes from v1): [1.0, -1.0, 1.0, -1.0]
        __m256d sign_p0 = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);

        // For psi1_new (comes from v0): [-1.0, 1.0, -1.0, 1.0]
        __m256d sign_p1 = _mm256_setr_pd(-1.0, 1.0, -1.0, 1.0);

#pragma omp parallel for
        for (std::ptrdiff_t k = 0; k < size / 2; k += 2) {
            // Flattened bitwise indexing to avoid branching and cache misses
            size_t idx0 = ((k >> q) << (q + 1)) | (k & mask_low);
            size_t idx1 = idx0 | half_step;

            double* p0 = &data_ptr[idx0 * 2];
            double* p1 = &data_ptr[idx1 * 2];

            __m256d v0 = _mm256_loadu_pd(p0);
            __m256d v1 = _mm256_loadu_pd(p1);

            // Process new p0 (which takes the values of v1, swapped and signed)
            __m256d v1_swapped = _mm256_permute_pd(v1, 0x05); // [I1_a, R1_a, I1_b, R1_b]
            __m256d p0_new = _mm256_mul_pd(v1_swapped, sign_p0); // [I1_a, -R1_a, I1_b, -R1_b]

            // Process new p1 (which takes the values of v0, swapped and signed)
            __m256d v0_swapped = _mm256_permute_pd(v0, 0x05); // [I0_a, R0_a, I0_b, R0_b]
            __m256d p1_new = _mm256_mul_pd(v0_swapped, sign_p1); // [-I0_a, R0_a, -I0_b, R0_b]

            _mm256_storeu_pd(p0, p0_new);
            _mm256_storeu_pd(p1, p1_new);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "PauliY applied in: " << ms.count() << " ms \n";
}

void CircuitUnitaryOperationAVX::applyPauliZ(StateVector& sv, size_t q) {
    std::vector<std::complex<double>>& states = sv.data();
    size_t size = sv.size();
    size_t mask = 1ULL << q;
    __m256d sign_mask = _mm256_set1_pd(-0.0);

    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < size; i += 2) {
        if (i & mask) {
            __m256d data = _mm256_loadu_pd((double*)&states[i]);

            data = _mm256_xor_pd(data, sign_mask);
            _mm256_storeu_pd((double*)&states[i], data);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "PauliZ applied in: " << ms.count() << " ms \n"; //Debug timing
}


void CircuitUnitaryOperationAVX::applyRotateX(StateVector& sv, size_t target, double theta) {
    std::vector<Complex>& states = sv.data();
    size_t size = states.size();

    double c = std::cos(theta / 2.0);
    double s = std::sin(theta / 2.0);

    auto start = std::chrono::high_resolution_clock::now();

    // Reinterpret as a flat array of doubles: [Re0, Im0, Re1, Im1, ...]
    double* data_ptr = reinterpret_cast<double*>(states.data());

    // Pre-load constants into AVX registers
    __m256d vec_c = _mm256_set1_pd(c);

    // This exact sequence [s, -s, s, -s] cleverly handles the multiplication by -i 
    // when combined with a Real/Imag swap, skipping actual complex multiplication.
    __m256d vec_s_pos_neg = _mm256_setr_pd(s, -s, s, -s);

    if (target == 0) {
        // Special case: target == 0. Paired amplitudes are adjacent in memory.
        #pragma omp parallel for
        for (std::ptrdiff_t i = 0; i < size; i += 2) {
            // Load 2 complex numbers [R0, I0, R1, I1]
            __m256d v = _mm256_loadu_pd(&data_ptr[i * 2]);

            // Multiply by cosine
            __m256d v_c = _mm256_mul_pd(v, vec_c);

            // Cross the complex numbers: [R1, I1, R0, I0]
            __m256d v_swap_pairs = _mm256_permute2f128_pd(v, v, 0x01);

            // Swap Real/Imag parts: [I1, R1, I0, R0]
            __m256d v_swap_re_im = _mm256_permute_pd(v_swap_pairs, 0x05);

            // Multiply by [s, -s, s, -s] to get [s I1, -s R1, s I0, -s R0]
            __m256d v_s = _mm256_mul_pd(v_swap_re_im, vec_s_pos_neg);

            // Add back together
            __m256d v_out = _mm256_add_pd(v_c, v_s);

            // Store back to memory
            _mm256_storeu_pd(&data_ptr[i * 2], v_out);
        }
    }
    else {
        // target >= 1. Paired amplitudes are at least 1 AVX register apart.
        size_t half_step = 1ULL << target;
        size_t mask_low = half_step - 1;

        // Flattened loop avoids skipping iterations and perfectly distributes over OpenMP threads
        #pragma omp parallel for
        for (std::ptrdiff_t k = 0; k < size / 2; k += 2) {

            // Bitwise math correctly maps the sequential loop index `k` to the jumping 
            // memory layout, avoiding expensive modulus/division operations.
            size_t idx0 = ((k >> target) << (target + 1)) | (k & mask_low);
            size_t idx1 = idx0 | half_step;

            double* p0 = &data_ptr[idx0 * 2];
            double* p1 = &data_ptr[idx1 * 2];

            __m256d v0 = _mm256_loadu_pd(p0);
            __m256d v1 = _mm256_loadu_pd(p1);

            // Calculate psi0_new = c * v0 - i * s * v1
            __m256d v0_c = _mm256_mul_pd(v0, vec_c);
            __m256d v1_swap = _mm256_permute_pd(v1, 0x05);         // Swap Real/Imag
            __m256d v1_s = _mm256_mul_pd(v1_swap, vec_s_pos_neg);  // Apply sine and -i signs
            __m256d psi0_new = _mm256_add_pd(v0_c, v1_s);

            // Calculate psi1_new = -i * s * v0 + c * v1
            __m256d v1_c = _mm256_mul_pd(v1, vec_c);
            __m256d v0_swap = _mm256_permute_pd(v0, 0x05);
            __m256d v0_s = _mm256_mul_pd(v0_swap, vec_s_pos_neg);
            __m256d psi1_new = _mm256_add_pd(v1_c, v0_s);

            _mm256_storeu_pd(p0, psi0_new);
            _mm256_storeu_pd(p1, psi1_new);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "RotateX applied in: " << ms.count() << " ms \n";
}

void CircuitUnitaryOperationAVX::applyRotateY(StateVector& sv, size_t target, double theta) {
    auto& states = sv.data();
    size_t size = sv.size();
    uint64_t step = 1ULL << target;

    double c = std::cos(theta / 2.0);
    double s = std::sin(theta / 2.0);
    __m256d vec_c = _mm256_set1_pd(c);
    __m256d vec_s = _mm256_set1_pd(s);

    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < size; i += 2 * step) {
        for (uint64_t j = i; j < i + step; j += 2) {
            __m256d v0 = _mm256_loadu_pd((double*)&states[j]);
            __m256d v1 = _mm256_loadu_pd((double*)&states[j + step]);

            __m256d res0 = _mm256_sub_pd(_mm256_mul_pd(vec_c, v0), _mm256_mul_pd(vec_s, v1));
            __m256d res1 = _mm256_add_pd(_mm256_mul_pd(vec_s, v0), _mm256_mul_pd(vec_c, v1));

            _mm256_storeu_pd((double*)&states[j], res0);
            _mm256_storeu_pd((double*)&states[j + step], res1);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "RotateY applied in: " << ms.count() << " ms \n"; //Debug timing
}

void CircuitUnitaryOperationAVX::applyPhase(StateVector& sv, size_t q, double theta) {
    std::vector<Complex>& states_vec = sv.data();
    double* data = reinterpret_cast<double*>(states_vec.data());
	size_t size = sv.size();

    double c = std::cos(theta);
    double s = std::sin(theta);
    __m256d vec_c = _mm256_set1_pd(c);
    // Sign vector for (Rc - Is) + i(Ic + Rs)
    __m256d vec_s = _mm256_setr_pd(-s, s, -s, s);


    auto start = std::chrono::high_resolution_clock::now();


    if (q == 0) {
#pragma omp parallel for
        for (long long i = 0; i < (long long)size; i += 2) {
            __m256d v = _mm256_loadu_pd(&data[i * 2]);
            __m256d v_swp = _mm256_permute_pd(v, 0x05);
            __m256d v_phased = _mm256_add_pd(_mm256_mul_pd(v, vec_c), _mm256_mul_pd(v_swp, vec_s));
            // Blend: Keep C0, take phased C1 (mask 0x0C = 1100 binary)
            _mm256_storeu_pd(&data[i * 2], _mm256_blend_pd(v, v_phased, 0x0C));
        }
    }
    else {
        size_t half_step = 1ULL << q;
        size_t mask_low = half_step - 1;
#pragma omp parallel for
        for (long long k = 0; k < (long long)size / 2; k += 2) {
            size_t idx1 = (((k >> q) << (q + 1)) | (k & mask_low)) | half_step;
            __m256d v1 = _mm256_loadu_pd(&data[idx1 * 2]);
            __m256d v1_swp = _mm256_permute_pd(v1, 0x05);
            __m256d res = _mm256_add_pd(_mm256_mul_pd(v1, vec_c), _mm256_mul_pd(v1_swp, vec_s));
            _mm256_storeu_pd(&data[idx1 * 2], res);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "Phase applied in: " << ms.count() << " ms \n"; //Debug timing
}

void CircuitUnitaryOperationAVX::applyRotateZ(StateVector& sv, size_t target, double theta) {
    std::vector<Complex>& states_vec = sv.data();
    double* data = reinterpret_cast<double*>(states_vec.data());
    size_t size = sv.size();

    double c = std::cos(theta / 2.0);
    double s = std::sin(theta / 2.0);
    __m256d vec_c = _mm256_set1_pd(c);
    __m256d vec_s0 = _mm256_setr_pd(s, -s, s, -s);  // For e^{-i*theta/2}
    __m256d vec_s1 = _mm256_setr_pd(-s, s, -s, s); // For e^{i*theta/2}

    auto start = std::chrono::high_resolution_clock::now();

    if (target == 0) {
        __m256d vec_s_mixed = _mm256_setr_pd(s, -s, -s, s);
#pragma omp parallel for
        for (long long i = 0; i < (long long)size; i += 2) {
            __m256d v = _mm256_loadu_pd(&data[i * 2]);
            __m256d v_swp = _mm256_permute_pd(v, 0x05);
            __m256d res = _mm256_add_pd(_mm256_mul_pd(v, vec_c), _mm256_mul_pd(v_swp, vec_s_mixed));
            _mm256_storeu_pd(&data[i * 2], res);
        }
    }
    else {
        size_t half_step = 1ULL << target;
        size_t mask_low = half_step - 1;
#pragma omp parallel for
        for (long long k = 0; k < (long long)size / 2; k += 2) {
            size_t idx0 = ((k >> target) << (target + 1)) | (k & mask_low);
            size_t idx1 = idx0 | half_step;

            __m256d v0 = _mm256_loadu_pd(&data[idx0 * 2]);
            __m256d v1 = _mm256_loadu_pd(&data[idx1 * 2]);

            __m256d res0 = _mm256_add_pd(_mm256_mul_pd(v0, vec_c), _mm256_mul_pd(_mm256_permute_pd(v0, 0x05), vec_s0));
            __m256d res1 = _mm256_add_pd(_mm256_mul_pd(v1, vec_c), _mm256_mul_pd(_mm256_permute_pd(v1, 0x05), vec_s1));

            _mm256_storeu_pd(&data[idx0 * 2], res0);
            _mm256_storeu_pd(&data[idx1 * 2], res1);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "RotateZ applied in: " << ms.count() << " ms \n"; //Debug timing
}

void CircuitUnitaryOperationAVX::applyCNOT(StateVector& sv, size_t control, size_t target) {
    std::vector<std::complex<double>>& states = sv.data();
    size_t size = sv.size();

    // Identify which bit is the smaller/larger jump in memory
    size_t min_q = std::min(control, target);
    size_t max_q = std::max(control, target);

    auto start = std::chrono::high_resolution_clock::now();

    // Loop exactly N/4 times. No branches, no wasted cycles.
    #pragma omp parallel for
    for (std::ptrdiff_t k = 0; k < size / 4; ++k) {

        // 1. Insert a "zero hole" at the min_q position
        size_t i1 = ((k >> min_q) << (min_q + 1)) | (k & ((1ULL << min_q) - 1));

        // 2. Insert a second "zero hole" at the max_q position
        size_t idx0 = ((i1 >> max_q) << (max_q + 1)) | (i1 & ((1ULL << max_q) - 1));

        // 3. Force the control bit to 1 (CNOT only acts when control is 1)
        idx0 |= (1ULL << control);

        // 4. The paired target state is exactly idx0 with the target bit flipped
        size_t idx1 = idx0 | (1ULL << target);

        // Perform the swap
        std::swap(states[idx0], states[idx1]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "CNOT applied in: " << ms.count() << " ms \n"; //Debug timing
}

void CircuitUnitaryOperationAVX::applyPauliX(StateVector& sv, size_t q) {
    std::vector<Complex>& states_vec = sv.data();
    double* data = reinterpret_cast<double*>(states_vec.data());
    size_t size = states_vec.size();
    if (size < 2) return;

    auto start = std::chrono::high_resolution_clock::now();

    if (q == 0) {
#pragma omp parallel for
        for (long long i = 0; i < (long long)size; i += 2) {
            __m256d v = _mm256_loadu_pd(&data[i * 2]);
            // Swaps the two complex numbers: [C0, C1] -> [C1, C0]
            v = _mm256_permute2f128_pd(v, v, 0x01);
            _mm256_storeu_pd(&data[i * 2], v);
        }
    }
    else {
        size_t half_step = 1ULL << q;
        size_t mask_low = half_step - 1;
#pragma omp parallel for
        for (long long k = 0; k < (long long)size / 2; k += 2) {
            size_t idx0 = ((k >> q) << (q + 1)) | (k & mask_low);
            size_t idx1 = idx0 | half_step;

            __m256d v0 = _mm256_loadu_pd(&data[idx0 * 2]);
            __m256d v1 = _mm256_loadu_pd(&data[idx1 * 2]);
            _mm256_storeu_pd(&data[idx0 * 2], v1);
            _mm256_storeu_pd(&data[idx1 * 2], v0);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "PauliX applied in: " << ms.count() << " ms \n"; //Debug timing
}

//methods bellow shold be changed

void CircuitUnitaryOperationAVX::applyGate(StateVector& sv, const Gate2x2& gate, size_t q) {
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