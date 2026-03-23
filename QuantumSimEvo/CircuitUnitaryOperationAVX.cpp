#include "CircuitUnitaryOperationAVX.h";
#include <chrono>
#include <iostream>
#include <immintrin.h>

void CircuitUnitaryOperationAVX::applyHadamard(StateVector& sv, size_t target)
{
    auto& amplitudes = sv.data();
    const size_t size = sv.size();

    const double invSqrt2 = 0.7071067811865475244;
    const __m256d vec_s = _mm256_set1_pd(invSqrt2);

    auto start = std::chrono::high_resolution_clock::now();

    if (target == 0) {
        #pragma omp parallel for schedule(static)
        for (std::ptrdiff_t i = 0; i < size; i += 2) {
            __m256d v = _mm256_loadu_pd((double*)&amplitudes[i]);

            __m256d v_low = _mm256_permute2f128_pd(v, v, 0x00); //binary 0000
            __m256d v_high = _mm256_permute2f128_pd(v, v, 0x11);//binary 1100

            __m256d res_sum = _mm256_add_pd(v_low, v_high);
            __m256d res_diff = _mm256_sub_pd(v_low, v_high);

            __m256d result = _mm256_blend_pd(res_sum, res_diff, 0x0C); // 0x0C binary: 1100
            _mm256_storeu_pd((double*)&amplitudes[i], _mm256_mul_pd(result, vec_s));
        }
    }
    else {
        const size_t mask = 1ULL << target;
        size_t num_sections = size >> (target + 1);

        #pragma omp parallel for schedule(static)
        for (std::ptrdiff_t s = 0; s < num_sections; ++s) {
            size_t base = s << (target + 1);
            for (size_t k = 0; k < mask; k += 2) {
                size_t i = base + k;
                size_t j = i + mask;

                __m256d v0 = _mm256_loadu_pd((double*)&amplitudes[i]);
                __m256d v1 = _mm256_loadu_pd((double*)&amplitudes[j]);

                __m256d sum = _mm256_add_pd(v0, v1);
                __m256d diff = _mm256_sub_pd(v0, v1);

                _mm256_storeu_pd((double*)&amplitudes[i], _mm256_mul_pd(sum, vec_s));
                _mm256_storeu_pd((double*)&amplitudes[j], _mm256_mul_pd(diff, vec_s));
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "Hadamard applied in: " << ms.count() << " ms\n";
}

void CircuitUnitaryOperationAVX::applyPauliY(StateVector& sv, size_t target) {
    std::vector<std::complex<double>>& amplitudes = sv.data();
    size_t size = sv.size();

    auto start = std::chrono::high_resolution_clock::now();

    double* amplitudes_ptr = reinterpret_cast<double*>(amplitudes.data());

    if (target == 0) {
        __m256d sign_mask = _mm256_setr_pd(1.0, -1.0, -1.0, 1.0);

        #pragma omp parallel for
        for (std::ptrdiff_t i = 0; i < size; i += 2) {
            __m256d v = _mm256_loadu_pd(&amplitudes_ptr[i * 2]);

            __m256d v_swap_128 = _mm256_permute2f128_pd(v, v, 0x01);

            __m256d v_swap_64 = _mm256_permute_pd(v_swap_128, 0x05);

            __m256d v_out = _mm256_mul_pd(v_swap_64, sign_mask);

            _mm256_storeu_pd(&amplitudes_ptr[i * 2], v_out);
        }
    }
    else {
        size_t half_step = 1ULL << target;
        size_t mask_low = half_step - 1;

        __m256d sign_p0 = _mm256_setr_pd(1.0, -1.0, 1.0, -1.0);

        __m256d sign_p1 = _mm256_setr_pd(-1.0, 1.0, -1.0, 1.0);

        #pragma omp parallel for
        for (std::ptrdiff_t k = 0; k < size / 2; k += 2) {
            size_t idx0 = ((k >> target) << (target + 1)) | (k & mask_low);
            size_t idx1 = idx0 | half_step;

            double* p0 = &amplitudes_ptr[idx0 * 2];
            double* p1 = &amplitudes_ptr[idx1 * 2];

            __m256d v0 = _mm256_loadu_pd(p0);
            __m256d v1 = _mm256_loadu_pd(p1);

            __m256d v1_swapped = _mm256_permute_pd(v1, 0x05);
            __m256d p0_new = _mm256_mul_pd(v1_swapped, sign_p0);

            __m256d v0_swapped = _mm256_permute_pd(v0, 0x05);
            __m256d p1_new = _mm256_mul_pd(v0_swapped, sign_p1);

            _mm256_storeu_pd(p0, p0_new);
            _mm256_storeu_pd(p1, p1_new);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "PauliY applied in: " << ms.count() << " ms \n";
}

void CircuitUnitaryOperationAVX::applyPauliZ(StateVector& sv, size_t q) {
    std::vector<std::complex<double>>& amplitudes = sv.data();
    size_t size = sv.size();
    size_t mask = 1ULL << q;
    __m256d sign_mask = _mm256_set1_pd(-0.0);

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < size; i += 2) {
        if (i & mask) {
            __m256d data = _mm256_loadu_pd((double*)&amplitudes[i]);

            data = _mm256_xor_pd(data, sign_mask);
            _mm256_storeu_pd((double*)&amplitudes[i], data);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "PauliZ applied in: " << ms.count() << " ms \n";
}


void CircuitUnitaryOperationAVX::applyRotateX(StateVector& sv, size_t target, double theta) {
    std::vector<Complex>& amplitudes = sv.data();
    double* data = reinterpret_cast<double*>(amplitudes.data());
    size_t size = amplitudes.size();

    double c = std::cos(theta / 2.0);
    double s = std::sin(theta / 2.0);

    auto start = std::chrono::high_resolution_clock::now();

    __m256d vec_c = _mm256_set1_pd(c);

    __m256d vec_s_pos_neg = _mm256_setr_pd(s, -s, s, -s);

    if (target == 0) {
        #pragma omp parallel for
        for (std::ptrdiff_t i = 0; i < size; i += 2) {
            __m256d v = _mm256_loadu_pd(&data[i * 2]);

            __m256d v_c = _mm256_mul_pd(v, vec_c);

            __m256d v_swap_pairs = _mm256_permute2f128_pd(v, v, 0x01);

            __m256d v_swap_re_im = _mm256_permute_pd(v_swap_pairs, 0x05);

            __m256d v_s = _mm256_mul_pd(v_swap_re_im, vec_s_pos_neg);

            __m256d v_out = _mm256_add_pd(v_c, v_s);

            _mm256_storeu_pd(&data[i * 2], v_out);
        }
    }
    else {
        size_t half_step = 1ULL << target;
        size_t mask_low = half_step - 1;

        #pragma omp parallel for
        for (std::ptrdiff_t k = 0; k < size / 2; k += 2) {
            size_t idx0 = ((k >> target) << (target + 1)) | (k & mask_low);
            size_t idx1 = idx0 | half_step;

            __m256d v0 = _mm256_loadu_pd((double*)&data[idx0 * 2]);
            __m256d v1 = _mm256_loadu_pd((double*)&data[idx1 * 2]);

            __m256d v0_c = _mm256_mul_pd(v0, vec_c);
            __m256d v1_swap = _mm256_permute_pd(v1, 0x05);
            __m256d v1_s = _mm256_mul_pd(v1_swap, vec_s_pos_neg);
            __m256d psi0_new = _mm256_add_pd(v0_c, v1_s);

            __m256d v1_c = _mm256_mul_pd(v1, vec_c);
            __m256d v0_swap = _mm256_permute_pd(v0, 0x05);
            __m256d v0_s = _mm256_mul_pd(v0_swap, vec_s_pos_neg);
            __m256d psi1_new = _mm256_add_pd(v1_c, v0_s);

            _mm256_storeu_pd((double*)&data[idx0 * 2], psi0_new);
            _mm256_storeu_pd((double*)&data[idx1 * 2], psi1_new);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "RotateX applied in: " << ms.count() << " ms \n";
}

void CircuitUnitaryOperationAVX::applyRotateY(StateVector& sv, size_t target, double theta) {
    std::vector<Complex>& amplitudes = sv.data();
    double* amplitudes_ptr = reinterpret_cast<double*>(amplitudes.data());
    size_t size = amplitudes.size();
    if (size < 2) return;

    double c = std::cos(theta / 2.0);
    double s = std::sin(theta / 2.0);

    auto start = std::chrono::high_resolution_clock::now();

    if (target == 0) {
        __m256d vec_c = _mm256_set1_pd(c);
        __m256d vec_s_mix = _mm256_setr_pd(-s, -s, s, s);

        #pragma omp parallel for
        for (long long i = 0; i < (long long)size; i += 2) {
            __m256d v = _mm256_loadu_pd(&amplitudes_ptr[i * 2]);

            __m256d v_swp = _mm256_permute2f128_pd(v, v, 0x01);

            __m256d term1 = _mm256_mul_pd(v, vec_c);
            __m256d term2 = _mm256_mul_pd(v_swp, vec_s_mix);

            _mm256_storeu_pd(&amplitudes_ptr[i * 2], _mm256_add_pd(term1, term2));
        }
    }
    else {
        size_t half_step = 1ULL << target;
        size_t mask_low = half_step - 1;

        __m256d vec_c = _mm256_set1_pd(c);
        __m256d vec_s = _mm256_set1_pd(s);

        #pragma omp parallel for
        for (long long k = 0; k < (long long)size / 2; k += 2) {
            size_t idx0 = ((k >> target) << (target + 1)) | (k & mask_low);
            size_t idx1 = idx0 | half_step;

            __m256d v0 = _mm256_loadu_pd(&amplitudes_ptr[idx0 * 2]);
            __m256d v1 = _mm256_loadu_pd(&amplitudes_ptr[idx1 * 2]);

            __m256d res0 = _mm256_sub_pd(_mm256_mul_pd(vec_c, v0), _mm256_mul_pd(vec_s, v1));

            __m256d res1 = _mm256_add_pd(_mm256_mul_pd(vec_s, v0), _mm256_mul_pd(vec_c, v1));

            _mm256_storeu_pd(&amplitudes_ptr[idx0 * 2], res0);
            _mm256_storeu_pd(&amplitudes_ptr[idx1 * 2], res1);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "RotateY applied in: " << ms.count() << " ms \n";
}

void CircuitUnitaryOperationAVX::applyPhase(StateVector& sv, size_t q, double theta) {
    std::vector<Complex>& amplitudes = sv.data();
    double* amplitudes_ptr = reinterpret_cast<double*>(amplitudes.data());
	size_t size = sv.size();

    double c = std::cos(theta);
    double s = std::sin(theta);
    __m256d vec_c = _mm256_set1_pd(c);
    __m256d vec_s = _mm256_setr_pd(-s, s, -s, s);


    auto start = std::chrono::high_resolution_clock::now();


    if (q == 0) {
        #pragma omp parallel for
        for (std::ptrdiff_t i = 0; i < size; i += 2) {
            __m256d v = _mm256_loadu_pd(&amplitudes_ptr[i * 2]);
            __m256d v_swp = _mm256_permute_pd(v, 0x05);
            __m256d v_phased = _mm256_add_pd(_mm256_mul_pd(v, vec_c), _mm256_mul_pd(v_swp, vec_s));
            _mm256_storeu_pd(&amplitudes_ptr[i * 2], _mm256_blend_pd(v, v_phased, 0x0C));
        }
    }
    else {
        size_t half_step = 1ULL << q;
        size_t mask_low = half_step - 1;
        #pragma omp parallel for
        for (std::ptrdiff_t k = 0; k < size / 2; k += 2) {
            size_t idx1 = (((k >> q) << (q + 1)) | (k & mask_low)) | half_step;
            __m256d v = _mm256_loadu_pd(&amplitudes_ptr[idx1 * 2]);
            __m256d v_swp = _mm256_permute_pd(v, 0x05);
            __m256d res = _mm256_add_pd(_mm256_mul_pd(v, vec_c), _mm256_mul_pd(v_swp, vec_s));
            _mm256_storeu_pd(&amplitudes_ptr[idx1 * 2], res);
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "Phase applied in: " << ms.count() << " ms \n"; //Debug timing
}

void CircuitUnitaryOperationAVX::applyRotateZ(StateVector& sv, size_t target, double theta) {
    std::vector<Complex>& amplitudes = sv.data();
    double* amplitudes_ptr = reinterpret_cast<double*>(amplitudes.data());
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
        for (std::ptrdiff_t i = 0; i < size; i += 2) {
            __m256d v = _mm256_loadu_pd(&amplitudes_ptr[i * 2]);
            __m256d v_swp = _mm256_permute_pd(v, 0x05);
            __m256d res = _mm256_add_pd(_mm256_mul_pd(v, vec_c), _mm256_mul_pd(v_swp, vec_s_mixed));
            _mm256_storeu_pd(&amplitudes_ptr[i * 2], res);
        }
    }
    else {
        size_t half_step = 1ULL << target;
        size_t mask_low = half_step - 1;
        #pragma omp parallel for
        for (std::ptrdiff_t k = 0; k < size / 2; k += 2) {
            size_t idx0 = ((k >> target) << (target + 1)) | (k & mask_low);
            size_t idx1 = idx0 | half_step;

            __m256d v0 = _mm256_loadu_pd(&amplitudes_ptr[idx0 * 2]);
            __m256d v1 = _mm256_loadu_pd(&amplitudes_ptr[idx1 * 2]);

            __m256d res0 = _mm256_add_pd(_mm256_mul_pd(v0, vec_c), _mm256_mul_pd(_mm256_permute_pd(v0, 0x05), vec_s0));
            __m256d res1 = _mm256_add_pd(_mm256_mul_pd(v1, vec_c), _mm256_mul_pd(_mm256_permute_pd(v1, 0x05), vec_s1));

            _mm256_storeu_pd(&amplitudes_ptr[idx0 * 2], res0);
            _mm256_storeu_pd(&amplitudes_ptr[idx1 * 2], res1);
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

void CircuitUnitaryOperationAVX::applyPauliX(StateVector& sv, size_t target) {
    std::vector<Complex>& amplitudes = sv.data();
    double* amplitudes_ptr = reinterpret_cast<double*>(amplitudes.data());
    size_t size = amplitudes.size();
    if (size < 2) return;

    auto start = std::chrono::high_resolution_clock::now();

    if (target == 0) {
        #pragma omp parallel for
        for (std::ptrdiff_t i = 0; i < size; i += 2) {
            __m256d v = _mm256_loadu_pd(&amplitudes_ptr[i * 2]);
            v = _mm256_permute2f128_pd(v, v, 0x01);
            _mm256_storeu_pd(&amplitudes_ptr[i * 2], v);
        }
    }
    else {
        size_t half_step = 1ULL << target;
        size_t mask_low = half_step - 1;
        #pragma omp parallel for
        for (std::ptrdiff_t k = 0; k < size / 2; k += 2) {
            size_t idx0 = ((k >> target) << (target + 1)) | (k & mask_low);
            size_t idx1 = idx0 | half_step;

            __m256d v0 = _mm256_loadu_pd(&amplitudes_ptr[idx0 * 2]);
            __m256d v1 = _mm256_loadu_pd(&amplitudes_ptr[idx1 * 2]);

            _mm256_storeu_pd(&amplitudes_ptr[idx0 * 2], v1);
            _mm256_storeu_pd(&amplitudes_ptr[idx1 * 2], v0);
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "PauliX applied in: " << ms.count() << " ms \n";
}

//methods bellow shold be changed

void CircuitUnitaryOperationAVX::applyGate(StateVector& sv, const Gate2x2& gate, size_t q) {
    std::vector<std::complex<double>>& states = sv.data();
    size_t N = sv.qubits();
    size_t size = sv.size();

    size_t sectionSize = 1ULL << q;

    auto start = std::chrono::high_resolution_clock::now();

#pragma omp parallel for
    for (std::ptrdiff_t i = 0; i < (size >> 1); ++i) {
        size_t i0 = ((i >> q) << (q + 1)) | (i & (sectionSize - 1));
        size_t i1 = i0 | sectionSize;

        std::complex<double> low = states[i0];
        std::complex<double> high = states[i1];

        states[i0] = gate.data[0][0] * low + gate.data[0][1] * high;
        states[i1] = gate.data[1][0] * low + gate.data[1][1] * high;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    std::cout << "Gate applied in: " << ms.count() << " ms \n"; //Debug timing
}