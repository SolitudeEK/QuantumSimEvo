#include "CircuitUnitaryOperationGPU.h"
#include "StateVector.h"
#include "Structurs.h"

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <iostream>

// ---------- error checking ----------
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",                \
                         #call, __FILE__, __LINE__, cudaGetErrorString(_e));    \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

static constexpr int BLOCK = 256;

__device__ __forceinline__ cuDoubleComplex cadd(cuDoubleComplex a, cuDoubleComplex b) { return cuCadd(a, b); }
__device__ __forceinline__ cuDoubleComplex cmul(cuDoubleComplex a, cuDoubleComplex b) { return cuCmul(a, b); }
__device__ __forceinline__ cuDoubleComplex cscale(double s, cuDoubleComplex a) { return make_cuDoubleComplex(s * a.x, s * a.y); }
__device__ __forceinline__ cuDoubleComplex cneg(cuDoubleComplex a) { return make_cuDoubleComplex(-a.x, -a.y); }

// ---------- kernels ----------

// Generic 2x2 unitary: covers H, RX, RY, arbitrary single-qubit gate.
__global__ void k_apply2x2(cuDoubleComplex* state,
                           cuDoubleComplex g00, cuDoubleComplex g01,
                           cuDoubleComplex g10, cuDoubleComplex g11,
                           size_t q, size_t halfSize)
{
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= halfSize) return;
    size_t step = 1ULL << q;
    size_t i0 = ((i >> q) << (q + 1)) | (i & (step - 1));
    size_t i1 = i0 | step;
    cuDoubleComplex v0 = state[i0];
    cuDoubleComplex v1 = state[i1];
    state[i0] = cadd(cmul(g00, v0), cmul(g01, v1));
    state[i1] = cadd(cmul(g10, v0), cmul(g11, v1));
}

__global__ void k_pauliX(cuDoubleComplex* state, size_t q, size_t halfSize) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= halfSize) return;
    size_t step = 1ULL << q;
    size_t i0 = ((i >> q) << (q + 1)) | (i & (step - 1));
    size_t i1 = i0 | step;
    cuDoubleComplex t = state[i0];
    state[i0] = state[i1];
    state[i1] = t;
}

__global__ void k_pauliY(cuDoubleComplex* state, size_t q, size_t halfSize) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= halfSize) return;
    size_t step = 1ULL << q;
    size_t i0 = ((i >> q) << (q + 1)) | (i & (step - 1));
    size_t i1 = i0 | step;
    cuDoubleComplex v0 = state[i0];
    cuDoubleComplex v1 = state[i1];
    // state[i0] = -i * v1   = (v1.y, -v1.x)
    // state[i1] =  i * v0   = (-v0.y,  v0.x)
    state[i0] = make_cuDoubleComplex( v1.y, -v1.x);
    state[i1] = make_cuDoubleComplex(-v0.y,  v0.x);
}

__global__ void k_pauliZ(cuDoubleComplex* state, size_t mask, size_t size) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= size) return;
    if (i & mask) state[i] = cneg(state[i]);
}

__global__ void k_phase(cuDoubleComplex* state, size_t mask, cuDoubleComplex phase, size_t size) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= size) return;
    if (i & mask) state[i] = cmul(state[i], phase);
}

// RotateZ: diagonal, two different phases on |0> and |1> of target qubit.
__global__ void k_rotateZ(cuDoubleComplex* state, size_t mask,
                          cuDoubleComplex phase0, cuDoubleComplex phase1, size_t size) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= size) return;
    state[i] = cmul(state[i], (i & mask) ? phase1 : phase0);
}

__global__ void k_cnot(cuDoubleComplex* state, size_t ctrlMask, size_t tgtMask, size_t size) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= size) return;
    if ((i & ctrlMask) && (i < (i ^ tgtMask))) {
        size_t j = i ^ tgtMask;
        cuDoubleComplex t = state[i];
        state[i] = state[j];
        state[j] = t;
    }
}

__global__ void k_cphase(cuDoubleComplex* state, size_t ctrlMask, size_t tgtMask,
                         cuDoubleComplex phase, size_t size) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= size) return;
    if ((i & ctrlMask) && (i & tgtMask)) state[i] = cmul(state[i], phase);
}

__global__ void k_mcphase(cuDoubleComplex* state, size_t allMask,
                          cuDoubleComplex phase, size_t size) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= size) return;
    if ((i & allMask) == allMask) state[i] = cmul(state[i], phase);
}

__global__ void k_swap(cuDoubleComplex* state,
                       size_t mask0, size_t mask1,
                       size_t min_q, size_t max_q, size_t low_mask, size_t quarter)
{
    size_t k = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (k >= quarter) return;
    size_t i = ((k >> min_q) << (min_q + 1)) | (k & low_mask);
    size_t idx = ((i >> max_q) << (max_q + 1)) | (i & ((1ULL << max_q) - 1));
    size_t a = idx | mask0;
    size_t b = idx | mask1;
    cuDoubleComplex t = state[a];
    state[a] = state[b];
    state[b] = t;
}

// ---------- host class ----------

static inline cuDoubleComplex toCu(const Complex& c) {
    return make_cuDoubleComplex(c.real(), c.imag());
}

CircuitUnitaryOperationGPU::CircuitUnitaryOperationGPU()
    : d_state(nullptr), d_size(0), bound(nullptr), hostDirty(false),
      stream(nullptr), evStart(nullptr), evStop(nullptr)
{
    CUDA_CHECK(cudaStreamCreate(&stream));
    cudaEvent_t s, e;
    CUDA_CHECK(cudaEventCreate(&s));
    CUDA_CHECK(cudaEventCreate(&e));
    evStart = s; evStop = e;
}

CircuitUnitaryOperationGPU::~CircuitUnitaryOperationGPU() {
    if (d_state) cudaFree(d_state);
    if (evStart) cudaEventDestroy(static_cast<cudaEvent_t>(evStart));
    if (evStop)  cudaEventDestroy(static_cast<cudaEvent_t>(evStop));
    if (stream)  cudaStreamDestroy(stream);
}

void CircuitUnitaryOperationGPU::ensureOnDevice(StateVector& sv) {
    bool sameBinding = (bound == &sv) && d_state;
    if (sameBinding && !hostDirty) return;

    // If a different state vector was bound, flush it back first.
    if (bound && bound != &sv && d_state) flush();

    size_t n = sv.size();
    if (d_state && d_size != n) {
        cudaFree(d_state);
        d_state = nullptr;
        d_size = 0;
    }
    if (!d_state) {
        CUDA_CHECK(cudaMalloc(&d_state, n * sizeof(cuDoubleComplex)));
        d_size = n;
    }
    CUDA_CHECK(cudaMemcpyAsync(d_state, sv.getRawPtr(),
                               n * sizeof(cuDoubleComplex),
                               cudaMemcpyHostToDevice, stream));
    bound = &sv;
    hostDirty = false;
}

void CircuitUnitaryOperationGPU::flush() {
    if (!bound || !d_state || hostDirty) return;
    CUDA_CHECK(cudaMemcpyAsync(bound->getRawPtr(), d_state,
                               d_size * sizeof(cuDoubleComplex),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

void CircuitUnitaryOperationGPU::markHostDirty() {
    hostDirty = true;
}

void CircuitUnitaryOperationGPU::timedLaunchStart() {
    if (print_steps) CUDA_CHECK(cudaEventRecord(static_cast<cudaEvent_t>(evStart), stream));
}

void CircuitUnitaryOperationGPU::timedLaunchEnd(const char* label) {
    if (!print_steps) return;
    CUDA_CHECK(cudaEventRecord(static_cast<cudaEvent_t>(evStop), stream));
    CUDA_CHECK(cudaEventSynchronize(static_cast<cudaEvent_t>(evStop)));
    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms,
                                    static_cast<cudaEvent_t>(evStart),
                                    static_cast<cudaEvent_t>(evStop)));
    std::cout << label << " applied in: " << ms << " ms (GPU)\n";
}

// helpers for grid sizing
static inline dim3 gridFor(size_t work) {
    return dim3(static_cast<unsigned int>((work + BLOCK - 1) / BLOCK));
}

// ---------- gate implementations ----------

void CircuitUnitaryOperationGPU::applyGate(StateVector& sv, const Gate2x2& g, size_t q) {
    ensureOnDevice(sv);
    size_t half = sv.size() >> 1;
    timedLaunchStart();
    k_apply2x2<<<gridFor(half), BLOCK, 0, stream>>>(
        d_state,
        toCu(g.data[0][0]), toCu(g.data[0][1]),
        toCu(g.data[1][0]), toCu(g.data[1][1]),
        q, half);
    timedLaunchEnd("Gate");
}

void CircuitUnitaryOperationGPU::applyHadamard(StateVector& sv, size_t q) {
    ensureOnDevice(sv);
    const double s = 0.7071067811865475244;
    cuDoubleComplex a = make_cuDoubleComplex( s, 0.0);
    cuDoubleComplex b = make_cuDoubleComplex(-s, 0.0);
    size_t half = sv.size() >> 1;
    timedLaunchStart();
    k_apply2x2<<<gridFor(half), BLOCK, 0, stream>>>(d_state, a, a, a, b, q, half);
    timedLaunchEnd("Hadamard");
}

void CircuitUnitaryOperationGPU::applyPauliX(StateVector& sv, size_t q) {
    ensureOnDevice(sv);
    size_t half = sv.size() >> 1;
    timedLaunchStart();
    k_pauliX<<<gridFor(half), BLOCK, 0, stream>>>(d_state, q, half);
    timedLaunchEnd("PauliX");
}

void CircuitUnitaryOperationGPU::applyPauliY(StateVector& sv, size_t q) {
    ensureOnDevice(sv);
    size_t half = sv.size() >> 1;
    timedLaunchStart();
    k_pauliY<<<gridFor(half), BLOCK, 0, stream>>>(d_state, q, half);
    timedLaunchEnd("PauliY");
}

void CircuitUnitaryOperationGPU::applyPauliZ(StateVector& sv, size_t q) {
    ensureOnDevice(sv);
    size_t n = sv.size();
    timedLaunchStart();
    k_pauliZ<<<gridFor(n), BLOCK, 0, stream>>>(d_state, 1ULL << q, n);
    timedLaunchEnd("PauliZ");
}

void CircuitUnitaryOperationGPU::applyPhase(StateVector& sv, size_t q, double theta) {
    ensureOnDevice(sv);
    size_t n = sv.size();
    cuDoubleComplex phase = make_cuDoubleComplex(std::cos(theta), std::sin(theta));
    timedLaunchStart();
    k_phase<<<gridFor(n), BLOCK, 0, stream>>>(d_state, 1ULL << q, phase, n);
    timedLaunchEnd("Phase");
}

void CircuitUnitaryOperationGPU::applyRotateX(StateVector& sv, size_t q, double theta) {
    ensureOnDevice(sv);
    double c = std::cos(theta / 2.0);
    double s = std::sin(theta / 2.0);
    cuDoubleComplex diag = make_cuDoubleComplex(c, 0.0);
    cuDoubleComplex off  = make_cuDoubleComplex(0.0, -s); // -i sin(theta/2)
    size_t half = sv.size() >> 1;
    timedLaunchStart();
    k_apply2x2<<<gridFor(half), BLOCK, 0, stream>>>(d_state, diag, off, off, diag, q, half);
    timedLaunchEnd("RotateX");
}

void CircuitUnitaryOperationGPU::applyRotateY(StateVector& sv, size_t q, double theta) {
    ensureOnDevice(sv);
    double c = std::cos(theta / 2.0);
    double s = std::sin(theta / 2.0);
    cuDoubleComplex cc = make_cuDoubleComplex(c, 0.0);
    cuDoubleComplex ms = make_cuDoubleComplex(-s, 0.0);
    cuDoubleComplex ps = make_cuDoubleComplex( s, 0.0);
    size_t half = sv.size() >> 1;
    timedLaunchStart();
    k_apply2x2<<<gridFor(half), BLOCK, 0, stream>>>(d_state, cc, ms, ps, cc, q, half);
    timedLaunchEnd("RotateY");
}

void CircuitUnitaryOperationGPU::applyRotateZ(StateVector& sv, size_t q, double theta) {
    ensureOnDevice(sv);
    cuDoubleComplex p0 = make_cuDoubleComplex(std::cos(-theta / 2.0), std::sin(-theta / 2.0));
    cuDoubleComplex p1 = make_cuDoubleComplex(std::cos( theta / 2.0), std::sin( theta / 2.0));
    size_t n = sv.size();
    timedLaunchStart();
    k_rotateZ<<<gridFor(n), BLOCK, 0, stream>>>(d_state, 1ULL << q, p0, p1, n);
    timedLaunchEnd("RotateZ");
}

void CircuitUnitaryOperationGPU::applyCNOT(StateVector& sv, size_t control, size_t target) {
    ensureOnDevice(sv);
    size_t n = sv.size();
    timedLaunchStart();
    k_cnot<<<gridFor(n), BLOCK, 0, stream>>>(d_state, 1ULL << control, 1ULL << target, n);
    timedLaunchEnd("CNOT");
}

void CircuitUnitaryOperationGPU::applyCPhase(StateVector& sv, size_t control, size_t target, double theta) {
    ensureOnDevice(sv);
    size_t n = sv.size();
    cuDoubleComplex phase = make_cuDoubleComplex(std::cos(theta), std::sin(theta));
    timedLaunchStart();
    k_cphase<<<gridFor(n), BLOCK, 0, stream>>>(d_state, 1ULL << control, 1ULL << target, phase, n);
    timedLaunchEnd("CPhase");
}

void CircuitUnitaryOperationGPU::applyMCPhase(StateVector& sv, const std::vector<size_t>& controls,
                                              size_t target, double theta) {
    ensureOnDevice(sv);
    size_t n = sv.size();
    size_t allMask = 1ULL << target;
    for (size_t c : controls) allMask |= (1ULL << c);
    cuDoubleComplex phase = make_cuDoubleComplex(std::cos(theta), std::sin(theta));
    timedLaunchStart();
    k_mcphase<<<gridFor(n), BLOCK, 0, stream>>>(d_state, allMask, phase, n);
    timedLaunchEnd("MCPhase");
}

void CircuitUnitaryOperationGPU::applySwap(StateVector& sv, size_t q0, size_t q1) {
    ensureOnDevice(sv);
    if (q0 == q1) return;
    size_t mask0 = 1ULL << q0;
    size_t mask1 = 1ULL << q1;
    size_t min_q = q0 < q1 ? q0 : q1;
    size_t max_q = q0 < q1 ? q1 : q0;
    size_t low_mask = (1ULL << min_q) - 1;
    size_t quarter = sv.size() >> 2;
    timedLaunchStart();
    k_swap<<<gridFor(quarter), BLOCK, 0, stream>>>(d_state, mask0, mask1,
                                                   min_q, max_q, low_mask, quarter);
    timedLaunchEnd("Swap");
}
