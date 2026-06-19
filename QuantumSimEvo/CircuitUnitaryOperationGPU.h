#ifndef CIRCUIT_UNITARY_OPERATION_GPU_H
#define CIRCUIT_UNITARY_OPERATION_GPU_H

#include "ICircuitUnitaryOperation.h"
#include <vector>
#include <cstddef>
#include <cuda_runtime.h>
#include <cuComplex.h>

class CircuitUnitaryOperationGPU : public ICircuitUnitaryOperation {
public:
    CircuitUnitaryOperationGPU();
    ~CircuitUnitaryOperationGPU() override;

    void applyGate(StateVector& sv, const Gate2x2& gate, size_t q) override;
    void applyCNOT(StateVector& sv, size_t control, size_t target) override;
    void applyPauliX(StateVector& sv, size_t q) override;
    void applyPauliY(StateVector& sv, size_t q) override;
    void applyPauliZ(StateVector& sv, size_t q) override;
    void applyPhase(StateVector& sv, size_t q, double theta) override;
    void applyRotateX(StateVector& sv, size_t q, double theta) override;
    void applyRotateY(StateVector& sv, size_t q, double theta) override;
    void applyRotateZ(StateVector& sv, size_t q, double theta) override;
    void applyHadamard(StateVector& sv, size_t targetQubit) override;
    void applyCPhase(StateVector& sv, size_t control, size_t target, double theta) override;
    void applyMCPhase(StateVector& sv, const std::vector<size_t>& controls, size_t target, double theta) override;
    void applySwap(StateVector& sv, size_t q0, size_t q1) override;

    // Copy device state back into the bound StateVector's host buffer.
    void flush() override;
    // Mark the host buffer as the authoritative copy; next gate will re-upload.
    void markHostDirty() override;

private:
    void ensureOnDevice(StateVector& sv);
    void timedLaunchEnd(const char* label);
    void timedLaunchStart();

    cuDoubleComplex* d_state;
    size_t           d_size;       // number of amplitudes
    StateVector*     bound;        // host owner currently mirrored on device
    bool             hostDirty;    // host buffer changed since last upload
    cudaStream_t     stream;

    // event handles for timing (void* to avoid including cuda_runtime.h in this header)
    void* evStart;
    void* evStop;
};

#endif
