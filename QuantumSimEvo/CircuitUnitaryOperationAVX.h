#ifndef CIRCUIT_UNITARY_OPERATION_AVX_H
#define CIRCUIT_UNITARY_OPERATION_AVX_H
#include "StateVector.h"
#include "ICircuitUnitaryOperation.h"
#include <vector>

class CircuitUnitaryOperationAVX : public ICircuitUnitaryOperation {
public:
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
};
#endif