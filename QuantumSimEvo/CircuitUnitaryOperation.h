#ifndef CIRCUIT_UNITARY_OPERATION_H
#define CIRCUIT_UNITARY_OPERATION_H
#include "StateVector.h"


class CircuitUnitaryOperation {
public:
    static void applyGate(StateVector& sv, const Gate2x2& gate, size_t q);
    static void applyCNOT(StateVector& sv, size_t control, size_t target);
    static void applyPauliX(StateVector& sv, size_t q);
    static void applyPauliY(StateVector& sv, size_t q);
    static void applyPauliZ(StateVector& sv, size_t q);
    static void applyPauliZAVX(StateVector& sv, size_t q);
    static void applyPhase(StateVector& sv, size_t q, double theta);
	static void applyRotateX(StateVector& sv, size_t q, double theta);
    static void applyRotateY(StateVector& sv, size_t q, double theta);
    static void applyRotateYAVX(StateVector& sv, size_t q, double theta);
    static void applyRotateZ(StateVector& sv, size_t q, double theta);
    static void applyHadamard(StateVector& sv, size_t targetQubit); 
    static void applyHadamardAVX(StateVector& sv, size_t targetQubit);
};

#endif