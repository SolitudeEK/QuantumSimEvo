#ifndef ICIRCUIT_UNITARY_OPERATION_H
#define ICIRCUIT_UNITARY_OPERATION_H
#include "StateVector.h"

class ICircuitUnitaryOperation {
public:
	virtual void applyGate(StateVector& sv, const Gate2x2& gate, size_t q) = 0;
	virtual void applyCNOT(StateVector& sv, size_t control, size_t target) = 0;
	virtual void applyPauliX(StateVector& sv, size_t q) = 0;
	virtual void applyPauliY(StateVector& sv, size_t q) = 0;
	virtual void applyPauliZ(StateVector& sv, size_t q) = 0;
	virtual void applyPhase(StateVector& sv, size_t q, double theta) = 0;
	virtual void applyRotateX(StateVector& sv, size_t q, double theta) = 0;
	virtual void applyRotateY(StateVector& sv, size_t q, double theta) = 0;
	virtual void applyRotateZ(StateVector& sv, size_t q, double theta) = 0;
	virtual void applyHadamard(StateVector& sv, size_t targetQubit) = 0;
};
#endif
