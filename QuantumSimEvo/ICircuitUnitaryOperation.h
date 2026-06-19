#ifndef ICIRCUIT_UNITARY_OPERATION_H
#define ICIRCUIT_UNITARY_OPERATION_H
#include "StateVector.h"
#include <vector>

class ICircuitUnitaryOperation {
public:
    virtual ~ICircuitUnitaryOperation() = default;

    void setPrintSteps(bool value) { print_steps = value; }

    // Backends that mirror state on a device (e.g. GPU) override these.
    // flush(): copy device -> host so the host StateVector is current.
    // markHostDirty(): the host StateVector was mutated externally;
    //                  re-upload to device before the next gate.
    virtual void flush() {}
    virtual void markHostDirty() {}

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
	virtual void applyCPhase(StateVector& sv, size_t control, size_t target, double theta) = 0;
	virtual void applyMCPhase(StateVector& sv, const std::vector<size_t>& controls, size_t target, double theta) = 0;
	virtual void applySwap(StateVector& sv, size_t q0, size_t q1) = 0;

protected:
    bool print_steps = false;
};
#endif
