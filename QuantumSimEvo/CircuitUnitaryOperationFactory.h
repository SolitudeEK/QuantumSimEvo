#pragma once
#include "ICircuitUnitaryOperation.h"

class CircuitUnitaryOperationFactory{
public:
	static std::unique_ptr <ICircuitUnitaryOperation> create(size_t numQubits);
private :
	static bool supportsAVX2();
	static bool supportsCUDA(size_t numQubits);
};