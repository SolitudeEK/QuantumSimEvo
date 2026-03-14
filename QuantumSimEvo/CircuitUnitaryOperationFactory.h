#pragma once
#include "ICircuitUnitaryOperation.h"

class CircuitUnitaryOperationFactory{
public:
	static std::unique_ptr <ICircuitUnitaryOperation> create();
private :
	static bool supportsAVX2();
};