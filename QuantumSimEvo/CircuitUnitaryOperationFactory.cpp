#include "CircuitUnitaryOperationFactory.h"
#include "CircuitUnitaryOperationAVX.h"
#include "CircuitUnitaryOperation.h"
#include <iostream>

std::unique_ptr<ICircuitUnitaryOperation> CircuitUnitaryOperationFactory::create() {
	if (supportsAVX2())
		return std::make_unique<CircuitUnitaryOperationAVX>();
	else
		return std::make_unique <CircuitUnitaryOperation>();
}

bool CircuitUnitaryOperationFactory::supportsAVX2() {

#if defined(__GNUC__) || defined(__clang__)
	bool isSupported = __builtin_cpu_supports("avx2");
    std::cout << "AVX2 support: " << (isSupported ? "Yes" : "No") << std::endl;
    return isSupported;

#elif defined(_MSC_VER)
    int info[4];
    __cpuidex(info, 7, 0);
    bool isSupported = (info[1] & (1 << 5)) != 0;
    std::cout << "AVX2 support: " << (isSupported ? "Yes" : "No") << std::endl;
    return isSupported;

#else
	std::cerr << "AVX2 detection not supported on this platform." << std::endl;
    return false;
#endif
}