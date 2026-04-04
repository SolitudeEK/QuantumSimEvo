#include "CircuitUnitaryOperationFactory.h"
#include "CircuitUnitaryOperationAVX.h"
#include "CircuitUnitaryOperation.h"
#include <immintrin.h>
#include <iostream>

std::unique_ptr<ICircuitUnitaryOperation> CircuitUnitaryOperationFactory::create() {
	if (supportsAVX2())
		return std::make_unique<CircuitUnitaryOperationAVX>();
	else
		return std::make_unique <CircuitUnitaryOperation>();
}

bool CircuitUnitaryOperationFactory::supportsAVX2() {
#if defined(_MSC_VER) || defined(__INTEL_COMPILER) || defined(__INTEL_LLVM_COMPILER)
    int info[4];
    // Check for AVX2 support (Leaf 7, Subleaf 0)
    __cpuidex(info, 7, 0);
    bool isSupported = (info[1] & (1 << 5)) != 0;

    std::cout << "AVX2 support (via CPUID): " << (isSupported ? "Yes" : "No") << std::endl;
    return isSupported;
#else
    // Fallback for other environments
    std::cerr << "AVX2 detection not supported on this platform." << std::endl;
    return false;
#endif
}