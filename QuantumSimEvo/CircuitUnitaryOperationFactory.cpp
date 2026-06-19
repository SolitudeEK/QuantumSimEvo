#include "CircuitUnitaryOperationFactory.h"
#include "CircuitUnitaryOperationAVX.h"
#include "CircuitUnitaryOperation.h"
#include "CircuitUnitaryOperationGPU.h"
#include <immintrin.h>
#include <iostream>
#include <cuda_runtime.h>

std::unique_ptr<ICircuitUnitaryOperation> CircuitUnitaryOperationFactory::create(size_t numQubits) {
	if (supportsCUDA(numQubits))
		return std::make_unique<CircuitUnitaryOperationGPU>();
	if (supportsAVX2())
		return std::make_unique<CircuitUnitaryOperationAVX>();
	return std::make_unique<CircuitUnitaryOperation>();
}

bool CircuitUnitaryOperationFactory::supportsCUDA(size_t numQubits) {
	int deviceCount = 0;
	cudaError_t err = cudaGetDeviceCount(&deviceCount);
	if (err != cudaSuccess || deviceCount == 0) {
		// Clear the error state so later CUDA calls aren't poisoned.
		cudaGetLastError();
		std::cout << "CUDA support: No (" << (err == cudaSuccess ? "no devices" : cudaGetErrorString(err)) << ")" << std::endl;
		return false;
	}

	const size_t stateBytes = (size_t(1) << numQubits) * sizeof(double) * 2;

	// Reserve a small overhead for CUDA context + workspace; require state + 64 MiB.
	const size_t overhead = size_t(64) << 20;
	const size_t required = stateBytes + overhead;

	auto toGiB = [](size_t b) { return static_cast<double>(b) / (1024.0 * 1024.0 * 1024.0); };

	// Pick the device with the most free VRAM that still fits the required allocation.
	int   bestDevice  = -1;
	size_t bestFree   = 0;
	size_t bestTotal  = 0;
	size_t maxFreeSeen = 0;

	for (int dev = 0; dev < deviceCount; ++dev) {
		if (cudaSetDevice(dev) != cudaSuccess) {
			cudaGetLastError();
			continue;
		}
		size_t freeMem = 0, totalMem = 0;
		if (cudaMemGetInfo(&freeMem, &totalMem) != cudaSuccess) {
			cudaGetLastError();
			continue;
		}

		cudaDeviceProp prop{};
		const char* name = (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) ? prop.name : "<unknown>";
		std::cout << "  CUDA device " << dev << ": " << name
		          << " — free " << toGiB(freeMem) << " / " << toGiB(totalMem) << " GiB" << std::endl;

		if (freeMem > maxFreeSeen) maxFreeSeen = freeMem;
		if (freeMem >= required && freeMem > bestFree) {
			bestDevice = dev;
			bestFree   = freeMem;
			bestTotal  = totalMem;
		}
	}

	if (bestDevice < 0) {
		std::cout << "CUDA support: No (need " << toGiB(required)
		          << " GiB, best free VRAM across " << deviceCount << " device(s) is "
		          << toGiB(maxFreeSeen) << " GiB) — falling back to CPU" << std::endl;
		// Reset selection to device 0 so we don't leave a useless device current.
		cudaSetDevice(0);
		cudaGetLastError();
		return false;
	}

	// Bind the chosen device for subsequent CUDA calls (kernel launches, allocations).
	if (cudaSetDevice(bestDevice) != cudaSuccess) {
		cudaGetLastError();
		std::cout << "CUDA support: No (cudaSetDevice(" << bestDevice << ") failed)" << std::endl;
		return false;
	}

	cudaDeviceProp prop{};
	if (cudaGetDeviceProperties(&prop, bestDevice) == cudaSuccess) {
		std::cout << "CUDA support: Yes — selected device " << bestDevice
		          << " (" << prop.name << ", SM " << prop.major << "." << prop.minor
		          << ", using " << toGiB(stateBytes) << " / " << toGiB(bestFree)
		          << " GiB free of " << toGiB(bestTotal) << " GiB)" << std::endl;
	} else {
		std::cout << "CUDA support: Yes — selected device " << bestDevice
		          << " (state " << toGiB(stateBytes) << " GiB)" << std::endl;
	}
	return true;
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