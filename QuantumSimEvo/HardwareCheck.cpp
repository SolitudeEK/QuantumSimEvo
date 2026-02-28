#include "HardwareCheck.h"
#include <iostream>
#include <cmath>
#include <thread>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/sysinfo.h>
#endif

// CUDA
#include <cuda_runtime.h>

CPUInfo HardwareCheck::getCPU() {
    CPUInfo info;

    info.cores = std::thread::hardware_concurrency();

#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    info.ramBytes = status.ullTotalPhys;
#else
    struct sysinfo memInfo;
    sysinfo(&memInfo);
    info.ramBytes = memInfo.totalram;
    info.ramBytes *= memInfo.mem_unit;
#endif

    return info;
}

GPUInfo HardwareCheck::getGPU() {
    GPUInfo info;
    
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);

	if (err != cudaSuccess) {
		std::cout << "CUDA Error: " << err << "\n";
        return info; // No GPU available
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    info.available = true;
    info.name = prop.name;
    info.vramBytes = prop.totalGlobalMem;
    
    return info;
}

int HardwareCheck::maxQubits(size_t bytes) {
    double usable = bytes * 0.8;
    double states = usable / 16.0;
    return (int)std::floor(std::log2(states));
}

void HardwareCheck::printReport() {

    auto cpu = getCPU();
    auto gpu = getGPU();

    int cpuQ = maxQubits(cpu.ramBytes);

    std::cout << "\n===== QuantumSim Hardware Check =====\n\n";

    std::cout << "CPU Cores : " << cpu.cores << "\n";
    std::cout << "CPU RAM   : "
        << cpu.ramBytes / (1024.0 * 1024 * 1024)
        << " GB\n";

    std::cout << "Max CPU Qubits : " << cpuQ << "\n\n";

    if (gpu.available) {
        int gpuQ = maxQubits(gpu.vramBytes);

        std::cout << "GPU        : " << gpu.name << "\n";
        std::cout << "GPU VRAM   : "
            << gpu.vramBytes / (1024.0 * 1024 * 1024)
            << " GB\n";

        std::cout << "Max GPU Qubits : " << gpuQ << "\n\n";

        std::cout << "Recommended Backend : "
            << (gpuQ > cpuQ ? "GPU" : "CPU")
            << "\n";
    }
    else {
        std::cout << "GPU        : Not detected\n";
        std::cout << "Recommended Backend : CPU\n";
    }

    std::cout << "\n=====================================\n";
}
