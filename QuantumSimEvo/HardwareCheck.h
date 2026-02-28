#ifndef HARDWARE_CHECK_H
#define HARDWARE_CHECK_H

#ifdef QUANTUM_EXPORTS
#define QUANTUM_API __declspec(dllexport)
#else
#define QUANTUM_API __declspec(dllimport)
#endif

#include <string>

struct CPUInfo {
    size_t ramBytes = 0;
    int cores = 0;
};

struct GPUInfo {
    bool available = false;
    std::string name = "None";
    size_t vramBytes = 0;
};

class QUANTUM_API HardwareCheck {
public:
    static CPUInfo getCPU();
    static GPUInfo getGPU();

    static int maxQubits(size_t bytes);
    static void printReport();
};

#endif
