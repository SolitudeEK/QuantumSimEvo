#ifndef STRUCTURES_H
#define STRUCTURES_H
#include <string>
#include <complex>

using Complex = std::complex<double>;

// Define the types of supported gates
enum class GateType { IDENTITY, X, Y, Z, H, PHASE, RX, RY, RZ, CNOT };

struct GateCommand {
    GateType type;
    size_t target;
    size_t control; // Only used for CNOT
    double theta;   // Only used for rotation gates
    std::string description;
};

// A fixed-size 2x2 matrix structure with an explicit constructor
struct Gate2x2 {
    Complex data[2][2];

    Gate2x2() = default;
    Gate2x2(Complex a, Complex b, Complex c, Complex d) {
        data[0][0] = a;
        data[0][1] = b;
        data[1][0] = c;
        data[1][1] = d;
    }

    Complex* operator[](int row) { return data[row]; }
    const Complex* operator[](int row) const { return data[row]; }
};

#endif