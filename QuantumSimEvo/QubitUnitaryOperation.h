#ifndef QUBIT_UNITARY_OPERATION_H
#define QUBIT_UNITARY_OPERATION_H

#ifdef QUANTUM_EXPORTS
#define QUANTUM_API __declspec(dllexport)
#else
#define QUANTUM_API __declspec(dllimport)
#endif

#include <complex>
#include "Structurs.h"

using Complex = std::complex<double>;

class QUANTUM_API QubitUnitaryOperation {
public:
    static constexpr Complex I{ 0.0, 1.0 };

    static Gate2x2 getIdentity();
    static Gate2x2 getPauliX();
    static Gate2x2 getPauliY();
    static Gate2x2 getPauliZ();
    static Gate2x2 getHadamard();
    static Gate2x2 getPhase(double theta);
    static Gate2x2 getRotateX(double theta);
    static Gate2x2 getRotateY(double theta);
    static Gate2x2 getRotateZ(double theta);
};

#endif