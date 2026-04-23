#ifndef STATE_VECTOR_H
#define STATE_VECTOR_H

#ifdef QUANTUM_EXPORTS
#define QUANTUM_API __declspec(dllexport)
#else
#define QUANTUM_API __declspec(dllimport)
#endif

#include <vector>
#include <complex>
#include <string>
#include "Structurs.h"

using Complex = std::complex<double>;

class QUANTUM_API StateVector {
private:
    size_t numQubits, stateSize;
    std::vector<Complex> stateVector;

public:
    StateVector(size_t num);

    //Sugested for CUDA implementation, to avoid copying data back and forth
    inline Complex* getRawPtr() {
        return stateVector.data();
    }

    // For CPU operations, we can still use the vector interface
    inline std::vector<Complex>& data(){
        return stateVector;
    }

    //size accessor for convenience
    inline size_t size() const { return stateSize; }
    inline size_t qubits() const { return numQubits; }

	void reset();
    size_t measure();
    int measureQubit(size_t qubit); // Partial measurement: collapses one qubit, returns 0 or 1
    void resetQubit(size_t qubit);  // Measures qubit, conditionally flips to |0>, always leaves qubit in |0>
    std::vector<size_t> sample(int numShots);
    void applyUnitaryOperation(const Gate2x2& gate, size_t targetQubit);
    void printState();
    std::string stateAsString(size_t index);
};

#endif