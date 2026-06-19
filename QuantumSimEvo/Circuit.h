#ifndef CIRCUIT_H
#define CIRCUIT_H

#ifdef QUANTUM_EXPORTS
#define QUANTUM_API __declspec(dllexport)
#else
#define QUANTUM_API __declspec(dllimport)
#endif

#include <vector>
#include <string>
#include <iostream>
#include "StateVector.h"
#include "Structurs.h"
#include "QubitUnitaryOperation.h"
#include "CircuitUnitaryOperation.h"
#include "Noise.h"

class QUANTUM_API Circuit {
private:
    StateVector stateVector;
    std::vector<GateCommand> commands;
    std::unique_ptr<ICircuitUnitaryOperation> circuitOp;

    bool simulateNoise;
    Noise pauliNoise;
public:
    // Constructor
    Circuit(size_t N, bool simulatePauliNoise = false);

    // Gate methods
    void hadamard(size_t q);
    void pauliX(size_t q);
    void pauliZ(size_t q);
    void pauliY(size_t q);
	void phase(double theta, size_t q);
    void rotateX(double theta, size_t q);
    void rotateY(double theta, size_t q);
    void rotateZ(double theta, size_t q);
    void cnot(size_t control, size_t target);
    void cphase(double theta, size_t control, size_t target);
    void mcphase(double theta, const std::vector<size_t>& controls, size_t target);
    void swap(size_t q0, size_t q1);

    // Execution & Measurement
    void execute(bool print_steps = false);
	size_t measure(); // Collapses the full state and returns the measured value
	std::vector<size_t> sample(int numShots); // Returns a vector of measurement results without collapsing the state
	void reset();

    // Utilities
    void printState();
	StateVector& getStateVector() { return stateVector; }
};

#endif