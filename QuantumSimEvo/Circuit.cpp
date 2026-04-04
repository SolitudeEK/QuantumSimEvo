#include "Circuit.h"
#include "Structurs.h"
#include "CircuitUnitaryOperationFactory.h"

Circuit::Circuit(size_t N, bool simulatePauliNoise) : numQubits(N), stateVector(N), pauliNoise(N, 42, 1), simulateNoise(simulatePauliNoise) {
    circuitOp = CircuitUnitaryOperationFactory::create();
}

void Circuit::hadamard(size_t q) {
    commands.push_back({ GateType::H, q, 0, 0.0, "Hadamard on qubit " + std::to_string(q) });
}

void Circuit::pauliX(size_t q) {
    commands.push_back({ GateType::X, q, 0, 0.0, "PauliX on qubit " + std::to_string(q) });
}

void Circuit::pauliZ(size_t q) {
    commands.push_back({ GateType::Z, q, 0, 0.0, "PauliZ on qubit " + std::to_string(q) });
}

void Circuit::pauliY(size_t q) {
    commands.push_back({ GateType::Y, q, 0, 0.0, "PauliX on qubit " + std::to_string(q) });
}

void Circuit::rotateX(double theta, size_t q) {
    commands.push_back({ GateType::RX, q, 0, theta, "RX on qubit " + std::to_string(q) });
}

void Circuit::rotateZ(double theta, size_t q) {
    commands.push_back({ GateType::RZ, q, 0, theta, "RZ on qubit " + std::to_string(q) });
}

void Circuit::rotateY(double theta, size_t q) {
    commands.push_back({ GateType::RY, q, 0, theta, "RY on qubit " + std::to_string(q) });
}

void Circuit::phase(double theta, size_t q) {
    commands.push_back({ GateType::PHASE, q, 0, theta, "Phase shift on qubit " + std::to_string(q) });
}

void Circuit::cnot(size_t control, size_t target) {
    commands.push_back({ GateType::CNOT, target, control, 0.0,
    "CNOT (control: " + std::to_string(control) + ", target: " + std::to_string(target) + ")" });
}

void Circuit::execute(bool print_steps) {
    if (print_steps) std::cout << "Initial state: |0...0>\n";

    for (const auto& cmd : commands) {
        if (print_steps) std::cout << "Executing: " << cmd.description << std::endl;

        if (simulateNoise)
            pauliNoise.accumulateError(cmd.type, cmd.target);

        switch (cmd.type) {
        case GateType::H:
            circuitOp->applyHadamard(stateVector, cmd.target);
            break;
        case GateType::X:
            circuitOp->applyPauliX(stateVector, cmd.target);
            break;
        case GateType::RX:
            circuitOp->applyRotateX(stateVector, cmd.target, cmd.theta);
            break;
        case GateType::RY:
            circuitOp->applyRotateY(stateVector, cmd.target, cmd.theta);
            break;
        case GateType::RZ:
            circuitOp->applyRotateZ(stateVector, cmd.target, cmd.theta);
            break;
        case GateType::CNOT:
            circuitOp->applyCNOT(stateVector, cmd.control, cmd.target);
            break;
        case GateType::Y:
            circuitOp->applyPauliY(stateVector, cmd.target);
            break;
        case GateType::PHASE:
            circuitOp->applyPhase(stateVector, cmd.target, cmd.theta);
            break;
        case GateType::Z:
            circuitOp->applyPauliZ(stateVector, cmd.target);
            break;
        default:
            std::cerr << "Unsupported gate type: " << static_cast<int>(cmd.type) << std::endl;
        }
    }

    if (simulateNoise)
        pauliNoise.applyPauliNoise(stateVector);
    if (print_steps)
        std::cout << "Total of occured errors: \n BitFlips:" << pauliNoise.getTotalBitFlips()
	<< " times\n PhaseFlips:" << pauliNoise.getTotalPhaseFlips() << " times.\n";
}

size_t Circuit::measure() {
    return stateVector.measure();
}

void Circuit::reset() {
    stateVector.reset();
    if (simulateNoise)
        pauliNoise.resetStats();
    commands.clear();
}

void Circuit::printState() {
    stateVector.printState();
}

std::vector<size_t> Circuit::sample(int numShots){
	return stateVector.sample(numShots);
}