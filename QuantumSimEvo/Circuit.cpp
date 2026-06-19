#include "Circuit.h"
#include "Structurs.h"
#include "CircuitUnitaryOperationFactory.h"

Circuit::Circuit(size_t N, bool simulatePauliNoise) : stateVector(N), simulateNoise(simulatePauliNoise), pauliNoise(N, 42, 1) {
    circuitOp = CircuitUnitaryOperationFactory::create(N);
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

void Circuit::cphase(double theta, size_t control, size_t target) {
    commands.push_back({ GateType::CPHASE, target, control, theta,
    "CPhase (theta: " + std::to_string(theta) + ", control: " + std::to_string(control) + ", target: " + std::to_string(target) + ")" });
}

void Circuit::mcphase(double theta, const std::vector<size_t>& controls, size_t target) {
    std::string desc = "MCPhase (theta: " + std::to_string(theta) + ", target: " + std::to_string(target) + ", controls:";
    for (size_t c : controls) desc += " " + std::to_string(c);
    desc += ")";
    GateCommand cmd{ GateType::MCPHASE, target, 0, theta, desc, 0, controls };
    commands.push_back(cmd);
}

void Circuit::swap(size_t q0, size_t q1) {
    commands.push_back({ GateType::SWAP, q0, q1, 0.0,
    "SWAP (q0: " + std::to_string(q0) + ", q1: " + std::to_string(q1) + ")" });
}

void Circuit::execute(bool print_steps) {
    if (print_steps) std::cout << "Initial state: |0...0>\n";
	circuitOp->setPrintSteps(print_steps);
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
        case GateType::CPHASE:
            circuitOp->applyCPhase(stateVector, cmd.control, cmd.target, cmd.theta);
            break;
        case GateType::MCPHASE:
            circuitOp->applyMCPhase(stateVector, cmd.controls_list, cmd.target, cmd.theta);
            break;
        case GateType::SWAP:
            circuitOp->applySwap(stateVector, cmd.target, cmd.control);
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
        pauliNoise.applyPauliNoise(stateVector, *circuitOp);
    if (print_steps)
        std::cout << "Total of occured errors: \n BitFlips:" << pauliNoise.getTotalBitFlips()
	<< " times\n PhaseFlips:" << pauliNoise.getTotalPhaseFlips() << " times.\n";
}

size_t Circuit::measure() {
    circuitOp->flush();
    return stateVector.measure();
}

void Circuit::reset() {
    stateVector.reset();
    circuitOp->markHostDirty();
    if (simulateNoise)
        pauliNoise.resetStats();
    commands.clear();
}

void Circuit::printState() {
    circuitOp->flush();
    stateVector.printState();
}

std::vector<size_t> Circuit::sample(int numShots){
    circuitOp->flush();
    return stateVector.sample(numShots);
}