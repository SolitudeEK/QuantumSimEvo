#include "Circuit.h"
#include "Structurs.h"

bool useAVX = true; // Global flag to control AVX usage

Circuit::Circuit(size_t N) : numQubits(N), stateVector(N) {}

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

        // Route the command to the optimized Executor backend
        switch (cmd.type) {
            case GateType::H:
                if (useAVX)
                    CircuitUnitaryOperation::applyHadamardAVX(stateVector, cmd.target);
                else
                    CircuitUnitaryOperation::applyHadamard(stateVector, cmd.target);
                break;
            case GateType::X:
                CircuitUnitaryOperation::applyPauliX(stateVector, cmd.target);
                break;
            case GateType::RX:
                if(useAVX)
					CircuitUnitaryOperation::applyRotateXAVX(stateVector, cmd.target, cmd.theta);
                else
				    CircuitUnitaryOperation::applyRotateX(stateVector, cmd.target, cmd.theta);
                break;
            case GateType::RY:
                if (useAVX)
                    CircuitUnitaryOperation::applyRotateYAVX(stateVector, cmd.target, cmd.theta);
				else
					CircuitUnitaryOperation::applyRotateY(stateVector, cmd.target, cmd.theta);
                break;
            case GateType::RZ:
                CircuitUnitaryOperation::applyRotateZ(stateVector, cmd.target, cmd.theta);
                break;
            case GateType::CNOT:
                CircuitUnitaryOperation::applyCNOT(stateVector, cmd.control, cmd.target);
				break;
			case GateType::Y:
                if (useAVX)
                    CircuitUnitaryOperation::applyPauliYAVX(stateVector, cmd.target);
				else
                    CircuitUnitaryOperation::applyPauliY(stateVector, cmd.target);
                break;
            case GateType::PHASE:
                CircuitUnitaryOperation::applyPhase(stateVector, cmd.target, cmd.theta);
				break;
			case GateType::Z:
                if (useAVX)
                    CircuitUnitaryOperation::applyPauliZAVX(stateVector, cmd.target);
                else
                    CircuitUnitaryOperation::applyPauliZ(stateVector, cmd.target);
                break;
            default:
				std::cerr << "Unsupported gate type: " << static_cast<int>(cmd.type) << std::endl;
            }
        }
    }

size_t Circuit::measure() {
    return stateVector.measure();
}

void Circuit::reset() {
    stateVector.reset();
    commands.clear();
}

void Circuit::printState() {
    stateVector.printState();
}
