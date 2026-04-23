#include <iostream>
#include <vector>
#include <string>
#include <complex>
#include <functional>
#include <cmath>
#include <iomanip>
#include <Windows.h>
#include <omp.h>
#include "../QuantumSimEvo/Circuit.h"
#include <chrono>


struct TestResult {
    std::string testName;
    bool passed;
};

const double PI = 3.14159265358979323846;

class Program
{
private:
    Circuit* qc;

    bool validate(std::string name, std::function<void(Circuit&)> gateFunc,
        const std::vector<std::complex<double>>& expectedState)
    {
        qc->reset();
        gateFunc(*qc);
        qc->execute(true);

        auto& actualState = qc->getStateVector().data();
        bool success = true;

        for (size_t i = 0; i < expectedState.size(); ++i) {
            double realDiff = std::abs(actualState[i].real() - expectedState[i].real());
            double imagDiff = std::abs(actualState[i].imag() - expectedState[i].imag());

            if (realDiff > 1e-12 || imagDiff > 1e-12) {
                success = false;
                setConsoleColor(FOREGROUND_RED | FOREGROUND_INTENSITY);
                std::cout << "[FAIL] " << name << " index " << i << "\n"
                    << "   Expected: " << expectedState[i] << "\n"
                    << "   Actual:   " << actualState[i] << std::endl;
                setConsoleColor(7); // Reset to white
            }
        }
        return success;
    }

    std::string benchmark(std::string name, std::function<void(Circuit&)> gateFunc, size_t iterations = 10) {

        qc->reset();

        for (int i = 0; i < iterations; ++i)
            gateFunc(*qc);

        auto start = std::chrono::high_resolution_clock::now();
            
        qc->execute(false);

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;

        double avgTime = duration.count() / iterations;

        std::ostringstream oss;
        oss << "[BENCHMARK] " << name << " run in avg: "
            << std::fixed << std::setprecision(4) << avgTime << " ms. \n";

        return oss.str();
	}

    // Test definitions
    bool testHadamard() {
        double s = 1.0 / std::sqrt(2.0);
        return validate("Hadamard", [](Circuit& c) { c.hadamard(0); }, { {s, 0}, {s, 0} });
    }

    bool testPauliX() {
        return validate("PauliX", [](Circuit& c) { c.pauliX(0); }, { {0, 0}, {1, 0} });
    }

    bool testPauliY() {
        return validate("PauliY", [](Circuit& c) { c.pauliY(0); }, { {0, 0}, {0, 1} });
    }

    bool testPauliZ() {
        return validate("PauliZ", [](Circuit& c) { c.pauliZ(0); }, { {1, 0}, {0, 0} });
    }

    bool testPhase() {
        return validate("Phase", [](Circuit& c) { c.phase(PI / 2.0, 0); }, { {1, 0}, {0, 0} });
    }

    bool testRotateX() {
        double theta = PI / 2.0;
        double c_val = std::cos(theta / 2.0);
        double s_val = std::sin(theta / 2.0);
        return validate("RotateX", [theta](Circuit& c) { c.rotateX(theta, 0); },
            { {c_val, 0}, {0, -s_val} });
    }

    bool testRotateY() {
        double theta = PI / 2.0;
        double val = std::cos(theta / 2.0);
        return validate("RotateY", [theta](Circuit& c) { c.rotateY(theta, 0); },
            { {val, 0}, {val, 0} });
    }

    bool testRotateZ() {
        double theta = PI / 2.0;
        double re = std::cos(theta / 2.0);
        double im = std::sin(theta / 2.0);
        return validate("RotateZ", [theta](Circuit& c) { c.rotateZ(theta, 0); },
            { {re, -im}, {0, 0} });
    }

    bool testSwap() {
        bool ok = true;
        const double tol = 1e-12;
        const double s = 1.0 / std::sqrt(2.0);

        struct SwapCase {
            std::string name;
            std::function<void(Circuit&)> setup;
            std::vector<std::complex<double>> expected;
        };

        // Qubit 0 is LSB: index = q1*2 + q0
        // |00>=idx0, |01>=idx1 (q0=1), |10>=idx2 (q1=1), |11>=idx3
        std::vector<SwapCase> cases = {
            { "|00> unchanged",
              [](Circuit& c) { c.swap(0, 1); },
              { {1,0},{0,0},{0,0},{0,0} } },
            { "|01> -> |10>",
              [](Circuit& c) { c.pauliX(0); c.swap(0, 1); },
              { {0,0},{0,0},{1,0},{0,0} } },
            { "|10> -> |01>",
              [](Circuit& c) { c.pauliX(1); c.swap(0, 1); },
              { {0,0},{1,0},{0,0},{0,0} } },
            { "H(0) superposition",
              [](Circuit& c) { c.hadamard(0); c.swap(0, 1); },
              { {s,0},{0,0},{s,0},{0,0} } },
            { "SWAP is self-inverse",
              [](Circuit& c) { c.pauliX(0); c.swap(0, 1); c.swap(0, 1); },
              { {0,0},{1,0},{0,0},{0,0} } },
        };

        for (auto& tc : cases) {
            Circuit c(2);
            tc.setup(c);
            c.execute(true);
            auto& actual = c.getStateVector().data();
            for (size_t i = 0; i < tc.expected.size(); ++i) {
                double dr = std::abs(actual[i].real() - tc.expected[i].real());
                double di = std::abs(actual[i].imag() - tc.expected[i].imag());
                if (dr > tol || di > tol) {
                    ok = false;
                    setConsoleColor(FOREGROUND_RED | FOREGROUND_INTENSITY);
                    std::cout << "[FAIL] Swap/" << tc.name << " index " << i
                              << "\n   Expected: " << tc.expected[i]
                              << "\n   Actual:   " << actual[i] << "\n";
                    setConsoleColor(7);
                }
            }
        }
        return ok;
    }

    static void setConsoleColor(WORD color) {
        SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), color);
    }

public:
    Program(size_t numQubits) {
        qc = new Circuit(numQubits);
    }

    int performTests() {
        std::vector<TestResult> results;

        std::cout << "Starting Quantum Simulator Hardware Logic Tests...\n";
        std::cout << "-----------------------------------------------\n";

        // Run tests and store results
        results.push_back({ "Hadamard", testHadamard() });
        results.push_back({ "PauliX",   testPauliX() });
        results.push_back({ "PauliY",   testPauliY() });
        results.push_back({ "PauliZ",   testPauliZ() });
        results.push_back({ "Phase",    testPhase() });
        results.push_back({ "RotateX",  testRotateX() });
        results.push_back({ "RotateY",  testRotateY() });
        results.push_back({ "RotateZ",  testRotateZ() });
        results.push_back({ "Swap",     testSwap() });

        // Final Report
        std::cout << "\n--- TEST SUMMARY ---\n";
        int passedCount = 0;

        for (const auto& res : results) {
            std::cout << std::left << std::setw(15) << res.testName << ": ";
            if (res.passed) {
                Program::setConsoleColor(FOREGROUND_GREEN | FOREGROUND_INTENSITY);
                std::cout << "PASSED" << std::endl;
                passedCount++;
            }
            else {
                Program::setConsoleColor(FOREGROUND_RED | FOREGROUND_INTENSITY);
                std::cout << "FAILED" << std::endl;
            }
            Program::setConsoleColor(7); // Reset
        }

        std::cout << "-----------------------------------------------\n";
        std::cout << "Result: " << passedCount << "/" << results.size() << " tests passed.\n";

        if (passedCount != results.size()) {
            std::cout << "Check the red lines above for precision errors." << std::endl;
        }

        return (passedCount == results.size()) ? 0 : 1;
    }

    void performBenchmarks(size_t n) {
		std::vector<std::string> benchmarkResults;

        std::cout << "\nStarting Quantum Simulator Performance Benchmarks...\n";
        
        benchmarkResults.push_back(benchmark("Hadamard  ", [](Circuit& c) { c.hadamard(0); }));
        benchmarkResults.push_back(benchmark("PauliX    ", [](Circuit& c) { c.pauliX(0); }));
        benchmarkResults.push_back(benchmark("PauliY    ", [](Circuit& c) { c.pauliY(0); }));
        benchmarkResults.push_back(benchmark("PauliZ    ", [](Circuit& c) { c.pauliZ(0); }));
        benchmarkResults.push_back(benchmark("Phase     ", [](Circuit& c) { c.phase(PI / 2.0, 0); }));
        benchmarkResults.push_back(benchmark("RotateX   ", [](Circuit& c) { c.rotateX(PI / 2.0, 0); }));
        benchmarkResults.push_back(benchmark("RotateY   ", [](Circuit& c) { c.rotateY(PI / 2.0, 0); }));
        benchmarkResults.push_back(benchmark("RotateZ   ", [](Circuit& c) { c.rotateZ(PI / 2.0, 0); }));
		benchmarkResults.push_back(benchmark("CNOT      ", [](Circuit& c) { c.cnot(0, 1); }));
		benchmarkResults.push_back(benchmark("Swap      ", [](Circuit& c) { c.swap(0, 1); }));

		std::cout << "\n------- Benchmark Summary for " << n << " Qubits -------\n";
        std::cout << "-----------------------------------------------\n";
        for (const auto& res : benchmarkResults) {
            std::cout << res;
		}
        std::cout << "-----------------------------------------------\n";
    }
};

int main()
{
	std::cout << "Run tests or benchmarks? (t/b): ";
	char choice;
	std::cin >> choice;

    if (choice == 'b' || choice == 'B') {
        std::cout << "The Benchmark can take minutes for 28+ qubits\n";
		std::cout << "Enter number of qubits for benchmarking (e.g., 25): ";
        size_t numQubits;
		std::cin >> numQubits;

        Program program(numQubits);
        program.performBenchmarks(numQubits);
        return 0;
	}
	if (choice == 't' || choice == 'T')
    {
        Program program(1);
        return program.performTests();
    }
    else {
        std::cout << "Invalid choice. Please run the program again and enter 't' for tests or 'b' for benchmarks." << std::endl;
        return 1;
	}
}

