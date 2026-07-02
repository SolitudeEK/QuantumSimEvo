# QuantumSimEvo

A high-performance, state-vector quantum circuit simulator written in C++. QuantumSimEvo
runs gate-based quantum circuits on the CPU (scalar or AVX2-vectorized) or on the GPU
(CUDA), automatically selecting the fastest backend that fits the problem at runtime. It
ships with a Python binding, a suite of well-known quantum algorithms, an optional Pauli
noise model, and a correctness test harness.

## Features

- **Full state-vector simulation** of `N`-qubit systems (2^N amplitudes as
  `std::complex<double>`).
- **Automatic backend selection** at circuit construction time:
  - **CUDA GPU** — chosen when a device with enough free VRAM is available.
  - **AVX2** — SIMD-vectorized CPU path, used when the CPU supports AVX2.
  - **Scalar CPU** — portable fallback.
- **Standard gate set**: Hadamard, Pauli-X/Y/Z, phase, RX/RY/RZ rotations, CNOT,
  controlled-phase, multi-controlled phase, and SWAP.
- **Measurement & sampling**: collapse the full state (`measure`) or draw many shots
  without collapsing (`sample`).
- **Optional Pauli noise model** — accumulates bit-flip / phase-flip errors per gate and
  applies them at execution time.
- **Python module** via pybind11 (`import QuantumSimEvo`).
- **Algorithm playground**: Grover, QFT/IQFT, Shor, QAOA (Max-Cut), and a TSP/QUBO solver.
- **Hardware reporting** and a **test harness** that validates gates against expected
  state vectors.

## Repository layout

| Path | Description |
| --- | --- |
| `QuantumSimEvo/` | Core simulator library (the engine). Builds to `QuantumSimEvo.dll`. |
| `QSEPythonModule/` | pybind11 wrapper exposing the `Circuit` API to Python. |
| `QSEplayground/` | Console app with example circuits and quantum algorithms. |
| `QSETests/` | Correctness tests validating gate behavior against known states. |
| `QuantumSimEvo.slnx` | Visual Studio solution (x64 / x86). |

### Core library (`QuantumSimEvo/`)

- **`Circuit`** (`Circuit.h/.cpp`) — the public API. Build a circuit by queuing gates, then
  call `execute()`. Holds the state vector, the command list, the chosen backend, and the
  optional noise model.
- **`StateVector`** (`StateVector.h/.cpp`) — owns the amplitude array; supports full and
  per-qubit measurement, reset, and multi-shot sampling.
- **`ICircuitUnitaryOperation`** (`ICircuitUnitaryOperation.h`) — backend interface. Each
  gate is a virtual `apply*` method. Backends that mirror state on a device expose
  `flush()` / `markHostDirty()` to keep host and device in sync.
- **Backends**:
  - `CircuitUnitaryOperation` — scalar CPU.
  - `CircuitUnitaryOperationAVX` — AVX2-vectorized CPU.
  - `CircuitUnitaryOperationGPU` (`.cu`) — CUDA GPU.
- **`CircuitUnitaryOperationFactory`** — picks a backend: CUDA if a device has room, else
  AVX2 if supported, else scalar.
- **`Noise`** (`Noise.h/.cpp`) — Pauli noise model; tracks and applies bit/phase flips.
- **`HardwareCheck`** — reports CPU cores / RAM and GPU name / VRAM, and estimates the max
  number of qubits that fit in a given memory budget.
- **`Structurs.h`** — shared types: `GateType`, `GateCommand`, and the `Gate2x2` matrix.

## The `Circuit` API

```cpp
Circuit(size_t N, bool simulatePauliNoise = false);

// Single-qubit gates
void hadamard(size_t q);
void pauliX(size_t q);
void pauliY(size_t q);
void pauliZ(size_t q);
void phase(double theta, size_t q);
void rotateX(double theta, size_t q);
void rotateY(double theta, size_t q);
void rotateZ(double theta, size_t q);

// Multi-qubit gates
void cnot(size_t control, size_t target);
void cphase(double theta, size_t control, size_t target);
void mcphase(double theta, const std::vector<size_t>& controls, size_t target);
void swap(size_t q0, size_t q1);

// Execution & measurement
void execute(bool print_steps = false);   // runs the queued gates (and noise, if enabled)
size_t measure();                          // collapses the state, returns the outcome index
std::vector<size_t> sample(int numShots);  // samples outcomes without collapsing
void reset();                              // resets state to |0...0> and clears the queue

// Utilities
void printState();
StateVector& getStateVector();
```

Gates are **queued**, not applied immediately — `execute()` runs the whole queue in order.
`reset()` restores `|0...0>` and clears the queued gates.

### C++ example — Bell state

```cpp
#include "Circuit.h"

Circuit qc(2);
qc.hadamard(0);
qc.cnot(0, 1);
qc.execute();
qc.printState();          // ~0.707|00> + ~0.707|11>
size_t outcome = qc.measure();
```

## Python module

The `QSEPythonModule` project builds a pybind11 extension named `QuantumSimEvo`. Method
names are snake_case:

```python
import QuantumSimEvo as qse

qc = qse.Circuit(2)              # Circuit(num_qubits, simulate_pauli_noise=False)
qc.hadamard(0)
qc.cnot(0, 1)
qc.execute()                     # execute(print_steps=False)
qc.print_state()

counts = qc.sample(1000)         # list of outcome indices, no collapse
outcome = qc.measure()           # collapses the state
```

Available methods: `hadamard`, `pauli_x`, `pauli_y`, `pauli_z`, `phase`, `rotate_x`,
`rotate_y`, `rotate_z`, `cnot`, `cphase`, `mcphase`, `swap`, `execute`, `measure`,
`sample`, `reset`, `print_state`.

## Algorithm playground (`QSEplayground/`)

Ready-to-run implementations driven from `QSEplayground.cpp::main` (uncomment the one you
want):

- **Grover** (`Grover::runGrover`) — unstructured search with oracle + diffusion.
- **QFT / IQFT** (`QFT::runQFT`, `QFT::runIQFT`) — quantum Fourier transform.
- **Shor** (`Shor::runShor`) — integer factoring (modular exponentiation + QFT).
- **QAOA** (`QAOA::runQAOA`) — Max-Cut, with a brute-force reference (`bruteForceMaxCut`).
- **TVS / TSP** (`TVS::runTVS`) — travelling-salesman as a QUBO/Ising problem.
- **Bell / GHZ** — entanglement demos showing measurement-outcome distributions.

## Building

This is a Visual Studio solution targeting Windows (x64 recommended).

Requirements:
- Visual Studio with the C++ toolset (MSVC; Intel oneAPI compiler is also configured).
- **CUDA Toolkit** — required to build the GPU backend and the factory (`cuda_runtime.h`).
- **pybind11** and a matching **Python** install — for the `QSEPythonModule` project.

Build steps:
1. Open `QuantumSimEvo.slnx` in Visual Studio.
2. Select the `x64` platform and a configuration (`Debug` / `Release`).
3. Build the solution. `QuantumSimEvo` produces the core DLL; `QSEplayground` produces the
   console demo; `QSEPythonModule` produces the Python extension; `QSETests` produces the
   test runner.

At runtime the factory prints which backend was selected (CUDA device / AVX2 / scalar CPU),
so you can confirm the engine is using the hardware you expect.

## Testing

Build and run the `QSETests` project. It applies gate sequences via the `Circuit` API and
compares the resulting state vector against analytically expected amplitudes (tolerance
`1e-12`), reporting pass/fail per case.

## License

See the repository for license details.
