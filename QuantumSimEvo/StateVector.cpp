#include "StateVector.h"
#include <vector>
#include <complex>
#include <iostream>
#include <random>
#include <bitset>
#include <chrono>
#include <omp.h>
#include "Structurs.h"

using Complex = std::complex<double>;


StateVector::StateVector(size_t num) : numQubits(num) {
	stateSize = 1ULL << numQubits;

	auto start = std::chrono::high_resolution_clock::now();

	try {
		stateVector.resize(stateSize, 0.0);
	}
	catch (const std::bad_alloc& e) {
		std::cerr << "Error: Not enough memory for " << num << " qubits.\n";
		exit(1);
	}
	// Initialize to |00...0>
	stateVector[0] = 1.0;

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> ms = end - start;
	std::cout << "Memory allocation done in: " << ms.count() << " ms \n"; //Debug timing
}

void StateVector::reset() {
	auto start = std::chrono::high_resolution_clock::now();

	#pragma omp parallel for
	for (std::ptrdiff_t i = 0; i < (std::ptrdiff_t)stateSize; ++i)
		stateVector[i] = (i == 0) ? Complex{ 1.0, 0.0 } : Complex{ 0.0, 0.0 };
	

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> ms = end - start;
	std::cout << "Memory cleaning done in: " << ms.count() << " ms \n";
}
/*size_t StateVector::measure() { v1
	std::vector<double> probs(stateSize);
	double totalProb = 0.0;

	auto start = std::chrono::high_resolution_clock::now();

	#pragma omp parallel for reduction(+:totalProb)
	for (std::ptrdiff_t i = 0; i < stateSize; ++i) {
		probs[i] = std::norm(stateVector[i]);
		totalProb += probs[i];
	}

	std::random_device rd;
	std::mt19937 gen(rd());
	std::discrete_distribution<> dist(probs.begin(), probs.end());
	size_t collapsedState = dist(gen);

	// Collapse the state vector
	#pragma omp parallel for
	for (std::ptrdiff_t i = 0; i < stateSize; ++i)
		stateVector[i] = (i == collapsedState) ? 1.0 : 0.0;
	
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> ms = end - start;
	std::cout << "Mesure done in: " << ms.count() << " ms \n"; //Debug timing

	return collapsedState;
}*/

std::vector<size_t> StateVector::sample(int numShots) {
	size_t size = this->stateSize;
	std::vector<double> cumulative(size);
	std::vector<size_t> results(numShots);

	auto start = std::chrono::high_resolution_clock::now();


#pragma omp parallel for
	for (std::ptrdiff_t i = 0; i < size; ++i) {
		cumulative[i] = std::norm(stateVector[i]);
	}

	// 2. Compute Cumulative Distribution (Scanning)
	// This part is inherently serial, but for 65k elements it is very fast
	for (size_t i = 1; i < size; ++i) {
		cumulative[i] += cumulative[i - 1];
	}

	// Normalizing the last element to handle floating point precision
	double totalProb = cumulative[size - 1];

	// 3. Parallel Sampling (The "Many Shots" part)
#pragma omp parallel
	{
		// Give each thread its own random engine to avoid race conditions
		std::random_device rd;
		std::mt19937 gen(rd() ^ omp_get_thread_num());
		std::uniform_real_distribution<double> dist(0.0, totalProb);

#pragma omp for
		for (int s = 0; s < numShots; ++s) {
			double target = dist(gen);

			// Binary search to find the index (O(log N))
			// This is much faster than a linear loop for large state vectors
			auto it = std::lower_bound(cumulative.begin(), cumulative.end(), target);
			results[s] = std::distance(cumulative.begin(), it);
		}
	}

	return results;

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> ms = end - start;
	std::cout << "Generated " << numShots << " samples in: " << ms.count() << " ms\n";

	return results;
}

size_t StateVector::measure() {
	auto start = std::chrono::high_resolution_clock::now();

	double totalProb = 0.0;

	#pragma omp parallel for reduction(+:totalProb)
	for (std::ptrdiff_t i = 0; i < stateSize; ++i)
		totalProb += std::norm(stateVector[i]);

	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<double> dist(0.0, totalProb);
	double target = dist(gen);

	size_t collapsedState = 0;
	double runningSum = 0.0;
	for (size_t i = 0; i < stateSize; ++i) {
		runningSum += std::norm(stateVector[i]);
		if (runningSum >= target) {
			collapsedState = i;
			break;
		}
	}

	#pragma omp parallel
	{
	#pragma omp for
		for (std::ptrdiff_t i = 0; i < stateSize; ++i) {
			stateVector[i] = 0.0;
		}
	}
	stateVector[collapsedState] = 1.0;

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> ms = end - start;
	std::cout << "Measure done in: " << ms.count() << " ms \n";

	return collapsedState;
}

void  StateVector::applyUnitaryOperation(const Gate2x2& gate, size_t targetQubit) {
	Complex u00 = gate[0][0], u01 = gate[0][1];
	Complex u10 = gate[1][0], u11 = gate[1][1];

	size_t halfState = stateSize >> 1;
	size_t lowerMask = (1ULL << targetQubit) - 1;
	size_t step = 1ULL << targetQubit;

	auto start = std::chrono::high_resolution_clock::now();

	#pragma omp parallel for
	for (std::ptrdiff_t i = 0; i < halfState; ++i) {
		size_t idx_zero = ((i & ~lowerMask) << 1) | (i & lowerMask);
		size_t idx_one = idx_zero | step;

		Complex alpha = stateVector[idx_zero];
		Complex beta = stateVector[idx_one];

		stateVector[idx_zero] = u00 * alpha + u01 * beta;
		stateVector[idx_one] = u10 * alpha + u11 * beta;
	}

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> ms = end - start;
	std::cout << "Gate applied in: " << ms.count() << " ms \n"; //Debug timing
}

void StateVector::printState() {
	// Safety limations for large state vectors
	size_t limit = (numQubits > 10) ? 32 : stateSize;

	for (size_t i = 0; i < limit; ++i) {
		if (std::abs(stateVector[i]) > 1e-10) {
			std::cout << stateAsString(i) << " : " << stateVector[i] << std::endl;
		}
	}
	if (numQubits > 10) std::cout << "... (truncated for large N)\n";
}
	
std::string StateVector::stateAsString(size_t index) {
	std::string s = std::bitset<64>(index).to_string();
	return "|" + s.substr(64 - numQubits) + ">";
}