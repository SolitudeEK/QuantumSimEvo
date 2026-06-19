#ifndef NOISE_H
#define NOISE_H
#include<vector>
#include<complex>
#include <random>
#include<map>

#include "ICircuitUnitaryOperation.h"
#include "Structurs.h"

using Complex = std::complex<double>;

class Noise
{
private:
	std::vector<int> bitFlipCounts;
	std::vector<int> phaseFlipCounts;
	std::map<GateType, double> errorRates;

	double errorMultiplier;
	std::mt19937 gen;
	std::uniform_real_distribution<double> dist;
public:
	Noise(size_t N, unsigned int seed=42, double errorRateMultiplayer=1);

	// Reuses the caller's backend so GPU device state stays consistent.
	void applyPauliNoise(StateVector& sv, ICircuitUnitaryOperation& op);
	void accumulateError(GateType type, size_t target);

	void resetStats();

	size_t getTotalBitFlips() const;
	size_t getTotalPhaseFlips() const;
};

#endif
