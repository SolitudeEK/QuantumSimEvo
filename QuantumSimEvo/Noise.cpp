#include "Noise.h"

Noise::Noise(size_t N, unsigned int seed, double errorRateMultiplier) :
	bitFlipCounts(N, 0),
	phaseFlipCounts(N, 0),
	errorMultiplier(errorRateMultiplier),
	gen(seed),
	dist(0.0, 1.0)
{
	errorRates[GateType::X] = 0.001;
	errorRates[GateType::Y] = 0.001;
	errorRates[GateType::Z] = 0.001;
	errorRates[GateType::H] = 0.002;
	errorRates[GateType::CNOT] = 0.005;
	errorRates[GateType::PHASE] = 0.001;
	errorRates[GateType::RX] = 0.001;
	errorRates[GateType::RY] = 0.001;
	errorRates[GateType::RZ] = 0.001;
}

void Noise::accumulateError(GateType type, size_t target)
{
	if (errorRates.find(type) == errorRates.end()) return;

	double p = errorRates[type] * errorMultiplier;

	if (dist(gen) < p)
		bitFlipCounts[target]++;
	if (dist(gen) < p)
		phaseFlipCounts[target]++;

}

void Noise::applyPauliNoise(StateVector& sv, ICircuitUnitaryOperation& op)
{
	for (size_t q = 0; q < bitFlipCounts.size(); q++)
	{
		if (bitFlipCounts[q] % 2 != 0)
			op.applyPauliX(sv, q);

		if (phaseFlipCounts[q] % 2 != 0)
			op.applyPauliZ(sv, q);
	}
}

void Noise::resetStats()
{
	std::fill(bitFlipCounts.begin(), bitFlipCounts.end(), 0);
	std::fill(phaseFlipCounts.begin(), phaseFlipCounts.end(), 0);
}

size_t Noise::getTotalBitFlips() const
{
	size_t total = 0;
	for (int count : bitFlipCounts) total += count;
	return total;
}

size_t Noise::getTotalPhaseFlips() const
{
	size_t total = 0;
	for (int count : phaseFlipCounts) total += count;
	return total;
}


