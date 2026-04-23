#ifndef Shor_H
#define Shor_H
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>
#include "../QuantumSimEvo/Circuit.h"

class Shor
{
public:
	static void runShor(int N, int a);
private:
	//Extra Gates
	static void applyQFT(std::vector<size_t>& reg, Circuit& qc);
	static void applyIQFT(std::vector<size_t>& reg, Circuit& qc);
	static void addConstantQFT(std::vector<size_t>& reg, int a, Circuit& qc);
	static void cAddConstantQFT(size_t ctrl, std::vector<size_t>& reg, int a, Circuit& qc);

	//Middle layer
	static void cModAdd(size_t ctrl, std::vector<size_t>& reg, std::vector<size_t>& aux, int a, int N, Circuit& qc);
	static void cModMult(size_t ctrl, std::vector<size_t>& acc, std::vector<size_t>& aux, int a, int N, Circuit& qc);

	//Top layer
	static int modExp(int base, int exp, int mod);
	static std::vector<int> shorsAlgorithm2n3(int a, int N, Circuit& qc);

	//Helpers
	static int gcd(int a, int b);
	static double binaryFractionToDouble(const std::vector<int>& bits);
	static int getDenominator(double phase, int max_den);

};
#endif
