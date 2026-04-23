#ifndef GROVER_H
#define GROVER_H
#include <vector>
class Circuit;

class Grover
{
public:
	// target : the marked item index to find (nQubits derived automatically)
	// shots  : number of samples for the result histogram
	static void runGrover(int target, int shots = 1000);
private:
	static void createGroverCircuit(Circuit& circuit, int nQubits, int target, int iterations);
	static void groverDiffusion(Circuit& circuit, int nQubits);
	static void groverOracle(Circuit& circuit, int nQubits, int target);
};

#endif
