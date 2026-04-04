#ifndef QAOA_H
#define QAOA_H

#include <map>
#include <string>
#include <vector>
#include <utility>

class QAOA
{
private:
	static int calculateCutForMask(int mask, const std::vector<std::pair<int, int>>& edges);
	static std::map<int, int> computePartition(
		const std::vector<int>& nodes,
		const std::vector<std::pair<int, int>>& edges,
		const std::vector<std::size_t>& results);
	static int cutSize(
		const std::vector<std::pair<int, int>>& edges,
		const std::map<int, int>& partition);
public:
	static void bruteForceMaxCut(int n,
		const std::vector<std::pair<int, int>>& edges);
	static void runQAOA(int p,
		const std::vector<int>& nodes,
		const std::vector<std::pair<int, int>>& edges,
		const std::vector<float>& gamma,
		const std::vector<float>& beta,
		int shots = 1000);
};

#endif