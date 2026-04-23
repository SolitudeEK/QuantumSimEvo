#include "QAOA.h"

#include <bitset>
#include <iostream>
#include <unordered_set>

#include "../QuantumSimEvo/Circuit.h"

void QAOA::bruteForceMaxCut(int n, const std::vector<std::pair<int, int>>& edges)
{
    size_t maxCutSize = 0;
    size_t bestMask = 0;

    size_t totalCombinations = 1LL << (n - 1);

    std::cout << "Checking " << totalCombinations << " combinations with OpenMP..." << '\n';

	#pragma omp parallel
    {
        size_t localMaxCut = 0;
        size_t localBestMask = 0;

		#pragma omp for
        for (std::ptrdiff_t i = 0; i < totalCombinations; ++i) {
            int currentCut = calculateCutForMask(i, edges);

            if (currentCut > localMaxCut) {
                localMaxCut = currentCut;
                localBestMask = i;
            }
        }

		#pragma omp critical
        {
            if (localMaxCut > maxCutSize) {
                maxCutSize = localMaxCut;
                bestMask = localBestMask;
            }
        }
    }

    std::cout << "--- Optimal Solution Found ---" << '\n';
    std::cout << "Maximum possible cut size: " << maxCutSize << '\n';
    std::cout << "One optimal partition (Bitmask): ";
    for (int i = 0; i < n; ++i) {
        std::cout << ((bestMask >> i) & 1);
    }
    std::cout << "\n------------------------------" << '\n';
}

int QAOA::calculateCutForMask(int mask, const std::vector<std::pair<int, int>>& edges)
{
    int cut = 0;
    for (const auto& edge : edges) {
        int u = edge.first;
        int v = edge.second;

        if (((mask >> u) & 1) != ((mask >> v) & 1)) {
            cut++;
        }
    }
    return cut;
}

std::map<int, int> QAOA::computePartition(
    const std::vector<int>& nodes,
    const std::vector<std::pair<int, int>>& edges,
    const std::vector<std::size_t>& results)
{
    std::unordered_set<size_t> unique_results(results.begin(), results.end());

    size_t best_mask = 0;
    int max_observed_cut = -1;

    for (size_t mask : unique_results) {
        int current_cut = calculateCutForMask(mask, edges);

        if (current_cut > max_observed_cut) {
            max_observed_cut = current_cut;
            best_mask = mask;
        }
    }

    std::map<int, int> partition;
    for (int node : nodes) {
        partition[node] = static_cast<int>((best_mask >> node) & 1);
    }

    return partition;
}

int QAOA::cutSize(
    const std::vector<std::pair<int, int>>& edges,
    const std::map<int, int>& partition)
{
    int count = 0;
    for (const auto& [u, v] : edges) {
        if (partition.at(u) != partition.at(v)) {
            count++;
        }
    }
    return count;
}

void QAOA::runQAOA(int p,
    const std::vector<int>& nodes,
    const std::vector<std::pair<int, int>>& edges, 
    const std::vector<float>& gamma, 
    const std::vector<float>& beta,
    int shots)
{
    if (gamma.size() != p || beta.size() != p) {
	    std::cout << "Error: gamma and beta vectors must have length equal to p.\n";
        return;
    }

	size_t n = nodes.size();
    std::cout << "Creating a QAOA circuit with " << n << " qubits and " << p << " layers...\n";
    Circuit circuit(n);

    for (size_t i = 0; i < n; ++i)
    {
        circuit.hadamard(i);
    }

    for (size_t i = 0; i < p; ++i)
    {
        for (auto edge : edges)
        {
            circuit.cnot(edge.first, edge.second);
            circuit.rotateZ(4 * gamma[i], edge.second);
            circuit.cnot(edge.first, edge.second);
        }

        for (size_t j = 0; j < n; j++)
        {
            circuit.rotateX(2 * beta[i], j);
        }
    }

    circuit.execute(false);
	circuit.printState();

	auto results = circuit.sample(shots);

    auto partition = computePartition(nodes, edges, results);
    auto partition_cut_size = cutSize(edges, partition);

    std::cout << "Partition results:" << '\n';
    for (auto const& [node, side] : partition) {
	    std::cout << "Node " << node << ": Group " << side << "\n";
    }
    std::cout << "Total cut size: " << partition_cut_size << '\n';
}
