#include <bitset>
#include<numbers>
#include <unordered_map>
#include "../QuantumSimEvo/Circuit.h"
#include "../QuantumSimEvo/HardwareCheck.h"
#include "QAOA.h"

using namespace std;

int main()
{
    vector<float> gamma { numbers::pi / 8, numbers::pi / 8, numbers::pi / 8 };
    vector<float> beta { numbers::pi / 3, numbers::pi / 3, numbers::pi / 3 };
	vector<int> nodes{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 };
    //vector<pair<int, int>> edges{
    //{0,1}, {0,2}, {0,3}, {1,2}, {1,3}, {2,3}, {2,4}, {3,5},
    //{4,5}, {4,6}, {4,7}, {5,6}, {5,7}, {6,7}, {6,8}, {7,9},
    //{8,9}, {8,10}, {8,11}, {9,10}, {9,11}, {10,11} };
    std::vector<std::pair<int, int>> edges{
        {0,1},  {0,2},  {0,3},  {1,2},  {1,3},  {2,3},  {2,4},  {3,5},
        {4,5},  {4,6},  {4,7},  {5,6},  {5,7},  {6,7},  {6,8},  {7,9},
        {8,9},  {8,10}, {8,11}, {9,10}, {9,11}, {10,11}, {10,12}, {11,13},
        {12,13}, {12,14}, {12,15}, {13,14}, {13,15}, {14,15}
    };
    size_t p = 3;

	QAOA::runQAOA(p, nodes, edges, gamma, beta, 100);
	QAOA::bruteForceMaxCut(nodes.size(), edges);
	return 0;
}

void bellState()
{
    HardwareCheck::printReport();
    int arr[4] = { 0,0,0,0 };
    int n = 1000;
    int qubits = 2;
    cout << "-----------------------------\n";
    cout << "Running " << n << " test circuits to verify the distribution of measurement outcomes...\n";

    Circuit qc(qubits, true);

    for (int i = 0; i < n; ++i) {

        cout << "Running test circuit " << (i + 1) << endl;

        qc.pauliX(0);

        qc.hadamard(0);
        qc.cnot(0, 1);
        qc.execute(false);


        qc.printState();

        size_t result = qc.measure();

        arr[result]++;

        cout << "Measured State Index: " << result << endl;

        qc.reset();
    }
    cout << "-----------------------------\n";

    cout << "|00>: appears " << arr[0] << " times out of " << n << endl;
    cout << "|01>: appears " << arr[1] << " times out of " << n << endl;
    cout << "|10>: appears " << arr[2] << " times out of " << n << endl;
    cout << "|11>: appears " << arr[3] << " times out of " << n << endl;
}

void ghz()
{
    int n = 25;
    int k = 1000;

    auto distribution = new map<size_t, int>();

    Circuit qc(n, false);

    cout << "-----------------------------\n";
    cout << "Running " << k << " test circuits to verify the distribution of measurement outcomes...\n";

    for (int i = 0; i < k; ++i)
    {
        cout << "Running test circuit " << (i + 1) << endl;

        qc.hadamard(0);

        for (int j = 0; j < n - 1; ++j)
        {
            qc.cnot(j, j + 1);
        }
        qc.execute();

        qc.printState();

        size_t result = qc.measure();

        (*distribution)[result]++;

        cout << "Measured State Index: " << result << endl;

        qc.reset();
    }

    cout << "-----------------------------\n";
    for (const auto& element : *distribution)
    {
        cout << "|" << bitset<64>(element.first).to_string().substr(64 - n) << "> appeared " << element.second << " times out of " << k << endl;
    }
}


