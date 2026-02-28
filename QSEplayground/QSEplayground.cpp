#include "../QuantumSimEvo/Circuit.h"
#include "../QuantumSimEvo/HardwareCheck.h"

using namespace std;
int main()
{
    HardwareCheck::printReport();
    int arr[4] = { 0,0,0,0 };
    int n = 10;
    int qubits = 10;
    cout << "-----------------------------\n";
    cout << "Running " << n << " test circuits to verify the distribution of measurement outcomes...\n";

    Circuit qc(qubits);

    for (int i = 0; i < n; ++i) {

        cout << "Running test circuit " << (i + 1) << endl;

        //qc.pauliX(0);

        //qc.hadamard(0);

        //qc.cnot(0, 1);
		qc.hadamard(5);

        qc.pauliX(5);

        qc.cnot(5, 9);

        qc.execute(false);

        //qc.printState();
        
        qc.printState();
        size_t result = qc.measure();
        //arr[result]++;
		
        cout << "Measured State Index: " << result << endl;

        qc.reset();
    }
    cout << "-----------------------------\n";

    cout << "|00>: appears " << arr[0] << " times out of " << n << endl;
    cout << "|01>: appears " << arr[1] << " times out of " << n << endl;
    cout << "|10>: appears " << arr[2] << " times out of " << n << endl;
    cout << "|11>: appears " << arr[3] << " times out of " << n << endl;

}


