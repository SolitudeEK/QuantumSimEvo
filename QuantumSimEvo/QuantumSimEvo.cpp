#include "HardwareCheck.h"
#include "StateVector.h"
#include "Circuit.h"

using namespace std;
int main()
{
    HardwareCheck::printReport();
	int arr[4] = { 0,0,0,0 };
    int n = 1;
    int qubits = 30;

    cout << "-----------------------------\n";
	cout << "Running " << n << " test circuits to verify the distribution of measurement outcomes...\n";

    Circuit qc(qubits);

    for (int i = 0; i < n; ++i) {

        cout << "Running test circuit " << (i + 1) << endl;

        //std::cout << "Applying PauliX gate to qubit 0...\n";
        qc.pauliX(0);

        //std::cout << "Applying Hadamard gate to qubit 0...\n";
        qc.hadamard(0);

        //std::cout << "Applying CNOT gate (control: qubit 0, target: qubit 1)...\n";
        qc.cnot(0, 1);

        qc.execute(false);

        size_t result = qc.measure();
		arr[result]++;
        cout << "Measured State Index: " << result << endl;
        //qc.printState();

		qc.reset();
    }
    cout << "-----------------------------\n";

    cout << "|00>: appears " << arr[0] << " times out of " << n << endl;
    cout << "|01>: appears " << arr[1] << " times out of " << n << endl;
    cout << "|10>: appears " << arr[2] << " times out of " << n << endl;
    cout << "|11>: appears " << arr[3] << " times out of " << n << endl;
}
