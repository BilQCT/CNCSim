
import argparse
import re
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from ccz_7T_decomposition import *

# suppress qiskit 1.0 deprecation warnings:
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Example code that raises warnings
warnings.warn("This is a warning!")

def ccz_circuit(n):
    # Ensure n > 3:
    if n < 3:
        raise ValueError("Must have at least 3 qubits to apply CCZ.")
    
    # Create quantum circuit
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    qc.ccz(0,1,2)
    qc.h(range(n))

    # Measure all qubits
    qc.measure(range(n), range(n))

    return qc


def main():
    parser = argparse.ArgumentParser(description="Run Hidden Shift Algorithm.")
    
    # Optional argument to save QASM
    parser.add_argument(
        "-b", "--save_qasm", action="store_true", help="Save hidden shift QASM"
    )

    args = parser.parse_args()

    n = 3

    # Create the hidden shift circuit
    qc = ccz_circuit(n)

    print(f"Original Circuit:\n")
    print(qc,"\n")

    # Simulate the circuit after decomposition
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(qc, simulator, shots=1024)
    result = job.result()
    counts = result.get_counts()

    print("\nQiskit Simulation Results without Decomposition:")
    print(counts,"\n")

    # Apply the 7T decomposition if applicable
    qc_with_decomposition = apply_ccz_via_7t_decomposition(qc)

    print(f"Circuit with T-gate decomposition:\n")
    print(qc_with_decomposition,"\n")

    # Convert the circuit to QASM format if requested
    if args.save_qasm:
        qasm_code = qc_with_decomposition.qasm()

        # Replace 'tdg' with 's; s; s; t;'
        modified_qasm_code = re.sub(r'\btdg\b', 's;\ns;\ns;\nt;', qasm_code)

        with open(f"ccz_circuit_n_{n}.qasm", "w") as f:
            f.write(modified_qasm_code)

    # Simulate the circuit after decomposition
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(qc_with_decomposition, simulator, shots=1024)
    result = job.result()
    counts = result.get_counts()

    print("\nQiskit Simulation Results with Decomposition:")
    print("Percentages: ",[(list(counts.keys())[i],list(counts.values())[i]/sum(list(counts.values()))) for i in range(len(list(counts.values())))],"\n")


if __name__ == "__main__":
    main()
