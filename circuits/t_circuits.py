
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

def t_circuit(n):
    # Create quantum circuit
    qc = QuantumCircuit(1, 1)
    qc.h(0)

    for _ in range(n):
        qc.t(0)
    
    qc.h(0)

    # Measure all qubits
    qc.measure(0,0)

    return qc


def main():
    parser = argparse.ArgumentParser(description="Run T Circuit.")
    
    # Optional argument to save QASM
    parser.add_argument(
        "-b", "--save_qasm", action="store_true", help="Save T QASM"
    )

    args = parser.parse_args()

    K = 6

    for n in range(1,K+1):

        print(f"Number of qubits: {n}\n")

        # Create the hidden shift circuit
        qc = t_circuit(n)

        print(f"Quantum Circuit:\n")
        print(qc,"\n")

        # Convert the circuit to QASM format if requested
        if args.save_qasm:
            qasm_code = qc.qasm()

            # Replace 'tdg' with 's; s; s; t;'
            modified_qasm_code = re.sub(r'\btdg\b', 's;\ns;\ns;\nt;', qasm_code)

            with open(f"./qasm_files/t_{n}.qasm", "w") as f:
                f.write(modified_qasm_code)

        # Simulate the circuit after decomposition
        simulator = Aer.get_backend('qasm_simulator')
        job = execute(qc, simulator, shots=4096)
        result = job.result()
        counts = result.get_counts()

        print("Qiskit Simulation Results:")
        #print(counts)
        percentages = [(k, v / sum(counts.values()))
                           for k, v in counts.items()]
        print(f"Percentages: {percentages}\n")


if __name__ == "__main__":
    main()
