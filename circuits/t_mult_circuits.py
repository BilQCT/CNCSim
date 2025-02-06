
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

def t_mult_circuit(n,m):
    # Create quantum circuit
    qc = QuantumCircuit(n, n)

    # round of hadamards
    qc.h(list(range(n)))

    # apply t gate m times:
    for _ in range(m):
        qc.t(list(range(n)))
    
    # round of hadamards
    qc.h(list(range(n)))

    # Measure all qubits
    for q in range(n):
        qc.measure(q,q)

    return qc


def main():
    parser = argparse.ArgumentParser(description="Run T Circuit.")
    
    # Optional argument to save QASM
    parser.add_argument(
        "-b", "--save_qasm", action="store_true", help="Save T QASM"
    )

    args = parser.parse_args()

    K = 8
    n = 2

    for m in range(K):

        print(f"Applied T gate: {m} times\n")

        # Create the hidden shift circuit
        qc = t_mult_circuit(n,m)

        #print(f"Quantum Circuit:\n")
        #print(qc,"\n")

        # Convert the circuit to QASM format if requested
        if args.save_qasm:
            qasm_code = qc.qasm()

            # Replace 'tdg' with 's; s; s; t;'
            modified_qasm_code = re.sub(r'\btdg\b', 's;\ns;\ns;\nt;', qasm_code)

            with open(f"./qasm_files/t_mult_n_{n}_m_{m}.qasm", "w") as f:
                f.write(modified_qasm_code)

        # Simulate the circuit after decomposition
        simulator = Aer.get_backend('qasm_simulator')
        job = execute(qc, simulator, shots=1024)
        result = job.result()
        counts = result.get_counts()

        print("Qiskit Simulation Results:")
        #print(counts)
        percentages = [(k, np.round(v / sum(counts.values()),2))
                           for k, v in counts.items()]
        print(f"Percentages: {percentages}\n")


if __name__ == "__main__":
    main()
