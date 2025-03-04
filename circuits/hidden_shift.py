
import argparse
import re
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from .ccz_7T_decomposition import *
import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('circuits/hidden_shift.py'), '..')))

# suppress qiskit 1.0 deprecation warnings:
import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")
# Example code that raises warnings
warnings.warn("This is a warning!")



#########################################################################


def to_bit_string(bit_list):
    bit_string = ""
    for x in bit_list:
        bit_string += str(x)
    return int(bit_string, 2)


# default clifford oracle composed of CZ,Z gates:
def clifford_oracle(qc,ni,nf):

    # Apply controlled-Z on nearest neighbor:
    for q in range(ni,nf-1):
        qc.h(q+1)
        qc.cx(q, q + 1)
        qc.h(q+1)
        qc.s(q)
        qc.s(q)
    
    # apply final round of Z:
    qc.s(nf)
    qc.s(nf)


# default non_clifford_oracle with O(n) CCZ gates
def non_clifford_oracle(qc,ni,nf):

    # Apply controlled-Z on nearest neighbor:
    for q in range(ni,nf):
        qc.s(q)
        qc.s(q)
        if q < nf:
            qc.h(q+1)
            qc.cx(q, q + 1)
            qc.h(q+1)
        if q < nf-1:
            qc.ccz(q,q+1,q+2)
    


def hidden_shift(m,shift,is_clifford = False):
    n = 2 * m

    # Create quantum circuit
    qc = QuantumCircuit(n, n)
    qc.h(range(n))

    # Apply controlled-Z
    for q in range(m):
        qc.h(q+m)
        qc.cx(q, q + m)
        qc.h(q+m)

    # Oracle on first m qubits:
    if is_clifford:
        clifford_oracle(qc,0,m-1)
    else:
        non_clifford_oracle(qc,0,m-1)
    
    # Apply Hadamards
    qc.h(range(n))

    # Apply controlled-Z
    for q in range(m):
        qc.h(q+m)
        qc.cx(q, q + m)
        qc.h(q+m)

    # Oracle on first m qubits:
    if is_clifford:
        clifford_oracle(qc,m,n-1)
    else:
        non_clifford_oracle(qc,m,n-1)

    # Apply Z gates based on the hidden shift
    for q in range(n):
        if shift[q] == 1:
            qc.s(q)
            qc.s(q)
    
    # Final round of Hadamards
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
    
    # Optional flag to specify if the oracle is Clifford
    parser.add_argument(
        "-c", "--is_clifford", action="store_true", help="Use Clifford oracle"
    )
    
    # Required argument 'm'
    parser.add_argument(
        "m", type=int, help="Value of m for the hidden shift algorithm"
    )

    args = parser.parse_args()

    n = 2 * args.m

    # Generate a random hidden shift
    shift = [np.random.choice(2) for _ in range(n)]

    # Create the hidden shift circuit
    qc = hidden_shift(args.m, shift, is_clifford=args.is_clifford)

    #print(f"Original Circuit:\n")
    #print(qc,"\n")

    # Apply the 7T decomposition if applicable
    qc_with_decomposition = apply_ccz_via_7t_decomposition(qc)

    #print(f"Circuit with T-gate decomposition:\n")
    #print(qc_with_decomposition,"\n")

    modified_qasm_code = None

    # Convert the circuit to QASM format if requested
    if args.save_qasm:
        qasm_code = qc_with_decomposition.qasm()

        # Replace 'tdg' with 's; s; s; t;'
        modified_qasm_code = re.sub(r'\btdg\b', 's;\ns;\ns;\nt;', qasm_code)

        # Save the modified QASM file
        if args.is_clifford:
            with open(f"hidden_shift_n_{n}_shift_{to_bit_string(shift)}_cliff.qasm", "w") as f:
                f.write(modified_qasm_code)
        else:
            with open(f"hidden_shift_n_{n}_shift_{to_bit_string(shift)}.qasm", "w") as f:
                f.write(modified_qasm_code)

    # Simulate the circuit after decomposition
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(qc_with_decomposition, simulator, shots=1024)
    result = job.result()
    counts = result.get_counts()

    print(f"\nHidden shift: {shift}")
    print("\nQiskit Simulation Results with Decomposition:")
    print("Percentages: ",[(list(counts.keys())[i],list(counts.values())[i]/sum(list(counts.values()))) for i in range(len(list(counts.values())))],"\n")

    filename = f"hidden_shift_n_{n}_shift_{to_bit_string(shift)}_cliff"
    qasm_file = filename+".qasm"

    # filename for generated msi clifford circuit:
    clifford_filename = filename+"_msi.qasm"

    #qcm.run_qcm("./",
    #        qasm_file,
    #        clifford_filename,
    #        shots=1024)


if __name__ == "__main__":
    main()
