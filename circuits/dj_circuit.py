
import argparse
import re
import numpy as np
from qiskit import QuantumCircuit, Aer, execute
from .toff_7T_decomposition import *

# suppress qiskit 1.0 deprecation warnings:
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Example code that raises warnings
warnings.warn("This is a warning!")


def create_constant_oracle(n_qubits, output):
    """
    Creates a 'constant' oracle.

    If `output` is 0, the oracle always returns 0.
    If `output` is 1, the oracle always returns 1.

    Args:
        n_qubits (int): The number of input qubits.
        output (int): The constant output value of the function (0 or 1).

    Returns:
        QuantumCircuit: A quantum circuit implementing the constant oracle.
    """
    oracle = QuantumCircuit(n_qubits)

    # If the oracle should always output 1, we flip the "output" qubit
    # using an X-gate (think of it as a NOT gate on a qubit).
    if output == 1:
        oracle.x(n_qubits-1)

    return oracle

def create_balanced_oracle(toff_count, n_qubits):
    """
    Creates a 'balanced' oracle.

    Half of the input bit patterns output 0, and the other half output 1.

    Args:
        toff_count (int): The number of Toffoli gates in the desired oracle.
        n_qubits (int): The number of qubits int the oracle.

    Returns:
        QuantumCircuit: A quantum circuit implementing the balanced oracle.
    """
    
    oracle=QuantumCircuit(n_qubits)
    toff_part=3*toff_count+1
    if n_qubits>=toff_part:
        for i in range(0,toff_part-1,3):
            oracle.ccx(i, i+1, i+2)
            oracle.cx(i+2, n_qubits-1)
        for i in range(toff_part-1,n_qubits-1,1):
            oracle.cx(i,n_qubits-1)

        return oracle
    else:
        print("The total qubit count must be greater or equal to three times the number of Toffoli gates plus one.")




def deutsch_jozsa_circuit(oracle, n_qubits):
    """
    Assembles the full Deutsch-Jozsa quantum circuit.

    The circuit performs the following steps:
    1. Start all 'input' qubits in |0>.
    2. Start the 'output' qubit in |1>.
    3. Apply Hadamard gates to all qubits.
    4. Apply the oracle.
    5. Apply Hadamard gates again to the input qubits.
    6. Measure the input qubits.

    Args:
        oracle (QuantumCircuit): The circuit encoding the 'mystery' function f(x).
        n_qubits (int): The number of input qubits.

    Returns:
        QuantumCircuit: The complete Deutsch-Jozsa circuit ready to run.
    """
    # Total of n_qubits for input, plus 1 for the output qubit
    dj_circuit = QuantumCircuit(n_qubits, n_qubits)

    # 1. The input qubits are already set to |0>.
    # 2. The output qubit is set to |1>. We achieve this by an X gate.
    dj_circuit.h(n_qubits-1)
    dj_circuit.s(n_qubits-1)
    dj_circuit.s(n_qubits-1)
    dj_circuit.h(n_qubits-1)

    # 3. Apply Hadamard gates to all qubits (input + output).
    for qubit in range(n_qubits):
        dj_circuit.h(qubit)

    # 4. Append the oracle circuit.
    dj_circuit.compose(oracle, inplace=True)

    # 5. Apply Hadamard gates again to the input qubits ONLY.
    for qubit in range(n_qubits-1):
        dj_circuit.h(qubit)

    # 6. Finally, measure the input qubits.
    for qubit in range(n_qubits-1):
        dj_circuit.measure(qubit, qubit)

    return dj_circuit


def run_deutsch_jozsa_test(n_qubits,toff_count,oracle_type,constant_output):
    """
    Builds and runs the Deutsch-Jozsa circuit for either a constant oracle
    or a balanced oracle, then prints the results.

    Args:
        n_qubits (int): Number of input qubits.
        oracle_type (str): Specifies the type of oracle, either 'constant' or 'balanced'.
        constant_output (int): If the oracle is constant, determines whether it returns 0 or 1.
    """
    # Create the chosen oracle
    if oracle_type == 'constant':
        oracle = create_constant_oracle(n_qubits, constant_output)
    else:
        oracle = create_balanced_oracle(toff_count,n_qubits)

    # Create the Deutsch-Jozsa circuit
    dj_circ = deutsch_jozsa_circuit(oracle, n_qubits)
    dj_circ = apply_toff_via_7t_decomposition(dj_circ)
    
    return dj_circ