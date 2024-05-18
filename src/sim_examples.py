from math import pi
import matplotlib.pyplot as plt;
import h5py
import numpy as np
from src.cnc import *
from qiskit import QuantumCircuit, Aer, execute


# General functions:


def xor_frequencies(dict):
    keys = dict.keys(); values = dict.values()
    xor_zero_freq = 0; xor_one_freq = 0
    for key in keys:
        xor = (sum([int(x,2) for x in list(key)])) % 2
        if xor == 0:
            xor_zero_freq += dict[key]
        else:
            xor_one_freq += dict[key]
    return [['0','1'],[xor_zero_freq,xor_one_freq]]

# Function to apply measurement in a specific basis
def apply_measurement(qc, qubit_index, basis):
    if basis == 'X':
        qc.h(qubit_index)
    elif basis == 'Y':
        qc.sdg(qubit_index)
        qc.h(qubit_index)

"""
Example 1: Classically siulating a quantum circuit for computing a nonlinear Boolean function:
"""

def qc_boolean_function():
    inputs = [[x,y] for x in range(2) for y in range(2)]
    outcome_counts = []
    for input in inputs:
        # Define inputs a and b
        a = input[0]  # Example input for qubit 1
        b = input[1]  # Example input for qubit 2

        # Create a quantum circuit with 3 qubits
        qc = QuantumCircuit(3, 3)  # 3 qubits and 3 classical bits for measurement results

        # Initialize qubits in |0> state (this is done by default)

        # Apply Hadamard gate to each qubit
        qc.h(range(3))

        # Apply controlled-Z gate from qubit 0 to qubit 1
        qc.cz(0, 1)

        # Apply controlled-Z gate from qubit 1 to qubit 2
        qc.cz(1, 2)

        # Determine measurement basis for qubit 1 based on input a
        if a == 0:
            measurement_basis_q1 = 'Y'
        else:
            measurement_basis_q1 = 'Z'

        # Determine measurement basis for qubit 2 based on input b
        if b == 0:
            measurement_basis_q2 = 'X'
        else:
            measurement_basis_q2 = 'Y'

        # Determine measurement basis for qubit 3 based on a + b mod 2
        if (a + b) % 2 == 0:
            measurement_basis_q3 = 'Y'
        else:
            measurement_basis_q3 = 'Z'

        # Measure each qubit in the determined basis
        apply_measurement(qc, 0, measurement_basis_q1)
        qc.measure(0, 0)  # Measure qubit 0 and store the result in classical bit 0
        apply_measurement(qc, 1, measurement_basis_q2)
        qc.measure(1, 1)  # Measure qubit 1 and store the result in classical bit 1
        apply_measurement(qc, 2, measurement_basis_q3)
        qc.measure(2, 2)  # Measure qubit 2 and store the result in classical bit 2
        print(f"Quantum circuit for inputs ({a},{b})")
        # Draw the circuit
        print(qc.draw(),"\n")

        # Simulate the circuit
        simulator = Aer.get_backend('qasm_simulator')
        job = execute(qc, simulator, shots=1000)
        result = job.result()

        # Get the counts
        counts = (result.get_counts(qc))
        outcome_counts.append(counts)

    xor_counts = [xor_frequencies(outcomes) for outcomes in outcome_counts]
    
    # Plot results:
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    ax = [[x,y] for x in [0,1] for y in [0,1]]

    for i in range(4):
        # Plot histograms
        axs[ax[i][0], ax[i][1]].bar(xor_counts[i][0], xor_counts[i][1])
        axs[ax[i][0], ax[i][1]].set_title(f'Input ({ax[i][0]},{ax[i][1]})')

    # Show plot
    plt.show()
    return qc


def magic_sim_boolean_function(initial_distribution,measurements,shots):
    outcome_counts = [simulate_from_distribution(initial_distribution, measurements[m], shots) for m in range(len(measurements))]
    xor_counts = [xor_frequencies(outcomes) for outcomes in outcome_counts]

    # Plot results:
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    ax = [[x,y] for x in [0,1] for y in [0,1]]

    for i in range(4):
        # Plot histograms
        axs[ax[i][0], ax[i][1]].bar(xor_counts[i][0], xor_counts[i][1])
        axs[ax[i][0], ax[i][1]].set_title(f'Input ({ax[i][0]},{ax[i][1]})')

    # Show plot
    plt.show()

    return outcome_counts




"""
Example 2: Classically simulate a single-qubit quantum circuit consisting of a single T-gate.
"""

def qc_HTH():
    # Create a quantum circuit with 1 qubit and 1 classical bit
    qc = QuantumCircuit(1, 1)

    # Initialize qubit in |0⟩ state
    # No need to explicitly initialize in |0⟩ state as it's the default state

    # Apply Hadamard gate
    qc.h(0)

    # Apply T gate (T gate is a rotation around Z-axis by π/4)
    qc.p(pi/4, 0)  # Equivalent to T gate

    # Apply Hadamard gate again
    qc.h(0)

    # Measure qubit in Z basis
    qc.measure(0, 0)

    # Draw the circuit
    print(qc.draw())

    # Simulate the circuit
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(qc, simulator, shots=1000)
    result = job.result()

    # Get the counts
    counts = result.get_counts(qc)

    # Plot the outcomes
    plt.bar([list(counts.keys())[1],list(counts.keys())[0]], [list(counts.values())[1],list(counts.values())[0]])
    #plt.bar(counts.keys(),counts.values())
    plt.xlabel('Outcome')
    plt.ylabel('Frequency')
    plt.title('Measurement Outcomes')
    plt.show()

def qc_magic_HTH():
    # Create a quantum circuit with 2 qubits and 1 classical bit
    qc = QuantumCircuit(2, 1)

    # Initialize qubit 1 in |0> state (this is done by default)

    # Initialize qubit 2 in |0> + exp(i*pi/4)|1> state
    #qc.u1(0.25 * 3.14159, 1)

    # Apply Hadamard gate to qubit 1
    qc.h(0)

    # Apply Hadamard gate to qubit 1
    qc.h(1)

    # Apply Z-rotation to qubit 1
    qc.p(pi/4,1)

    # Apply CNOT gate from qubit 1 to qubit 2
    qc.cx(0, 1)

    # Measure qubit 2 in Z basis and store the result in classical bit 0
    qc.measure(1, 0)

    # Conditional operation based on the measurement outcome
    qc.p(pi/2,0).c_if(0, 1)  # Apply S gate if outcome is 1, otherwise do nothing

    # Measure qubit 1 in X basis
    qc.h(0)

    # Measure the circuit
    qc.measure(0, 0)

    # Draw the circuit
    print(qc.draw())

    # Simulate the circuit
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(qc, simulator, shots=1000)
    result = job.result()

    # Get the counts
    counts = result.get_counts(qc)
    #print("Measurement results:", counts)

    # Plot the outcomes
    plt.bar([list(counts.keys())[1],list(counts.keys())[0]], [list(counts.values())[1],list(counts.values())[0]])
    plt.xlabel('Outcome')
    plt.ylabel('Frequency')
    plt.title('Measurement Outcomes')
    plt.show()

def qc_HTH_PBC():
    # Create a quantum circuit with 1 qubit and 1 classical bit
    qc = QuantumCircuit(1, 1)

    # Initialize qubit in the T state
    qc.h(0)
    qc.p(pi/4, 0)  # T gate equivalent to u3(pi/4, 0, 0)

    # Coin toss to decide the measurement basis
    coin_toss = 0#np.random.randint(2)  # Randomly choose 0 or 1
    if coin_toss == 0:
        # Measure qubit in X basis
        qc.h(0)  # Apply Hadamard gate to change to X basis
        qc.measure(0, 0)  # Measure qubit and store result in classical bit
    else:
        # Measure qubit in Y basis
        qc.sdg(0)  # Apply Sdg gate to change to Y basis
        qc.h(0)  # Apply Hadamard gate to change to Y basis
        qc.measure(0, 0)  # Measure qubit and store result in classical bit

    # Draw the circuit
    print(qc.draw())

    # Simulate the circuit
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(qc, simulator, shots=1000)
    result = job.result()

    # Get the counts
    counts = result.get_counts(qc)

    # Plot the outcomes
    plt.bar([list(counts.keys())[1],list(counts.keys())[0]], [list(counts.values())[1],list(counts.values())[0]])
    plt.xlabel('Outcome')
    plt.ylabel('Frequency')
    plt.title('Measurement Outcomes')
    plt.show()

def magic_sim_HTH(initial_distribution,measurements,shots):
    outcome_counts = [simulate_from_distribution(initial_distribution, measurements[m], shots) for m in range(len(measurements))]

    # Plot the outcomes
    plt.bar(outcome_counts.keys(), outcome_counts.values())
    plt.xlabel('Outcome')
    plt.ylabel('Frequency')
    plt.title('Measurement Outcomes')
    plt.show()

    return outcome_counts