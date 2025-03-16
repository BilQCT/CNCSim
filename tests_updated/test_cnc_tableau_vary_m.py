import timeit
import numpy as np
import timeit
from random import choice

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('tests_updated/test_cnc_tableau.py'), '..')))

from src import chp
from src import cnc_simulator as cnc


# generate random cliffords:
def generate_gate_sequence(n, beta):
    """Generate a random sequence of Clifford gates."""
    num_gates = int(beta * n) #* np.log2(n)
    gates = ['h', 's', 'cnot']
    sequence = []
    for _ in range(num_gates):
        gate = choice(gates)
        if gate == 'h' or gate == 's':  # Single qubit gates
            qubit = np.random.randint(0, n)
            sequence.append(f"{gate}_{qubit}")
        elif gate == 'cnot':  # Two-qubit gate
            control = np.random.randint(0, n)
            target = np.random.randint(0, n)
            while target == control:  # Ensure control != target
                target = np.random.randint(0, n)
            sequence.append(f"{gate}_{control}_{target}")
    return sequence



# apply random gate sequence with cnc sim:
def apply_sequence_of_clifford(simulator, sequence):
    """Apply a sequence of Clifford gates to the simulator."""
    for gate in sequence:
        if gate.startswith('h'):
            qubit = int(gate.split('_')[1])
            simulator.apply_hadamard(qubit)
        elif gate.startswith('s'):
            qubit = int(gate.split('_')[1])
            simulator.apply_phase(qubit)
        elif gate.startswith('cnot'):
            control, target = map(int, gate.split('_')[1:])
            simulator.apply_cnot(control, target)



# Loop over beta and n ranges
#beta_min = 1.00; beta_max = 1.01; delta = 0.4
#beta_values = np.arange(beta_min, beta_max, delta)

# define qubit ranges:
n_min = 200; n_max = 1201; n_delta = 50
n_values = range(n_min, n_max, n_delta)  # n from 200 to 1000

results = []

for n in n_values:

    m_values = [1,np.int64(np.round(n/4,0)),np.int64(np.round(n/2,0)),np.int64(np.round((3*n)/4,0)),n]
    #print(n,m_values)

    for m in m_values:

        print(f"Running simulation for n={n}, m={m}:\n")

        # Generate Clifford gate sequence
        sequence = generate_gate_sequence(n, 1.0)

        # Initialize the CNC simulator
        simulator = cnc.CncSimulator(n, m)

        # Apply the gate sequence
        apply_sequence_of_clifford(simulator, sequence)

        # Prepare measurement bases
        zero_matrix = np.zeros((n, n), dtype=int)
        identity_matrix = np.eye(n, dtype=int)
        measurement_bases = np.hstack((identity_matrix,zero_matrix))

        # Define the setup code
        setup_code = f"""
import numpy as np
from __main__ import simulator, measurement_bases
"""

        # Define the statement to measure
        stmt_code = """
for base in range(measurement_bases.shape[0]):
    simulator.measure(measurement_bases[base, :])
"""

        # Time the execution
        repeat = 5
        execution_time = timeit.timeit(stmt=stmt_code, setup=setup_code, number=repeat)
        avg_time = execution_time /(repeat*n)
        print(f"Average time per iteration for n={n}, m={m}: {avg_time:.6f} seconds")

        # Store results
        results.append((n, m, avg_time))

# Print all results
for n, m, avg_time in results:
    print(f"n={n}, m={m}, Avg Time={avg_time:.6f} seconds")

# Save to .npy file
np.save(f"./figures/cnc_measurement_n_{n_min}_{n_max}_{n_delta}_m_1_n_beta_1.npy", np.array(results))



import numpy as np
import matplotlib.pyplot as plt

# Your numpy array
data = np.array(results)

# Extract unique beta values
m_values = np.unique(data[:, 1])

# Create the plot
plt.figure(figsize=(10, 6))

m_strings = ["1","n/4","n/2","3n/4","n"]

for m in range(5):
    subset_indices = [5*i+m for i in range(len(n_values))]
    subset = data[subset_indices]  # Filter rows with the current beta
    n = subset[:, 0]
    avg_time = subset[:, 2]
    plt.plot(n, avg_time, marker='o', label=f"m={m_strings[m]}")  # LaTeX syntax for beta

# Label the axes and add a legend
plt.xlabel("n (Number of Qubits)")
plt.ylabel("Average Time (s)")
plt.title(f"CNC: Measurement Time vs. n for Different m Values")  # Use beta in title
plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig(f"./figures/cnc_measurement_plot_n_{n_min}_{n_max}_{n_delta}_m_1_n_beta_1.png", format="png", dpi=300)

# Show the plot (optional)
plt.show()
