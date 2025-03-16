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
beta_min = 1.00; beta_max = 1.01; delta = 0.4
beta_values = np.arange(beta_min, beta_max, delta)

# define qubit ranges:
n_min = 200; n_max = 1001; n_delta = 50
n_values = range(n_min, n_max, n_delta)  # n from 200 to 1000

m = "vary"

results = []

for beta in beta_values:
    for n in n_values:
        print(f"Running simulation for beta={beta:.1f}, n={n}")

        # Generate Clifford gate sequence
        sequence = generate_gate_sequence(n, beta)

        # Initialize the CNC simulator
        em = int(n/2)
        simulator = cnc.CncSimulator(n, em)

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
        print(f"Average time per iteration for beta={beta:.1f}, n={n}: {avg_time:.6f} seconds")

        # Store results
        results.append((beta, n, avg_time))

# Print all results
for beta, n, avg_time in results:
    print(f"Beta={beta:.1f}, n={n}, Avg Time={avg_time:.6f} seconds")

# Save to .npy file
np.save(f"./figures/cnc_measurement_n_{n_min}_{n_max}_{n_delta}_m_{m}_beta_{beta_min}_{beta_max}_{delta}.npy", np.array(results))



import numpy as np
import matplotlib.pyplot as plt

# Your numpy array
data = np.array(results)

# Extract unique beta values
betas = np.unique(data[:, 0])

# Create the plot
plt.figure(figsize=(10, 6))

for beta in betas:
    subset = data[data[:, 0] == beta]  # Filter rows with the current beta
    n = subset[:, 1]
    avg_time = subset[:, 2]
    plt.plot(n, avg_time, marker='o', label=f"$\\beta={round(beta,1)}$")  # LaTeX syntax for beta

# Label the axes and add a legend
plt.xlabel("n (Number of Qubits)")
plt.ylabel("Average Time (s)")
plt.title(f"CNC: Measurement Time vs. n for Different $\\beta$ Values)")  # Use beta in title
plt.legend()
plt.grid(True)

# Save the plot to a file
plt.savefig(f"./figures/cnc_measurement_plot_n_{n_min}_{n_max}_{n_delta}_m_{m}_beta_{beta_min}_{beta_max}_{delta}.png", format="png", dpi=300)

# Show the plot (optional)
plt.show()
