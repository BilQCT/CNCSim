import numpy as np
import timeit
import matplotlib.pyplot as plt
from random import choice
import sys
import os
from test_functions import *

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('tests_updated/test_cnc_tableau.py'), '..')))

from src import cnc_simulator as cnc

##########################################
#                                        #
#      Varying beta, fixed m             #
#                                        #
##########################################

# Number of iterations for timing measurements
iterations = 1

# Define range of beta values and qubit counts (n)
beta_min = 0.6
beta_max = 1.21
delta = 0.2
beta_values = np.arange(beta_min, beta_max, delta)

n_min = 200
n_max = 2001   # n will range from 200 to 1200
n_delta = 200
n_values = range(n_min, n_max, n_delta)

# Strings used for documentation and plotting
m_doc_string = "n_over_2"
m_fig_string = "n/2"

print(f"Generating results for varying beta, fixed m:\n Min. beta = {beta_min}, Max. beta = {beta_max}\n Min. n = {n_min}, Max. n = {n_max}\n")

results = []  # List to store simulation results

# Loop over beta and n values to run simulations
for beta in beta_values:
    for n in n_values:
        print(f"Running simulation for beta={beta:.1f}, n={n}")
        # Generate a random sequence of Clifford gates for the current parameters
        sequence = generate_gate_sequence(n, beta)
        # Set m to be n/2
        m = int(n/2)
        # Initialize the CNC simulator with n qubits and m parameter
        simulator = cnc.CncSimulator(n, m)
        # Apply the generated gate sequence to the simulator
        apply_sequence_of_clifford(simulator, sequence)
        # Create measurement bases by horizontally concatenating an identity and zero matrix
        zero_matrix = np.zeros((n, n), dtype=int)
        identity_matrix = np.eye(n, dtype=int)
        measurement_bases = np.hstack((identity_matrix, zero_matrix))
        # Setup code string for the timeit module to import necessary objects
        setup_code = f"""
import numpy as np
from __main__ import simulator, measurement_bases
"""
        # Statement to be timed: perform measurement on each basis vector
        stmt_code = """
for base in range(measurement_bases.shape[0]):
    simulator.measure(measurement_bases[base, :])
"""
        # Time the execution and compute the average time per measurement
        repeat = iterations
        execution_time = timeit.timeit(stmt=stmt_code, setup=setup_code, number=repeat)
        avg_time = execution_time / (repeat * n)
        print(f"Average time per measurement ({repeat} iterations) for beta={beta:.1f}, n={n}: {avg_time:.6f} seconds \n")
        # Append the result as a tuple (beta, n, average_time)
        results.append((beta, n, avg_time))

# Save the results as a NumPy .npy file for later use
np.save(f"./figures/cnc_measurement_n_{n_min}_{n_max}_{n_delta}_m_{m_doc_string}_beta_{beta_min}_{beta_max}_{delta}.npy", np.array(results))

##########################################
# Print and plot the results
##########################################

data = np.array(results)
# Get unique beta values from the data
betas = np.unique(data[:, 0])

plt.figure(figsize=(10, 6))
# Plot average time vs. n for each beta value
for beta in betas:
    subset = data[data[:, 0] == beta]
    n_vals = subset[:, 1]
    avg_times = subset[:, 2]
    plt.plot(n_vals, avg_times, marker='o', label=f"$\\beta={round(beta,1)}$")
plt.xlabel("n (Number of Qubits)")
plt.ylabel("Average Time (s)")
plt.title(f"CNC: Measurement Time vs. n for Different $\\beta$ Values (m={m_fig_string})")
plt.legend()
plt.grid(True)
plt.savefig(f"./figures/cnc_measurement_plot_n_{n_min}_{n_max}_{n_delta}_m_{m_doc_string}_beta_{beta_min}_{beta_max}_{delta}.png", format="png", dpi=300)
plt.show()
