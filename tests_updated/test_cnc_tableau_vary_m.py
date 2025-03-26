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
#      Fixed beta, varying m           #
##########################################

iterations = 1

# Fix beta value for these simulations
beta = 1
# Define segmentation factor k for m
k = 4
n_min = 200
n_max = 2401   # n from 200 to 2400
n_delta = 200
n_values = range(n_min, n_max, n_delta)

repeat = 3  # number of repetitions

print(f"Generating results for varying m by increments of n/{k}\n Min. n = {n_min}, Max. n = {n_max}\n")

results = []  # Reset results list

for n in n_values:
    # Generate a random sequence of Clifford gates for fixed beta
    sequence = generate_gate_sequence(n, beta)
    # Define m values: 0, 1, and then int(r*n/k) for r = 1 to k
    m_values = [0, 1] + [int(r * n / k) for r in range(1, k+1)]
    for m in m_values:
        print(f"Running simulation for m={m}, n={n}")
        # Generate a random sequence of Clifford gates for the current parameters
        sequence = generate_gate_sequence(n, beta)
        
        # Time the full simulation (initialization + measurement)
        full_time = timeit.timeit(lambda: simulate_full(n, beta, m), number=repeat)
        
        # Time the initialization only
        init_time = timeit.timeit(lambda: simulate_init(n, beta, m), number=repeat)
        
        # The measurement time is full_time minus init_time
        measurement_time = full_time - init_time
        # Calculate average time per measurement by dividing by number of measurement operations (n)
        avg_time = measurement_time / (n*repeat)
        
        print(f"Full simulation time: {full_time:.6f} s, Initialization time: {init_time:.6f} s")
        print(f"Adjusted measurement time per measurement for beta={beta:.1f}, n={n}: {avg_time:.6f} seconds\n")
        
        results.append((beta, n, m, avg_time))

# Save the results for fixed beta and varying m
np.save(f"./figures/cnc_measurement_n_{n_min}_{n_max}_{n_delta}_m_vary_{k}_beta_{beta}.npy", np.array(results))

##########################################
# Plot results grouped by m values
##########################################

# Convert results to a numpy array
data = np.array(results)
# Group data by beta value and then by m groups:
k = 4  # number of segments for m grouping
groups = {i: [] for i in range(0, k+2)}
for beta, n, m, avg_time in results:
    if m == 0:
        group = 0
    elif m == 1:
        group = 1
    else:
        # Determine group based on integer division of n
        for r in range(2, k+2):
            if m == int(r * n / k):
                group = r
                break
        else:
            continue
    groups[group].append((n, avg_time))

# Sort each group's data by n
for key in groups:
    groups[key] = np.array(sorted(groups[key], key=lambda x: x[0]))

# Define group labels (adjust for k = 4)
labels = ["m = 0", "m = 1", "m = n/2", "m = 3n/4", "m = n", "Extra"]

plt.figure(figsize=(10, 6))
for i in range(k+2):
    if groups[i].size > 0:
        plt.plot(groups[i][:, 0], groups[i][:, 1], marker='o', label=labels[i])
plt.xlabel("n (Number of Qubits)")
plt.ylabel("Average Time per Measurement (s)")
plt.title("CNC Simulation: Average Time vs n (Varying m)")
plt.legend()
plt.grid(True)
plt.savefig(f"./figures/cnc_measurement_plot_n_{n_min}_{n_max}_{n_delta}_m_{k}_beta_{beta}.png", format="png", dpi=300)
plt.show()
