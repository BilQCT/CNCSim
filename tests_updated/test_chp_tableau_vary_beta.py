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
#          Simulation Timing             #
#                                        #
##########################################

results = []  # List to store simulation results

# Define parameter ranges
beta_min = 0.6
beta_max = 1.21
delta = 0.2
beta_values = np.arange(beta_min, beta_max, delta)

n_min = 200
n_max = 2401   # n from 200 to 1200
n_delta = 200
n_values = range(n_min, n_max, n_delta)

for beta in beta_values:
    for n in n_values:
        print(f"Running simulation for beta={beta:.1f}, n={n}")
        repeat = 3  # Number of repetitions

        # Generate a random sequence of Clifford gates for the current parameters
        sequence = generate_gate_sequence(n, beta)
        
        # Time the full simulation (initialization + measurement)
        full_time = timeit.timeit(lambda: simulate_full_chp(n, beta), number=repeat)
        
        # Time the initialization only
        init_time = timeit.timeit(lambda: simulate_init_chp(n, beta), number=repeat)
        
        # The measurement time is full_time minus init_time
        measurement_time = full_time - init_time
        # Calculate average time per measurement by dividing by number of measurement operations (n)
        avg_time = measurement_time / (n*repeat)
        
        print(f"Full simulation time: {full_time:.6f} s, Initialization time: {init_time:.6f} s")
        print(f"Adjusted measurement time per measurement for beta={beta:.1f}, n={n}: {avg_time:.6f} seconds\n")
        
        results.append((beta, n, avg_time))

# Optionally, save results to a file
np.save(f"./figures/chp_measurement_adjusted_n_{n_min}_{n_max}_{n_delta}_beta_{beta_min}_{beta_max}_{delta}.npy", np.array(results))



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
plt.title(f"CHP: Measurement Time vs. n for Different $\\beta$ Values")
plt.legend()
plt.grid(True)
plt.savefig(f"./figures/chp_measurement_plot_n_{n_min}_{n_max}_{n_delta}_beta_{beta_min}_{beta_max}_{delta}.png", format="png", dpi=300)
plt.show()
