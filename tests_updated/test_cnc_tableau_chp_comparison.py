import numpy as np
import timeit
import matplotlib.pyplot as plt
from random import choice
import sys
import os
from test_functions import *
import chp as chp

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('tests_updated/test_cnc_tableau.py'), '..')))

from src import cnc_simulator as cnc

##########################################
#          Comparison with CHP           #
#          Varying beta, m = 0           #
##########################################

iterations = 1

# Generate CHP simulation results for varying beta, fixed m
beta_min = 0.6
beta_max = 1.21
delta = 0.2
beta_values = np.arange(beta_min, beta_max, delta)

n_min = 200
n_max = 2401
n_delta = 200
n_values = range(n_min, n_max, n_delta)

print(f"Generating CHP results for varying beta, fixed m:\n Min. beta = {beta_min}, Max. beta = {beta_max}\n Min. n = {n_min}, Max. n = {n_max}\n")

chp_results = []

for beta in beta_values:
    for n in n_values:
        print(f"Running simulation for beta={beta:.1f}, n={n}")
        sequence = generate_gate_sequence(n, beta)
        simulator = chp.ChpSimulator(n)
        apply_sequence_of_clifford_chp(simulator, sequence)
        setup_code = f"""
import numpy as np
from __main__ import simulator, n
"""
        stmt_code = """
for qubit in range(n):
    simulator.measure(qubit)
"""
        repeat = iterations
        execution_time = timeit.timeit(stmt=stmt_code, setup=setup_code, number=repeat)
        avg_time = execution_time / (repeat * n)
        print(f"Average time per measurement ({repeat} iterations) for beta={beta:.1f}, n={n}: {avg_time:.6f} seconds \n")
        chp_results.append((beta, n, avg_time))

np.save(f"./figures/chp_measurement_n_{n_min}_{n_max}_{n_delta}_beta_{beta_min}_{beta_max}_{delta}.npy", np.array(chp_results))

##########################################
# CNC results for CHP comparison (fixed m = 0)
##########################################

m = 0
cnc_results = []

for beta in beta_values:
    for n in n_values:
        print(f"Running simulation for beta={beta:.1f}, n={n}")
        sequence = generate_gate_sequence(n, beta)
        simulator = cnc.CncSimulator(n, m)
        apply_sequence_of_clifford(simulator, sequence)
        zero_matrix = np.zeros((n, n), dtype=int)
        identity_matrix = np.eye(n, dtype=int)
        measurement_bases = np.hstack((identity_matrix, zero_matrix))
        setup_code = f"""
import numpy as np
from __main__ import simulator, measurement_bases
"""
        stmt_code = """
for base in range(measurement_bases.shape[0]):
    simulator.measure(measurement_bases[base, :])
"""
        repeat = iterations
        execution_time = timeit.timeit(stmt=stmt_code, setup=setup_code, number=repeat)
        avg_time = execution_time / (repeat * n)
        print(f"Average time per measurement ({repeat} iterations) for beta={beta:.1f}, n={n}: {avg_time:.6f} seconds \n")
        cnc_results.append((beta, n, avg_time))

np.save(f"./figures/cnc_measurement_n_{n_min}_{n_max}_{n_delta}_m_0_beta_{beta_min}_{beta_max}_{delta}.npy", np.array(cnc_results))

##########################################
# Plot comparison between CHP and CNC models
##########################################

chp_data = np.array(chp_results)
cnc_data = np.array(cnc_results)

betas_chp = np.unique(chp_data[:, 0])
betas_cnc = np.unique(cnc_data[:, 0])

plt.figure(figsize=(12, 6))

# Plot CHP model results
plt.subplot(1, 2, 1)
for beta in betas_chp:
    subset = chp_data[chp_data[:, 0] == beta]
    n_vals = subset[:, 1]
    avg_times = subset[:, 2]
    plt.plot(n_vals, avg_times, marker='o', label=f"$\\beta={round(beta,1)}$")
plt.xlabel("n (Number of Qubits)")
plt.ylabel("Average Time (s)")
plt.title("CHP Model")
plt.legend()
plt.grid(True)

# Plot CNC model results
plt.subplot(1, 2, 2)
for beta in betas_cnc:
    subset = cnc_data[cnc_data[:, 0] == beta]
    n_vals = subset[:, 1]
    avg_times = subset[:, 2]
    plt.plot(n_vals, avg_times, marker='o', label=f"$\\beta={round(beta,1)}$")
plt.xlabel("n (Number of Qubits)")
plt.ylabel("Average Time (s)")
plt.title("CNC Model")
plt.legend()
plt.grid(True)

plt.suptitle("CHP vs. CNC: Measurement Times vs. n for Different $\\beta$ Values")
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("./figures/measurement_times_comparison.png", dpi=300)
plt.show()