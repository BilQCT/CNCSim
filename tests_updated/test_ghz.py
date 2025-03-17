import numpy as np
from random import choice
import time
from test_functions import *
import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('tests_updated/test_cnc_tableau.py'), '..')))

from src import updated_cnc_tableau as cnc

# define qubit ranges:
n_min = 200; n_max = 801; n_delta = 50
n_values = range(n_min, n_max, n_delta)  # n from 200 to 1000

# Initialize list to store measurement times
measurement_times = []
avg_measurement_times = []

# check results for range of qubit values:
for n in n_values: 
    print(f"Performing GHZ test for n={n} qubits:\n")
    
    # fix n-qubits:
    m = 1

    # initialize measurements:
    measurements = measurement_bases(n)

    print("Measuring qubits.\n")
    
    # initialize empty list for outcomes and total time:
    outcomes_array = []
    total_time = 0.0
    
    for i in range(n):
        print(f"Measuring stabilizer {i+1}:")

        # initialize simulator:
        simulator = cnc.CncSimulator(n, m)

        # prepare GHZ state
        ghz_state_prep(simulator, n)

        bases = measurements[i, :, :]
        outcomes = []

        # Start timing the measurement round
        start_time = time.perf_counter()

        for j in range(n):
            outcome = simulator.measure(bases[j, :])
            outcomes.append(outcome)

        # Stop timing and accumulate total time
        round_time = time.perf_counter() - start_time
        total_time += round_time

        measurement_times.append((n, i, round_time))

        print(f"Round of measurements in {round_time} seconds.\n")

        outcomes_array.append(outcomes)

    # check results:
    mapping = ghz_function(n)
    xor_outcomes = [sum(outcomes_array[k]) % 2 for k in range(n)]
    bool_list = [mapping[k] == xor_outcomes[k] for k in range(n)]

    # print results
    print(f"Simulation for {n} qubits gives correct result:", all(bool_list))

    # Compute and print average time per round
    avg_time = total_time / (n*n)
    avg_measurement_times.append((n,avg_time))
    print(f"Average time per measurement round for n={n}: {avg_time:.6f} seconds\n")


# Convert results to numpy array and save
measurement_times_array = np.array(measurement_times)
np.save("measurement_times.npy", measurement_times_array)

# Convert results to numpy array and save
avg_measurement_times_array = np.array(avg_measurement_times)
np.save("avg_measurement_times.npy", avg_measurement_times_array)