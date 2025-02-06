import numpy as np
from random import choice
import time
import os
import sys

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('tests_updated/test_cnc_tableau.py'), '..')))

from src import updated_cnc_tableau as cnc


"""
GHZ Test:
n-qubit GHZ state has stabilizers: X1...Xn, -Y1Y2...Xn, -Y1X2Y3..Xn, etc.
Meausuring bases: X1,...,Xn; Y1,Y2,...,Xn; Y1,X2,...,Xk-1,Yk,Xk+1,...,Xn, etc.
GHZ: Z2^n ->Z2: (0...0) \mapsto 0, otherwise: (1,1,0,...,0) \mapsto 1, (1,0,...,0,1,0,...,0) \maps 1, etc.

Since the local measurements always anticommute with at least one stabilizer, this algorithm only ever uses the Case IV updates. 

"""


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



# by default our initial state is |+>\otimes n:
def ghz_state_prep(simulator,n):
    for m in range(1,n):
        simulator.apply_hadamard(m)
    for m in range(1,n):
        simulator.apply_cnot(0,m)



def measurement_bases(n):
    # Initialize empty 3D array
    bases_array = np.empty((0, n, 2*n), dtype=int)

    # Create the first measurement basis
    zero_matrix = np.zeros((n, n), dtype=int)
    identity_matrix = np.eye(n, dtype=int)

    # Stack and expand the dimension to make it 3D
    first_basis = np.hstack((identity_matrix, zero_matrix))[np.newaxis, :, :]

    # initialize empty row vector:
    yint = np.zeros((1,2*n),dtype=int)

    # modify first row:
    y1 = yint.copy()
    y1[0,n] = 1
    y1[0,0] = 1

    # Concatenate along the first axis
    bases_array = np.concatenate((bases_array, first_basis), axis=0)

    # generate remaining measurements:
    for k in range(1,n):
        # initialize measurement bases:
        basis = first_basis.copy()
        basis[0,0,:] = y1

        # modify kth row:
        yk = yint.copy()
        yk[0,k] = 1
        yk[0,k+n] = 1
        basis[0,k,:] = yk

        # Concatenate along the first axis
        bases_array = np.concatenate((bases_array, basis), axis=0)
        

    return bases_array


def ghz_function(n):
    boolean_mapping = dict()
    for k in range(n):
        if k == 0:
            image = 0
        else:
            image = 1
        
        boolean_mapping[k] = image
    return boolean_mapping


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