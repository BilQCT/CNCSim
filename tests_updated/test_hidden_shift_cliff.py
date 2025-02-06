import numpy as np              # import numpy
from random import choice       # random number generator
import os                       # os, sys for appending path
import sys
import re
import time                   # for timing measurements
from qiskit import QuantumCircuit, Aer, execute

# Adjust sys.path: Add the parent directory of tests_updated to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

# import simulation programs:
from circuits import hidden_shift as hs
import qcm_stab_sim as qcm

# suppress qiskit 1.0 deprecation warnings:
import warnings
warnings.filterwarnings("ignore")
warnings.warn("This is a warning!")

#########################################################################################
#########################################################################################
#########################################################################################

for m in range(1, 16):

    n = 2 * m

    # Generate shift of all 1's
    shift = [1 for _ in range(n)]

    # Create the hidden shift circuit
    qc = hs.hidden_shift(m, shift, is_clifford=True)

    qasm_code = qc.qasm()

    # Replace 'tdg' with 's; s; s; t;'
    modified_qasm_code = re.sub(r'\btdg\b', 's;\ns;\ns;\nt;', qasm_code)

    # output directory for qasm files:
    file_dir = "qasm_files"
    # output path
    path_to_dir = os.path.join(parent_dir, file_dir)

    # ensure the directory exists
    os.makedirs(path_to_dir, exist_ok=True)

    # algorithm identifier:
    filename = f"hidden_shift_n_{n}_shift_{hs.to_bit_string(shift)}_cliff"
    # qasm filename:
    qasm_file = filename + ".qasm"
    # path to qasm file in directory:
    path_to_qasm_file = os.path.join(path_to_dir, qasm_file)

    # write qasm to directory:
    with open(path_to_qasm_file, "w") as f:
        f.write(modified_qasm_code)

    #############################################################################
    # Qiskit Simulation with Timing
    #############################################################################
    simulator = Aer.get_backend('qasm_simulator')
    
    start_qiskit = time.time()
    job = execute(qc, simulator, shots=1024)
    result = job.result()
    qiskit_counts = result.get_counts()
    end_qiskit = time.time()
    qiskit_time = end_qiskit - start_qiskit

    #############################################################################
    # QCM Simulation with Timing
    #############################################################################
    # filename for generated msi clifford circuit:
    clifford_qasm = filename + "_msi.qasm"
    path_to_clifford_qasm = os.path.join(path_to_dir, clifford_qasm)

    start_qcm = time.time()
    qcm_counts = qcm.run_qcm(
        path_to_dir + os.sep,  # ensure the trailing separator
        qasm_file,
        clifford_qasm,
        shots=1024
    )
    end_qcm = time.time()
    qcm_time = end_qcm - start_qcm

    #############################################################################
    # Write the output including timing information
    #############################################################################
    output_dir = "hidden_shift_outputs"
    path_to_output_dir = os.path.join(current_dir, output_dir)
    os.makedirs(path_to_output_dir, exist_ok=True)  # ensure output directory exists
    path_to_output_file = os.path.join(path_to_output_dir, filename + ".txt")

    with open(path_to_output_file, "w") as f:
        # general information:
        f.write(f"Number of qubits: {n}\n")
        f.write(f"Hidden shift: {shift}\n\n")
        
        # Qiskit simulation results and time:
        f.write("Qiskit Simulation Results:\n")
        qiskit_percentages = [(k, v / sum(qiskit_counts.values()))
                              for k, v in qiskit_counts.items()]
        f.write("Percentages: " + str(qiskit_percentages) + "\n")
        f.write(f"Simulation Time: {qiskit_time:.4f} seconds\n\n")
        
        # QCM simulation results and time:
        f.write("Tableau (QCM) Simulation Results:\n")
        qcm_percentages = [(k, v / sum(qcm_counts.values()))
                           for k, v in qcm_counts.items()]
        f.write("Percentages: " + str(qcm_percentages) + "\n")
        f.write(f"Simulation Time: {qcm_time:.4f} seconds\n")
