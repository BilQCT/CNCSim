import numpy as np
from src import updated_cnc_tableau as cnc
import input_prep as prep
import compile_keys as keys
from src import tableau_helper_functions as helper
import re
import os
import random
import copy
import time  # <-- Import the time module for timing

def extract_measured_qubit(line: str) -> dict | None:
    """
    Extract measurement information from a QASM line.

    The function expects a line of QASM code in the format:
        measure <qubit_register>[<qubit_index>] -> <classical_register>[<classical_index>];

    It returns a dictionary containing:
      - qubit_register: Name of the qubit register (e.g. "q_stab" or "q_magic")
      - qubit_index: The index of the measured qubit (converted to int)
      - classical_register: The classical register name receiving the measurement
      - classical_index: The index in the classical register (converted to int)
      - register_type: A string ("stabilizer", "magic", or "unknown") determined from the qubit register name

    Parameters:
        line (str): A QASM statement for measurement.

    Returns:
        dict: A dictionary with measurement details if the line matches the expected format.
        None: If the line does not match the expected format.
    """
    measure_pattern = re.compile(r"measure\s+(\w+)\[(\d+)\]\s*->\s*(\w+)\[(\d+)\];")
    match = measure_pattern.match(line)
    if match:
        qubit_register, qubit_index, classical_register, classical_index = match.groups()
        
        # Determine the register type based on the qubit register name.
        register_type = (
            "stabilizer" if qubit_register == "q_stab" 
            else "magic" if qubit_register == "q_magic" 
            else "unknown"
        )
        
        return {
            "qubit_register": qubit_register,
            "qubit_index": int(qubit_index),
            "classical_register": classical_register,
            "classical_index": int(classical_index),
            "register_type": register_type
        }
    return None


def parse_conditional_command(line: str) -> dict | None:
    """
    Parse a conditional QASM command.

    This function handles QASM lines of the form:
        if(c_magic[0]==1) s q_stab[2];
    and returns a dictionary with the parsed components:
      - gate: The gate to be applied (e.g. "s")
      - target_register: The register on which the gate is applied (e.g. "q_stab")
      - target_index: The index within that register (converted to int)

    Parameters:
        line (str): A QASM statement containing a conditional command.

    Returns:
        dict: A dictionary with keys 'gate', 'target_register', and 'target_index' if parsing is successful.
        None: If the line does not match the expected conditional command format.
    """
    conditional_pattern = re.compile(
        r"if\((\w+)\[(\d+)\]==1\)\s+(\w+)\s+(\w+)\[(\d+)\];"
    )
    match = conditional_pattern.match(line)
    
    if match:
        classical_register, classical_index, gate, target_register, target_index = match.groups()
        # The register type of the target can be inferred if needed.
        register_type = (
            "stabilizer" if target_register == "q_stab" 
            else "magic" if target_register == "q_magic" 
            else "unknown"
        )
        return {
            "gate": gate,
            "target_register": target_register,
            "target_index": int(target_index),
        }
    return None


def sign_of_quasiprobability(weight: float) -> int:
    """
    Return the sign of a quasiprobability weight.

    Parameters:
        weight (float): The quasiprobability weight, which may be negative.

    Returns:
        int: 1 if the weight is positive, -1 if the weight is negative.
    """
    return 1 if weight > 0 else -1


def apply_circuit(circuit_list: list, q_count: int, t_count: int, cnc_tableau: cnc.CncSimulator) -> dict:
    """
    Execute a sequence of gate operations and measurements on a CNC tableau.

    The function processes a list of QASM lines (as strings) that describe a circuit.
    It applies gates (Hadamard, phase, CNOT) and measurements to the provided
    `cnc_tableau` (an instance of CncSimulator) and returns the outcomes of measurements
    on the stabilizer qubits.

    The measurement handling distinguishes between "magic" qubits and "stabilizer" qubits:
      - For a magic qubit measurement, the function expects a subsequent line that specifies
        the correction (via a conditional command) to be applied if the measurement outcome is 1.
      - For a stabilizer qubit measurement, the outcome is directly stored.

    Parameters:
        circuit_list (list): List of QASM strings representing circuit operations.
        q_count (int): Number of qubits in the stabilizer register.
        t_count (int): Number of T-gate (magic state) qubits.
        cnc_tableau (cnc.CncSimulator): The current CNC tableau simulator instance.

    Returns:
        dict: A dictionary mapping measured stabilizer qubit indices to their outcomes.
    """
    n_total = q_count + t_count
    stabilizer_outcomes = dict()
    ancilla_outcomes = dict()

    for line_idx in range(len(circuit_list)):
        line = circuit_list[line_idx]

        if line.startswith('h '):
            # Extract qubit index from a Hadamard operation on a stabilizer qubit.
            x = line.partition('q_stab[')
            y = x[2].partition(']')
            i = int(y[0])
            #print(f"Applied Hadamard on {i}\n")
            cnc_tableau.apply_hadamard(i)
            #print(f"Updated Tableau: \n {cnc_tableau}\n")

        elif line.startswith('s '):
            # Extract qubit index from a Phase gate on a stabilizer qubit.
            x = line.partition('q_stab[')
            y = x[2].partition(']')
            i = int(y[0])
            #print(f"Applied Phase on {i}\n")
            cnc_tableau.apply_phase(i)
            #print(f"Updated Tableau: \n {cnc_tableau}\n")

        elif line.startswith('cx '):
            # Parse the CNOT command. It extracts control and target qubits.
            w = line.partition('q_stab[')
            x = w[2].partition('],')
            i = int(x[0])
            y = x[2].partition('[')
            z = y[2].partition(']')
            if y[0] == 'q_stab': 
                j = int(z[0])
            elif y[0] == 'q_magic': 
                j = int(z[0]) + q_count  # Offset for magic qubits
            #print(f"Applied CNOT from {i} to {j}\n")
            cnc_tableau.apply_cnot(i, j)
            #print(f"Updated Tableau: \n {cnc_tableau}\n")

        elif line.startswith('measure'):
            # Process measurement commands.
            measurement = extract_measured_qubit(line)
            basis = np.zeros(2 * n_total, dtype=int)
            q = measurement['qubit_index']

            #print(f"CNC Tableau before measurement:\n\n {cnc_tableau}\n")

            if measurement['qubit_register'] == 'q_magic':
                # For magic qubits, we expect an additional line that specifies the correction.
                # (The check for additional line is omitted here.)
                correction = parse_conditional_command(circuit_list[line_idx+1])
                # Set the measurement basis entry for the magic qubit.
                basis[n_total + q + q_count] = 1

                outcome = cnc_tableau.measure(basis)
                ancilla_outcomes[measurement['qubit_index']] = outcome

                #print(f"Measurement on magic qubit {q}, outcome: {outcome} \n")
                #print(f"Updated Tableau: \n {cnc_tableau}\n")

                # If the measurement outcome is 1, apply the conditional correction.
                if outcome == 1:
                    #print(f"Applied Correction Phase on {correction['target_index']}\n")
                    cnc_tableau.apply_phase(correction['target_index'])
                    #print(f"Updated Tableau: \n {cnc_tableau}\n")

            else:
                # For stabilizer qubits, set the measurement basis accordingly.
                basis[n_total + q] = 1
                outcome = cnc_tableau.measure(basis)
                stabilizer_outcomes[measurement['qubit_index']] = copy.deepcopy(outcome)

                #print(f"Updated Tableau: \n {cnc_tableau}\n")
                #print(f"Measurement on stabilizer qubit {q}, outcome: {outcome}\n")
                #print(f"Updated Tableau: \n {cnc_tableau}\n")
        
    return stabilizer_outcomes


def run_qcm(file_loc: str, input_file_name: str, clifford_file_name: str, shots: int = 1024) -> tuple:
    """
    Run the gadgetized (adaptive) Clifford circuit simulation and return measurement outcomes.

    This function performs the following steps:
      1. Prepares the quantum circuit (using the provided QASM file) by calling
         `prep.QuCirc` and generating an equivalent gadgetized Clifford circuit.
      2. Retrieves keys (tableaux) for T-state resources from `keys.get_keys`.
      3. Samples from the quasiprobability distribution to choose one of the keys for each shot.
      4. For each shot:
         - Composes the initial stabilizer tableau with the sampled key using `helper.compose_tableaus`.
         - Creates a new `CncSimulator` instance from the composed tableau.
         - Runs the circuit operations (gates and measurements) using `apply_circuit`.
         - Records the measurement outcomes along with the sign of the corresponding quasiprobability.
      5. Computes and prints theoretical probabilities and sampling frequencies.

    Parameters:
        file_loc (str): Directory containing the input QASM file.
        input_file_name (str): The input QASM file name.
        clifford_file_name (str): The output filename for the gadgetized Clifford circuit.
        shots (int, optional): Number of simulation shots to run. Default is 1024.

    Returns:
        tuple: A tuple containing:
            - counts (dict): A dictionary with unique outcomes as keys and their frequency as values.
            - outputs (list): A list of tuples, each containing (outcome, sign) for each shot.
            - born_rule_estimates (list): A list of tuples (outcome, estimated value) computed using the negativity and outcomes.
    """
    # Create the gadgetized Clifford circuit.
    qc = prep.QuCirc(file_loc, input_file_name)
    clifford, q_count, t_count, cx_count, hs_count, nr_mmts = qc.msi_circuit(file_loc, clifford_file_name)
    # q_count: number of qubits of the original circuit.
    # t_count: number of T-gates (magic states) used.
    # nr_mmts: number of measurements in the gadgetized circuit.

    outcomes = []
    outputs = []
    shot_times = []  # List to store time per shot
    n_total = q_count + t_count

    # Retrieve keys for T-states and compute the quasiprobability distribution.
    qprob, prob, sim_keys = keys.get_keys(t_count)
    negativity = sum(np.abs(q) for q in qprob)

    # Create an index list corresponding to the keys.
    key_indices = list(range(len(prob)))

    # Sample keys according to the probability distribution.
    samples = random.choices(key_indices, weights=prob, k=shots)

    # Initialize the stabilizer part of the tableau.
    stab_tableau = cnc.CncSimulator(q_count, 0)
    for i in range(q_count):
        stab_tableau.apply_hadamard(i)

    shot_number = 0
    for sample in samples:
        # Record the start time for this shot.
        start_time = time.time()

        sign = sign_of_quasiprobability(qprob[sample])
        shot_number += 1

        # Create a copy of the initial stabilizer tableau.
        stab_array = copy.deepcopy(stab_tableau.tableau[:-1, :])
        cnc_array = copy.deepcopy(sim_keys[sample][1])
        cnc_array = helper.compose_tableaus(stab_array, cnc_array, 0, sim_keys[sample][-1])
        cnc_tableau = cnc.CncSimulator.from_tableau(n_total, sim_keys[sample][-1], cnc_array)

        #print(f"Sampled CNC Key: {sample}\n{cnc_tableau}\n")

        # Run the circuit operations without adaptivity.
        simulation_results = apply_circuit(clifford, q_count, t_count, cnc_tableau)
        outcome = tuple(int(x) for x in simulation_results.values())
        outcomes.append(outcome)
        outputs.append((outcome, sign))

        # Record the end time and compute elapsed time for this shot.
        end_time = time.time()
        elapsed = end_time - start_time
        shot_times.append(elapsed)


    print("------------------------------\nInitial conditions for simulation:\n------------------------------\n")
    print(f"Negativity:\n {negativity}\n")

    print("------------------------------\nOutputs of Simulation:\n------------------------------\n")
    distinct_outcomes = set(outcomes)
    counts = {item: outcomes.count(item) for item in distinct_outcomes}

    # Print shot timing statistics.
    avg_shot_time = np.mean(shot_times)
    print(f"Average time per shot: {avg_shot_time:.6f} seconds\n")

    # Compute Born rule estimates for each distinct outcome.
    born_rule_estimates = []
    for outcome in distinct_outcomes:
        print("Distinct Outcome: " + str(outcome))
        total_outputs = [outputs[i][1] for i in range(shots) if outputs[i][0] == outcome]
        born_rule_estimate = negativity * sum(total_outputs) / shots
        print(f"Born Rule Estimate: {born_rule_estimate}\n")
        born_rule_estimates.append((outcome, born_rule_estimate))

    return counts, outputs, born_rule_estimates, shot_times


if __name__ == "__main__":

    # Specify the directory where the QASM file is located.
    file_directory = "./qasm_files/"
    
    filename = "ccz_circuit_n_3"
    qasm_file = filename + ".qasm"
    clifford_filename = filename + "_msi.qasm"

    # Run the gadgetized circuit simulation.
    counts, outputs, born_rule_estimates, shot_times = run_qcm(file_directory, qasm_file, clifford_filename, shots=4096)
