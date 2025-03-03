import numpy as np
from src import updated_cnc_tableau as cnc
import qasm_prep as prep
import compile_keys as keys
from src import tableau_helper_functions as helper
import re
import os
import random
import copy
import time  # for timing

def is_msi_qasm(qasm_str: str) -> bool:
    """
    Check if a given QASM string represents an MSI (gadgetized) circuit.

    Looks for markers such as 'qreg q_magic', 'creg c_magic', or conditional commands 'if(c_magic['.
    
    Parameters:
        qasm_str (str): A QASM string.
    
    Returns:
        bool: True if MSI-modified, False otherwise.
    """
    markers = ["qreg q_magic", "creg c_magic", "if(c_magic["]
    return any(marker in qasm_str for marker in markers)

def extract_measured_qubit(line: str) -> dict | None:
    """
    Extract measurement info from a QASM line (e.g., "measure q_stab[0] -> c_stab[0];").
    
    Returns a dict with keys: qubit_register, qubit_index, classical_register,
    classical_index, and register_type.
    """
    measure_pattern = re.compile(r"measure\s+(\w+)\[(\d+)\]\s*->\s*(\w+)\[(\d+)\];")
    match = measure_pattern.match(line)
    if match:
        qubit_register, qubit_index, classical_register, classical_index = match.groups()
        register_type = ("stabilizer" if qubit_register == "q_stab"
                         else "magic" if qubit_register == "q_magic"
                         else "unknown")
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
    Parse a conditional QASM command like: if(c_magic[0]==1) s q_stab[2];
    
    Returns a dict with keys: gate, target_register, and target_index.
    """
    conditional_pattern = re.compile(r"if\((\w+)\[(\d+)\]==1\)\s+(\w+)\s+(\w+)\[(\d+)\];")
    match = conditional_pattern.match(line)
    if match:
        _, _, gate, target_register, target_index = match.groups()
        return {
            "gate": gate,
            "target_register": target_register,
            "target_index": int(target_index),
        }
    return None

def sign_of_quasiprobability(weight: float) -> int:
    """
    Return 1 if weight > 0, else -1.
    """
    return 1 if weight > 0 else -1

def apply_circuit(circuit_list: list, q_count: int, t_count: int, cnc_tableau: cnc.CncSimulator) -> dict:
    """
    Process a list of QASM lines (gates and measurements) and apply them on the CNC tableau.
    
    Returns a dict mapping measured stabilizer qubit indices to outcomes.
    """
    n_total = q_count + t_count
    stabilizer_outcomes = dict()
    ancilla_outcomes = dict()

    for line_idx in range(len(circuit_list)):
        line = circuit_list[line_idx]
        if line.startswith('h '):
            x = line.partition('q_stab[')
            y = x[2].partition(']')
            i = int(y[0])
            cnc_tableau.apply_hadamard(i)
        elif line.startswith('s '):
            x = line.partition('q_stab[')
            y = x[2].partition(']')
            i = int(y[0])
            cnc_tableau.apply_phase(i)
        elif line.startswith('cx '):
            w = line.partition('q_stab[')
            x = w[2].partition('],')
            i = int(x[0])
            y = x[2].partition('[')
            z = y[2].partition(']')
            if y[0] == 'q_stab': 
                j = int(z[0])
            elif y[0] == 'q_magic': 
                j = int(z[0]) + q_count  # Offset for magic qubits
            cnc_tableau.apply_cnot(i, j)
        elif line.startswith('measure'):
            measurement = extract_measured_qubit(line)
            basis = np.zeros(2 * n_total, dtype=int)
            q = measurement['qubit_index']
            if measurement['qubit_register'] == 'q_magic':
                correction = parse_conditional_command(circuit_list[line_idx+1])
                basis[n_total + q + q_count] = 1
                outcome = cnc_tableau.measure(basis)
                ancilla_outcomes[measurement['qubit_index']] = outcome
                if outcome == 1:
                    cnc_tableau.apply_phase(correction['target_index'])
            else:
                basis[n_total + q] = 1
                outcome = cnc_tableau.measure(basis)
                stabilizer_outcomes[measurement['qubit_index']] = copy.deepcopy(outcome)
    return stabilizer_outcomes

def run_qcm(msi_qasm_string: str = None, file_loc: str = None, 
            input_file_name: str = None, clifford_file_name: str = None,
            shots: int = 1024) -> tuple:
    """
    Run the gadgetized (adaptive) Clifford circuit simulation and return measurement outcomes.
    
    Two modes are supported:
      1. Direct QASM input: If an MSI QASM string is provided via msi_qasm_string,
         then if it is already MSI-modified it is processed directly. Otherwise, the
         conversion is performed via prep.QuCirc (which now directly accepts a QASM string).
      2. File-based input: If no QASM string is provided, the QASM file is loaded and
         processed similarly.
    
    Parameters:
        msi_qasm_string (str, optional): A QASM string representing the circuit.
        file_loc (str, optional): Directory for input QASM file (if no string provided).
        input_file_name (str, optional): Name of the input QASM file (if no string provided).
        clifford_file_name (str, optional): Gadgetized circuit filename (if no string provided).
        shots (int, optional): Number of simulation shots. Default is 1024.
    
    Returns:
        tuple: (counts, outputs, born_rule_estimates, shot_times)
            counts: dict mapping unique outcomes to frequencies.
            outputs: list of (outcome, sign) tuples for each shot.
            born_rule_estimates: list of (outcome, estimated value) tuples.
            shot_times: list of elapsed times per shot.
    """
    # Determine the input QASM string.
    if msi_qasm_string is not None:
        if is_msi_qasm(msi_qasm_string):
            circuit_list = msi_qasm_string.splitlines(keepends=True)
        else:
            # Not MSI-modified: run the prep conversion.
            #qc = prep.QuCirc(msi_qasm_string)
            msi_qasm_string = prep.QuCirc(msi_qasm_string).msi_circuit()  # Use default file handling inside QuCirc.
            circuit_list = msi_qasm_string.splitlines(keepends=True)
    else:
        # Load from file.
        if file_loc is None or input_file_name is None or clifford_file_name is None:
            raise ValueError("File location and file names must be provided if no MSI QASM string is given.")
        with open(os.path.join(file_loc, clifford_file_name), 'r') as f:
            msi_qasm_string = f.read()
        if not is_msi_qasm(msi_qasm_string):
            qc = prep.QuCirc(msi_qasm_string)
            msi_qasm_string = qc.msi_circuit()
        circuit_list = msi_qasm_string.splitlines(keepends=True)

    # Extract registers: number of stabilizer qubits (q_stab) and magic qubits (q_magic).
    q_count_extracted = None
    t_count_extracted = 0
    for line in circuit_list:
        q_match = re.search(r"qreg\s+q_stab\[(\d+)\];", line)
        if q_match:
            q_count_extracted = int(q_match.group(1))
        t_match = re.search(r"qreg\s+q_magic\[(\d+)\];", line)
        if t_match:
            t_count_extracted = int(t_match.group(1))
    if q_count_extracted is None:
        raise ValueError("Could not determine the number of stabilizer qubits from the QASM.")
    q_count = q_count_extracted
    t_count = t_count_extracted

    cx_count = sum(1 for line in circuit_list if line.startswith('cx '))
    hs_count = sum(1 for line in circuit_list if line.startswith('h ') or line.startswith('s '))

    # Use the circuit_list as the MSI-modified circuit.
    clifford = circuit_list

    outcomes = []
    outputs = []
    shot_times = []  # Time per shot
    n_total = q_count + t_count

    if t_count > 0:
        qprob, prob, sim_keys = keys.get_keys(t_count)
        negativity = sum(np.abs(q) for q in qprob)
        key_indices = list(range(len(prob)))
        samples = random.choices(key_indices, weights=prob, k=shots)
    else:
        negativity = 1
        samples = [i for i in range(shots)]

    # Initialize the stabilizer tableau.
    stab_tableau = cnc.CncSimulator(q_count, 0)
    for i in range(q_count):
        stab_tableau.apply_hadamard(i)

    for sample in samples:
        if t_count > 0:
            sign = sign_of_quasiprobability(qprob[sample])
            stab_array = copy.deepcopy(stab_tableau.tableau[:-1, :])
            cnc_array = copy.deepcopy(sim_keys[sample][1])
            cnc_array = helper.compose_tableaus(stab_array, cnc_array, 0, sim_keys[sample][-1])
            cnc_tableau = cnc.CncSimulator.from_tableau(n_total, sim_keys[sample][-1], cnc_array)
        else:
            sign = 1
            cnc_tableau = cnc.CncSimulator(q_count, 0)
            for i in range(q_count):
                cnc_tableau.apply_hadamard(i)

        start_time = time.time()
        simulation_results = apply_circuit(clifford, q_count, t_count, cnc_tableau)
        outcome = tuple(int(x) for x in simulation_results.values())
        outcomes.append(outcome)
        outputs.append((outcome, sign))
        elapsed = time.time() - start_time
        shot_times.append(elapsed)

    print("------------------------------\nInitial conditions for simulation:\n------------------------------\n")
    print(f"Negativity:\n {negativity}\n")
    print("------------------------------\nOutputs of Simulation:\n------------------------------\n")
    distinct_outcomes = set(outcomes)
    counts = {item: outcomes.count(item) for item in distinct_outcomes}
    avg_shot_time = np.mean(shot_times)
    print(f"Average time per shot: {avg_shot_time:.6f} seconds\n")
    
    born_rule_estimates = []
    for outcome in distinct_outcomes:
        total_outputs = [outputs[i][1] for i in range(shots) if outputs[i][0] == outcome]
        born_rule_estimate = negativity * sum(total_outputs) / shots
        print("Distinct Outcome: " + str(outcome))
        print(f"Born Rule Estimate: {born_rule_estimate}\n")
        born_rule_estimates.append((outcome, born_rule_estimate))
    
    return counts, outputs, born_rule_estimates, shot_times

if __name__ == "__main__":
    # Example usage using a direct QASM string from Qiskit.
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1, 1)
    n = 4
    for _ in range(n):
        qc.t(0)
        qc.h(0)
    qc.measure(0, 0)
    qc_qasm = qc.qasm()  # This produces a QASM string.
    
    # Run simulation. If qc_qasm is not MSI-modified, the prep code (msi_circuit) is run.
    counts, outputs, born_rule_estimates, shot_times = run_qcm(msi_qasm_string=qc_qasm, shots=4096)
    print(counts)
