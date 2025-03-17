import numpy as np
import os
import re
import time  # for timing
import copy

import src.qasm_prep as prep
import src.qcm_functions as functions
import src.compile_keys as keys
import src.tableau_helper_functions as helper
import src.cnc_simulator as cnc


def run_qcm(msi_qasm_string: str = None, file_loc: str = None, 
            input_file_name: str = None, clifford_file_name: str = None,
            hoeffding: bool = False,
            epsilon: float = 1e-2,
            prob_fail: float = 1e-2,
            shots: int = 1024) -> tuple:
    """
    Run the gadgetized (adaptive) Clifford circuit simulation and return measurement outcomes.
    
    The function supports two input modes:
      1. Direct QASM input: If an MSI QASM string is provided via 'msi_qasm_string',
         then:
           - If it is already MSI-modified (as determined by is_msi_qasm), it is processed directly.
           - Otherwise, the circuit is converted using prep.QuCirc.msi_circuit.
      2. File-based input: If no QASM string is provided, the QASM file is loaded from the 
         specified file location and processed similarly.
    
    The number of simulation shots is determined as follows:
      - If 'hoeffding' is True, the number of shots (N) is computed using Hoeffding’s inequality:
          N = (negativity**2) * (2 / (epsilon**2)) * ln(2 / prob_fail)
      - If 'hoeffding' is False, the provided 'shots' value is used.
      (Note: When hoeffding is False, the epsilon and prob_fail parameters are ignored.)
    
    Parameters
    ----------
    msi_qasm_string : str, optional
        A QASM string representing the circuit. If provided, it is processed directly.
    file_loc : str, optional
        Directory containing the input QASM file (used if msi_qasm_string is not provided).
    input_file_name : str, optional
        Name of the input QASM file (used if msi_qasm_string is not provided).
    clifford_file_name : str, optional
        Filename for the gadgetized circuit (used if msi_qasm_string is not provided).
    hoeffding : bool, optional
        Whether to compute the number of shots using Hoeffding’s inequality. Default is True.
    epsilon : float, optional
        The epsilon parameter for Hoeffding’s inequality. Default is 1e-6.
    prob_fail : float, optional
        The acceptable failure probability for Hoeffding’s inequality. Default is 1e-6.
    shots : int, optional
        The manual number of simulation shots to use if hoeffding is False. Default is 1024.
    
    Returns
    -------
    tuple
        A tuple containing:
            - counts : dict
                A dictionary mapping unique measurement outcomes (as tuples) to their frequency.
            - outputs : list
                A list of tuples, each containing (outcome, sign) for each shot.
            - born_rule_estimates : list
                A list of tuples (outcome, estimated value) computed using the negativity and outcomes.
            - shot_times : list
                A list of elapsed times (in seconds) for each shot.
    """
    # Process input QASM string.
    if msi_qasm_string is not None:
        if functions.is_msi_qasm(msi_qasm_string):
            circuit_list = msi_qasm_string.splitlines(keepends=True)
        else:
            # Not MSI-modified: run conversion using prep.QuCirc.
            qc = prep.QuCirc(msi_qasm_string)
            msi_qasm_string = qc.msi_circuit()  # Uses default behavior of QuCirc.
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
    
    # Extract register information: number of stabilizer qubits (q_stab) and magic qubits (q_magic).
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
    shot_times = []  # List to store time per shot
    n_total = q_count + t_count

    # Compute negativity from keys if there are magic qubits.
    if t_count > 0:
        negativity = keys.compute_total_negativity(t_count)
    else:
        negativity = 1

    # Initialize the stabilizer tableau.
    stab_tableau = cnc.CncSimulator(q_count, 0)
    for i in range(q_count):
        stab_tableau.apply_hadamard(i)

    # Determine the number of simulation shots.
    if hoeffding:
        N = (negativity**2) * (2 / (epsilon**2)) * np.log(2 / prob_fail)
    else:
        N = shots
    
    print("------------------------------\nInitial conditions for simulation:\n------------------------------\n")
    print(f"Negativity:\n {negativity}\n")
    print(f"Number of shots:\n {int(N)}\n")

    for _ in range(int(N)):
        if t_count > 0:
            cnc_array, q, m = keys.sample_single_key(t_count)
            sign = np.sign(q)
            stab_array = copy.deepcopy(stab_tableau.tableau[:-1, :])
            cnc_array = helper.compose_tableaus(stab_array, cnc_array, 0, m)
            cnc_tableau = cnc.CncSimulator.from_tableau(n_total, m, cnc_array)
        else:
            sign = 1
            cnc_tableau = cnc.CncSimulator(q_count, 0)
            for i in range(q_count):
                cnc_tableau.apply_hadamard(i)
    
        start_time = time.time()
        simulation_results = functions.apply_circuit(clifford, q_count, t_count, cnc_tableau)
        outcome = tuple(int(x) for x in simulation_results.values())
        outcomes.append(outcome)
        outputs.append((outcome, sign))
        elapsed = time.time() - start_time
        shot_times.append(elapsed)


    print("------------------------------\nOutputs of Simulation:\n------------------------------\n")
    distinct_outcomes = set(outcomes)
    counts = {item: outcomes.count(item) for item in distinct_outcomes}
    avg_shot_time = np.mean(shot_times)
    print(f"Average time per shot: {avg_shot_time:.6f} seconds\n")
    
    born_rule_estimates = []
    for outcome in distinct_outcomes:
        total_outputs = [outputs[i][1] for i in range(int(N)) if outputs[i][0] == outcome]
        born_rule_estimate = negativity * sum(total_outputs) / N
        print("Distinct Outcome: " + str(outcome))
        print(f"Born Rule Estimate: {born_rule_estimate}\n")
        born_rule_estimates.append((outcome, born_rule_estimate))
    
    return (outputs, born_rule_estimates, shot_times)