import re
import copy
import src.updated_cnc_tableau as cnc
import numpy as np

#################################################################
#                                                               #
#                Functions for running QCM sim                  #
#                                                               #                                                            
#################################################################

def is_msi_qasm(qasm_str: str) -> bool:
    """
    Check if a given QASM string represents an MSI (gadgetized) circuit.
    
    The function looks for specific markers in the QASM string that indicate it has been 
    modified by the MSI (magic state injection) process. Markers include:
        - 'qreg q_magic'
        - 'creg c_magic'
        - Conditional commands starting with 'if(c_magic['
    
    Parameters
    ----------
    qasm_str : str
        The QASM string representing the quantum circuit.
    
    Returns
    -------
    bool
        True if the QASM string appears to be MSI-modified, False otherwise.
    """
    markers = ["qreg q_magic", "creg c_magic", "if(c_magic["]
    return any(marker in qasm_str for marker in markers)


def extract_measured_qubit(line: str) -> dict | None:
    """
    Extract measurement information from a QASM measurement statement.
    
    The function expects a QASM measurement statement in the format:
    
        measure <qubit_register>[<qubit_index>] -> <classical_register>[<classical_index>];
    
    It returns a dictionary containing:
        - qubit_register: Name of the qubit register (e.g. "q_stab" or "q_magic")
        - qubit_index: Index of the measured qubit (as an integer)
        - classical_register: Name of the classical register receiving the measurement
        - classical_index: Index in the classical register (as an integer)
        - register_type: Either "stabilizer" (if qubit_register is "q_stab") or "magic" (if "q_magic"), 
                         otherwise "unknown".
    
    Parameters
    ----------
    line : str
        A QASM statement for measurement.
    
    Returns
    -------
    dict or None
        A dictionary with measurement details if the line matches the expected format;
        otherwise, None.
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
    Parse a conditional QASM command.
    
    This function is designed to handle lines of the form:
    
        if(c_magic[0]==1) s q_stab[2];
    
    It returns a dictionary with:
        - gate: The gate to be applied (e.g. "s")
        - target_register: The target register (e.g. "q_stab")
        - target_index: The index in the target register (as an integer)
    
    Parameters
    ----------
    line : str
        A QASM statement containing a conditional command.
    
    Returns
    -------
    dict or None
        A dictionary with keys 'gate', 'target_register', and 'target_index' if parsing is successful;
        otherwise, None.
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


def apply_circuit(circuit_list: list, q_count: int, t_count: int, cnc_tableau: cnc.CncSimulator) -> dict:
    """
    Process a list of QASM lines and apply the corresponding gates and measurements on the CNC tableau.
    
    This function iterates over the QASM lines (provided as a list of strings) and applies:
        - Hadamard (h) and Phase (s) gates on the stabilizer register (q_stab).
        - CNOT (cx) gates, with special handling for magic qubits (q_magic) where an offset is applied.
        - Measurement operations. For magic qubit measurements, it also processes the following conditional command.
    
    Parameters
    ----------
    circuit_list : list of str
        The list of QASM lines representing the circuit.
    q_count : int
        The number of qubits in the stabilizer register.
    t_count : int
        The number of T-gate (magic state) qubits.
    cnc_tableau : cnc.CncSimulator
        An instance of the CNC simulator on which the circuit operations are applied.
    
    Returns
    -------
    dict
        A dictionary mapping measured stabilizer qubit indices to their measurement outcomes.
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