import numpy as np
from random import choice
import sys, os
import chp as chp

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname('tests_updated/test_functions.py'), '..')))

from src import cnc_simulator as cnc

##########################################
#                                        #
#          Supporting functions          #
#                                        #
##########################################

def generate_gate_sequence(n, beta):
    """
    Generate a random sequence of Clifford gates for an n-qubit system.
    
    The number of gates is determined by beta * n * log2(n). For each gate,
    a random choice is made among 'h' (Hadamard), 's' (Phase), and 'cnot'.
    For single-qubit gates (h and s), a random qubit index in [0, n-1] is chosen.
    For a CNOT gate, two distinct qubit indices (control and target) are selected.
    
    Parameters:
        n (int): The number of qubits.
        beta (float): A scaling parameter for the number of gates.
    
    Returns:
        list: A list of strings representing the gate sequence, e.g. ["h_3", "cnot_1_5", ...].
    """
    num_gates = int(beta * n * np.log2(n))
    gates = ['h', 's', 'cnot']
    sequence = []
    for _ in range(num_gates):
        gate = choice(gates)
        if gate in ['h', 's']:  # Single-qubit gate
            qubit = np.random.randint(0, n)
            sequence.append(f"{gate}_{qubit}")
        elif gate == 'cnot':  # Two-qubit gate; ensure distinct control and target
            control = np.random.randint(0, n)
            target = np.random.randint(0, n)
            while target == control:
                target = np.random.randint(0, n)
            sequence.append(f"{gate}_{control}_{target}")
    return sequence

def apply_sequence_of_clifford(simulator, sequence):
    """
    Apply a sequence of Clifford gates to the given simulator (for CNC).
    
    The function iterates over each gate in the sequence and calls the
    appropriate simulator method:
      - For 'h' gates, apply Hadamard.
      - For 's' gates, apply Phase.
      - For 'cnot' gates, apply CNOT with control and target qubits.
    
    Parameters:
        simulator: An instance of cnc.CncSimulator.
        sequence (list): The list of gate strings to be applied.
    """
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

def apply_sequence_of_clifford_chp(simulator, sequence):
    """
    Apply a sequence of Clifford gates to the given CHP simulator.
    
    Similar to the CNC version, but uses the CHP simulator’s naming:
      - For 'h' gates, call hadamard(qubit).
      - For 's' gates, call phase(qubit).
      - For 'cnot' gates, call cnot(control, target).
    
    Parameters:
        simulator: An instance of chp.ChpSimulator.
        sequence (list): The list of gate strings to be applied.
    """
    for gate in sequence:
        if gate.startswith('h'):
            qubit = int(gate.split('_')[1])
            simulator.hadamard(qubit)
        elif gate.startswith('s'):
            qubit = int(gate.split('_')[1])
            simulator.phase(qubit)
        elif gate.startswith('cnot'):
            control, target = map(int, gate.split('_')[1:])
            simulator.cnot(control, target)




##########################################
#                                        #
#     Full Simulation Functions          #
#                                        #
##########################################

def simulate_init(n, beta, m, sequence=None):
    """
    Initialization only: set up the simulator by generating (or using a provided)
    gate sequence, applying it, and preparing measurement bases.
    
    Parameters:
        n (int): Number of qubits.
        beta (float): Scaling parameter for the gate sequence.
        m (int): Simulator parameter (e.g., number of auxiliary qubits).
        sequence (list, optional): If provided, use this gate sequence instead of generating one.
    
    Returns:
        tuple: (simulator, measurement_bases, sequence)
    """
    # Use provided sequence or generate a new one
    if sequence is None:
        sequence = generate_gate_sequence(n, beta)
    simulator = cnc.CncSimulator(n, m)
    apply_sequence_of_clifford(simulator, sequence)
    # Prepare measurement bases: horizontal stack of identity and zero matrix
    zero_matrix = np.zeros((n, n), dtype=int)
    identity_matrix = np.eye(n, dtype=int)
    measurement_bases = np.hstack((identity_matrix, zero_matrix))
    return simulator, measurement_bases, sequence

def simulate_full(n, beta, m, sequence=None):
    """
    Full simulation: runs initialization (using the same gate sequence if provided)
    and then performs the measurement loop.
    
    Parameters:
        n (int): Number of qubits.
        beta (float): Scaling parameter for the gate sequence.
        m (int): Simulator parameter.
        sequence (list, optional): If provided, use this gate sequence.
    
    Returns:
        None
    """
    simulator, measurement_bases, sequence = simulate_init(n, beta, m, sequence)
    for base in range(measurement_bases.shape[0]):
        simulator.measure(measurement_bases[base, :])
    return sequence  # Optionally return the sequence for consistency





##########################################
#                                        #
#     Full CHP Simulation Functions      #
#                                        #
##########################################

def simulate_init_chp(n, beta, sequence=None):
    """
    Initialization only: set up the simulator by generating (or using a provided)
    gate sequence, applying it, and preparing measurement bases.
    
    Parameters:
        n (int): Number of qubits.
        beta (float): Scaling parameter for the gate sequence.
        m (int): Simulator parameter (e.g., number of auxiliary qubits).
        sequence (list, optional): If provided, use this gate sequence instead of generating one.
    
    Returns:
        tuple: (simulator, measurement_bases, sequence)
    """
    # Use provided sequence or generate a new one
    if sequence is None:
        sequence = generate_gate_sequence(n, beta)
    simulator = chp.ChpSimulator(n)
    apply_sequence_of_clifford_chp(simulator, sequence)
    return simulator, sequence

def simulate_full_chp(n, beta, sequence=None):
    """
    Full simulation: runs initialization (using the same gate sequence if provided)
    and then performs the measurement loop.
    
    Parameters:
        n (int): Number of qubits.
        beta (float): Scaling parameter for the gate sequence.
        m (int): Simulator parameter.
        sequence (list, optional): If provided, use this gate sequence.
    
    Returns:
        None
    """
    simulator, sequence = simulate_init_chp(n, beta, sequence)
    for qubit in range(n):
        simulator.measure(qubit)
    return sequence  # Optionally return the sequence for consistency






##########################################
# GHZ Test:
# n-qubit GHZ state has stabilizers: X1...Xn, -Y1Y2...Xn, -Y1X2Y3..Xn, etc.
# Meausuring bases: X1,...,Xn; Y1,Y2,...,Xn; Y1,X2,...,Xk-1,Yk,Xk+1,...,Xn, etc.
# GHZ: Z2^n ->Z2: (0...0) \mapsto 0, otherwise: (1,1,0,...,0) \mapsto 1, (1,0,...,0,1,0,...,0) \maps 1, etc.
#
# Since the local measurements always anticommute with at least one stabilizer, this algorithm only ever uses the Case IV updates. 
##########################################





##########################################
#                                        #
#        GHZ State Preparation           #
#                                        #
##########################################
def ghz_state_prep(simulator, n):
    """
    Prepares an n-qubit GHZ state on the given simulator.
    
    The protocol assumes that the simulator's initial state is |+>⊗n.
    It applies a Hadamard gate to each qubit from index 1 to n-1,
    and then applies CNOT gates from qubit 0 (the control) to each
    of the remaining qubits (the targets). This creates an entangled
    GHZ state of the form: (|0...0> + |1...1>)/sqrt(2).
    
    Parameters:
        simulator: An instance of a quantum simulator with methods 
                   apply_hadamard(qubit) and apply_cnot(control, target).
        n (int): The number of qubits.
    """
    # Apply Hadamard gates to qubits 1 through n-1
    for m in range(1, n):
        simulator.apply_hadamard(m)
    # Apply CNOT gates from qubit 0 to each qubit 1 through n-1
    for m in range(1, n):
        simulator.apply_cnot(0, m)


##########################################
#                                        #
#       Measurement Bases Generator      #
#                                        #
##########################################
def measurement_bases(n):
    """
    Generates a 3D array of measurement bases for an n-qubit system.
    
    The function constructs an array where each element is a 2D measurement
    basis represented as a binary matrix of shape (n, 2*n). It begins by
    creating a "first" measurement basis using the identity and zero matrices.
    Then it generates additional bases by modifying the first row and the kth row
    for k = 1, ..., n-1.
    
    Returns:
        bases_array (np.ndarray): A 3D NumPy array of shape (num_bases, n, 2*n)
                                  containing the measurement bases.
    """
    # Initialize an empty 3D array with zero rows; dimensions: (0, n, 2*n)
    bases_array = np.empty((0, n, 2 * n), dtype=int)
    
    # Create the first measurement basis by concatenating an identity and zero matrix
    zero_matrix = np.zeros((n, n), dtype=int)
    identity_matrix = np.eye(n, dtype=int)
    first_basis = np.hstack((identity_matrix, zero_matrix))[np.newaxis, :, :]  # Expand dims to get 3D array
    
    # Initialize a row vector of zeros (1 x 2*n) to modify basis rows
    yint = np.zeros((1, 2 * n), dtype=int)
    
    # Create a modified version of the row vector for the first row:
    y1 = yint.copy()
    y1[0, 0] = 1      # Set first element to 1
    y1[0, n] = 1      # Set the (n+1)-th element to 1
    
    # Add the first measurement basis to the bases_array
    bases_array = np.concatenate((bases_array, first_basis), axis=0)
    
    # Generate remaining measurement bases by modifying rows of the first_basis
    for k in range(1, n):
        # Copy the first basis for modification
        basis = first_basis.copy()
        # Set the first row of the basis to the modified y1 vector
        basis[0, 0, :] = y1
        # Create a new row vector yk and modify it: set the kth element and its corresponding element in the second half to 1
        yk = yint.copy()
        yk[0, k] = 1
        yk[0, k + n] = 1
        # Replace the kth row in the basis with yk
        basis[0, k, :] = yk
        # Append this modified basis to the bases_array
        bases_array = np.concatenate((bases_array, basis), axis=0)
        
    return bases_array


##########################################
#                                        #
#       GHZ Function Mapping             #
#                                        #
##########################################
def ghz_function(n):
    """
    Creates a boolean mapping for an n-qubit GHZ state.
    
    This function returns a dictionary where the key is the qubit index and the
    value is a boolean: 0 for the first qubit and 1 for all subsequent qubits.
    This mapping can represent, for instance, the expected outcome pattern
    of a GHZ measurement.
    
    Parameters:
        n (int): The number of qubits.
    
    Returns:
        dict: A mapping from qubit indices (0 to n-1) to 0 or 1.
    """
    boolean_mapping = dict()
    for k in range(n):
        # First qubit (index 0) is mapped to 0, rest to 1
        image = 0 if k == 0 else 1
        boolean_mapping[k] = image
    return boolean_mapping
