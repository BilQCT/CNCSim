from __future__ import annotations

import io
import os
import zipfile
from abc import ABC
from dataclasses import dataclass
from math import isclose, pi
from typing import Sequence, Union
import matplotlib.pyplot as plt;

import h5py
import numpy as np
#from qutip import Qobj
#from qutip.measurement import measure

# Quantum Gates
PAULI_X = np.array([[0, 1], [1, 0]])
PAULI_Y = np.array([[0, -1j], [1j, 0]])
PAULI_Z = np.array([[1, 0], [0, -1]])
H_GATE = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
T_GATE = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])


class Pauli:
    """Class for Pauli operators."""

    def __init__(self, identifier: Union[np.ndarray, str, Sequence[int]]) -> None:
        """Initializes the Pauli operator.

        Args:
            identifier (Union[np.ndarray, str, Sequence[int]]): Identifier of the Pauli operator.
                It can be a numpy array or any sequence(i.e. list or tuple) of 0s and 1s or a string of Pauli operators.
                Check https://arxiv.org/abs/2004.01992 for the definition of Pauli operators.
                For example, "XZ" is equivalent to [1, 0, 0, 1].
        """
        if isinstance(identifier, str):
            bsf = pauli_str_to_bsf(identifier)
        elif isinstance(identifier, Sequence):
            if not all(isinstance(i, int) for i in identifier) or not all(
                i in [0, 1] for i in identifier
            ):
                raise ValueError(
                    "Identifier provided as a Sequence must contain only integers 0 and 1."
                )
            bsf = np.array(identifier)
        elif isinstance(identifier, np.ndarray):
            if not all(isinstance(i, np.int_) for i in identifier) or not all(
                i in [0, 1] for i in identifier
            ):
                raise ValueError(
                    "Identifier provided as a Numpy array must contain only integers 0 and 1."
                )
            bsf = identifier
        else:
            raise RuntimeError("Invalid identifier")

        if len(bsf) < 0 or len(bsf) % 2 == 1:
            raise ValueError(
                "Size of a is not correct. It must be an positive even number."
            )

        self._n = len(bsf) // 2
        self._bsf = bsf
        self._phase = (self.bsf[: self.n].dot(self.bsf[self.n :])) % 4

    @property
    def n(self) -> int:
        """Number of qubits."""
        return self._n

    @property
    def bsf(self) -> np.ndarray:
        """Binary Symplectic Form of the Pauli operator. For example, "XZ" is equivalent to [1, 0, 0, 1].
        For n qubits, the length of the binary array is 2n."""
        return self._bsf

    @property
    def basis_order(self) -> int:
        """Basis order of the Pauli is its order when all Paulis with same number of qubits are sorted
        lexicographically according to their string representations. This number is used in enumaration
        of Pauli basis.
        """
        pauli_str = pauli_bsf_to_str(self.bsf)
        basis_order = 0
        for i in range(self.n):
            qubit_index = self.n - i - 1
            inc_unit = 4**i
            if pauli_str[qubit_index] == "X":
                basis_order += inc_unit
            elif pauli_str[qubit_index] == "Y":
                basis_order += inc_unit * 2
            elif pauli_str[qubit_index] == "Z":
                basis_order += inc_unit * 3

        return basis_order

    @property
    def phase(self) -> int:
        """Phase of the Pauli operator. It is the phase factor that arises when the Pauli operator is
        represented as a matrix. It is calculated as the dot product of the X and Z parts of the Pauli operator. For example,
        Y has a phase of 1 and XZ has a phase of 0."""
        return self._phase

    @classmethod
    def identity(cls, n: int) -> Pauli:
        """It is the convenience function for creating n qubit identity Paulis. It returns the n qubit identity Pauli."""
        return cls(np.zeros(2 * n, dtype=int))

    @classmethod
    def from_basis_order(cls, n: int, basis_order: int) -> Pauli:
        """Creates a Pauli with a given number of qubits and basis order."""
        if basis_order < 0 or basis_order >= 4**n:
            raise ValueError(
                "Basis order exceeds its limit. It must be between 0 and 4^n - 1."
            )

        index_to_pauli = {0: "I", 1: "X", 2: "Y", 3: "Z"}
        inc_unit = 4 ** (n - 1)
        pauli_str = ""
        while inc_unit != 0:
            pauli_index, basis_order = divmod(basis_order, inc_unit)
            pauli_str += index_to_pauli[pauli_index]
            inc_unit //= 4

        return cls(pauli_str)

    def get_operator(self) -> np.ndarray:
        """Creates the matrix representation of the Pauli operator."""
        result = np.eye(1)
        for i in range(self.n):
            temp = np.eye(2)
            if self.bsf[i] == 1:
                temp = temp @ PAULI_X
            if self.bsf[i + self.n] == 1:
                temp = temp @ PAULI_Z
            result = np.kron(result, temp)

        result = (1j**self.phase) * result
        return result

    def calculate_gamma(self, other: Pauli) -> int:
        """Calculates the gamma value that arises when two Pauli operators are multiplied.

        Note:
            .. math::
                T_a T_b = (i)^{\\gamma} T_{a + b}

        Args:
            other (Pauli): The other Pauli operator.

        Returns:
            int: gamma value (in modulo 4).
        """
        if self.n != other.n:
            raise ValueError("Size of the operators is not correct")

        n = self.n
        a_x = self.bsf[:n]
        a_z = self.bsf[n:]
        b_x = other.bsf[:n]
        b_z = other.bsf[n:]

        x_terms = (a_x + b_x) % 2
        z_terms = (a_z + b_z) % 2

        combined_phase = x_terms.dot(z_terms) % 4
        temp = (self.phase + other.phase + 2 * a_z.dot(b_x)) % 4
        gamma = (temp - combined_phase) % 4
        return (gamma) % 4

    def calculate_beta(self, other: Pauli) -> int:
        """Calculates the gamma value that arises when two commuting Pauli operators are multiplied.

        Note:
            .. math::
                T_a T_b = (-1)^{\\beta} T_{a + b}

        Warning:
            It is the special case of :func:`calculate_gamma` method when the operators are commuting.

        Args:
            other (Pauli): The other Pauli operator.

        Returns:
            int: beta value (in modulo 2).
        """
        if self.calculate_omega(other) != 0:
            raise ValueError("The operators are not commuting")

        return self.calculate_gamma(other) // 2

    def calculate_omega(self, other: Pauli) -> int:
        """Calculates the omega value that determines if two Pauli operators are commuting or anticommuting.

        Note:
            .. math::
                T_a T_b = (-1)^{\\omega} T_b T_a

        Args:
            other (Pauli): The other Pauli operator.

        Returns:
            int: omega value (in modulo 2).
        """
        if self.n != other.n:
            raise ValueError("Size of the operators is not correct")
        n = self.n
        a_z = self.bsf[:n]
        a_x = self.bsf[n:]
        b_z = other.bsf[:n]
        b_x = other.bsf[n:]
        return (a_z.dot(b_x) - a_x.dot(b_z)) % 2

    def __str__(self) -> str:
        pauli_str = "Pauli Operator: " + pauli_bsf_to_str(self.bsf)
        return pauli_str

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: Pauli) -> bool:
        return np.array_equal(self.bsf, other.bsf)

    def __add__(self, other: Pauli) -> Pauli:
        """Addition of the arrays that represent the Pauli operators. The addition is done modulo 2.
        For given a representing T_a and b representing T_b it gives (a+b) representing T_(a+b).

        Args:
            other (Pauli): The other Pauli operator.

        Returns:
            Pauli: Result of the addition of the Pauli operators.
        """
        if self.n != other.n:
            raise ValueError(
                "Pauli operators with different number of qubits cannot be added."
            )

        added_a = (self.bsf + other.bsf) % 2
        return Pauli(added_a)

    def __hash__(self) -> int:
        return hash(self.__str__())


def load_all_maximal_cncs_matrix(n: int) -> np.ndarray:
    """Loads the precomputed matrix of all maximal CNC states for n qubits. The matrix is stored in a .jld file
    which zipped to reduce the size. Every column of the matrix represents a CNC state in its Pauli basis
    representation.

    Args:
        n (int): Number of qubits.

    Returns:
        np.ndarray: Matrix of all possible CNC states.  Every column of the matrix represents a CNC state in its
        Pauli basis representation.
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("Number of qubits must be a positive integer.")

    zip_file = f"maximal_cnc_matrices/all_maximal_cncs_matrix_{n}.zip"
    jld_file = f"all_maximal_cncs_matrix_{n}.jld"

    if not os.path.exists(zip_file):
        raise ValueError(
            "There is no precomputed matrix of all maximal CNC atates for the given number of qubits."
        )

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        with zip_ref.open(jld_file) as file:
            with h5py.File(io.BytesIO(file.read()), "r") as f:
                # cast to numpy array
                return np.array(f["matrix"]).T


def pauli_str_to_bsf(pauli_string: str) -> np.ndarray:
    """Converts a string of Pauli operators to a array of 0s and 1s which is its Binary Symplectic Form.
    For example "XZ" is converted to [1, 0, 0, 1]. The string is case insensitive.

    Args:
        pauli_string (str): String representation of the Pauli operator.

    Returns:
        np.ndarray: Binary Symplectic Form of the Pauli operator.
    """
    if not isinstance(pauli_string, str):
        raise RuntimeError("Pauli string must be a string.")

    a_x = np.zeros(len(pauli_string))
    a_z = np.zeros(len(pauli_string))

    pauli_string = pauli_string.upper().strip()

    for i, op in enumerate(pauli_string):
        if op == "X":
            a_x[i] = 1
        elif op == "Y":
            a_x[i] = 1
            a_z[i] = 1
        elif op == "Z":
            a_z[i] = 1
        elif op == "I":
            pass
        else:
            raise ValueError("Invalid Pauli string. It must contain only X, Y, Z, I.")

    bsf = np.concatenate((a_x, a_z)).astype(int)

    return bsf


def pauli_bsf_to_str(pauli_bsf: Union[np.ndarray, Sequence[int]]) -> str:
    """Converts Binary Symplectic Form of a Pauli operator to its string representation.
    For example [1, 0, 0, 1] is converted to "XZ".

    Args:
        pauli_bsf (Union[np.ndarray, Sequence[int]]): Binary Symplectic Form of the Pauli operator.
            It can be a numpy array or any sequence(i.e. list or tuple) of 0s and 1s.

    Returns:
        str: String representation of the Pauli operator.
    """
    if not isinstance(pauli_bsf, np.ndarray) and not isinstance(pauli_bsf, Sequence):
        raise RuntimeError("Binary symplectic form of the Pauli operator must be a numpy array or a sequence.")

    if not all(i in [0, 1] for i in pauli_bsf):
        raise ValueError("Binary symplectic form of the Pauli operator is not a binary array.")

    if len(pauli_bsf) % 2 == 1 or len(pauli_bsf) <= 0:
        raise ValueError(
            "Size of the given Binary Symplectic Form is not correct. It must be a positive even number."
        )

    pauli_str = ""

    n = len(pauli_bsf) // 2
    for i in range(n):
        if pauli_bsf[i] == 1 and pauli_bsf[i + n] == 1:
            pauli_str += "Y"
        elif pauli_bsf[i] == 1:
            pauli_str += "X"
        elif pauli_bsf[i + n] == 1:
            pauli_str += "Z"
        else:
            pauli_str += "I"

    return pauli_str

"""
def qutip_simuation(
    initial_state: Qobj, measurements: list[Qobj], num_simulations: int
) -> dict[str, int]:
    "Runs a simulation with the given initial state and measurements using QuTiP. The simulation is repeated
    num_simulations times and the counts of the measurement outcomes are returned.

    Args:
        initial_state (Qobj): Initial state of the system.
        measurements (list[Qobj]): List of measurement operators.
        num_simulations (int): Number of simulations.

    Returns:
        dict[str, int]: Dictionary of measurement outcomes and their counts.
    "
    counts = []
    for _ in range(num_simulations):
        state = initial_state
        outcomes = []
        for measurement in measurements:
            outcome, state = measure(state, measurement)
            if isclose(outcome, 1):
                outcome = 0
            else:
                outcome = 1
            outcomes.append(outcome)

        counts.append("".join(str(i) for i in outcomes))

    counts.sort()
    counts = {x: counts.count(x) for x in counts}

    return counts
"""

def get_n_from_pauli_basis_representation(
    pauli_basis_representation: np.ndarray,
) -> int:
    """Calculates the number of qubits from the Pauli basis representation of a state.

    Args:
        pauli_basis_representation (np.ndarray): Pauli basis representation of a state.

    Returns:
        int: Number of qubits.
    """
    if not isinstance(pauli_basis_representation, np.ndarray):
        raise RuntimeError("Pauli basis representation must be a numpy array.")

    n = 1
    l = 4
    while l < len(pauli_basis_representation):
        n += 1
        l *= 4

    if l != len(pauli_basis_representation):
        raise ValueError("The size of the basis representation must be a power of 4.")

    return n


class PhasePointOperator(ABC):
    """Abstract class for phase point operators."""

    pass


@dataclass
class DecompositionElement:
    """Class for the elements of the decomposition of a state.

    Attributes:
        operator (Union[np.ndarray, PhasePointOperator]): Operator of the element.
        probability (float): Probability of the operator in the decomposition.
    """

    operator: Union[np.ndarray, PhasePointOperator]  # Operator of the element.
    probability: float

    def __post_init__(self) -> None:
        if not isinstance(self.probability, (int, float)):
            raise ValueError("Probability must be a number.")
        if self.probability <= 0:
            raise ValueError("Probability must be a positive number.")





"""
Circuits:

"""




def xor_frequencies(dict):
    keys = dict.keys(); values = dict.values()
    xor_zero_freq = 0; xor_one_freq = 0
    for key in keys:
        xor = (sum([int(x,2) for x in list(key)])) % 2
        if xor == 0:
            xor_zero_freq += dict[key]
        else:
            xor_one_freq += dict[key]
    return [['0','1'],[xor_zero_freq,xor_one_freq]]

# Function to apply measurement in a specific basis
def apply_measurement(qc, qubit_index, basis):
    if basis == 'X':
        qc.h(qubit_index)
    elif basis == 'Y':
        qc.sdg(qubit_index)
        qc.h(qubit_index)

from qiskit import QuantumCircuit, Aer, execute

def run_boolean_function():
    inputs = [[x,y] for x in range(2) for y in range(2)]
    outcome_counts = []
    for input in inputs:
        # Define inputs a and b
        a = input[0]  # Example input for qubit 1
        b = input[1]  # Example input for qubit 2

        # Create a quantum circuit with 3 qubits
        qc = QuantumCircuit(3, 3)  # 3 qubits and 3 classical bits for measurement results

        # Initialize qubits in |0> state (this is done by default)

        # Apply Hadamard gate to each qubit
        qc.h(range(3))

        # Apply controlled-Z gate from qubit 0 to qubit 1
        qc.cz(0, 1)

        # Apply controlled-Z gate from qubit 1 to qubit 2
        qc.cz(1, 2)

        # Determine measurement basis for qubit 1 based on input a
        if a == 0:
            measurement_basis_q1 = 'Y'
        else:
            measurement_basis_q1 = 'Z'

        # Determine measurement basis for qubit 2 based on input b
        if b == 0:
            measurement_basis_q2 = 'X'
        else:
            measurement_basis_q2 = 'Y'

        # Determine measurement basis for qubit 3 based on a + b mod 2
        if (a + b) % 2 == 0:
            measurement_basis_q3 = 'Y'
        else:
            measurement_basis_q3 = 'Z'

        # Measure each qubit in the determined basis
        apply_measurement(qc, 0, measurement_basis_q1)
        qc.measure(0, 0)  # Measure qubit 0 and store the result in classical bit 0
        apply_measurement(qc, 1, measurement_basis_q2)
        qc.measure(1, 1)  # Measure qubit 1 and store the result in classical bit 1
        apply_measurement(qc, 2, measurement_basis_q3)
        qc.measure(2, 2)  # Measure qubit 2 and store the result in classical bit 2
        print(f"Quantum circuit for inputs ({a},{b})")
        # Draw the circuit
        print(qc.draw(),"\n")

        # Simulate the circuit
        simulator = Aer.get_backend('qasm_simulator')
        job = execute(qc, simulator, shots=1000)
        result = job.result()

        # Get the counts
        counts = (result.get_counts(qc))
        outcome_counts.append(counts)
    return outcome_counts


from math import pi

def qc_magic_HTH():
    # Create a quantum circuit with 2 qubits and 1 classical bit
    qc = QuantumCircuit(2, 1)

    # Initialize qubit 1 in |0> state (this is done by default)

    # Initialize qubit 2 in |0> + exp(i*pi/4)|1> state
    #qc.u1(0.25 * 3.14159, 1)

    # Apply Hadamard gate to qubit 1
    qc.h(0)

    # Apply Hadamard gate to qubit 1
    qc.h(1)

    # Apply Z-rotation to qubit 1
    qc.p(pi/4,1)

    # Apply CNOT gate from qubit 1 to qubit 2
    qc.cx(0, 1)

    # Measure qubit 2 in Z basis and store the result in classical bit 0
    qc.measure(1, 0)

    # Conditional operation based on the measurement outcome
    qc.p(pi/2,0).c_if(0, 1)  # Apply S gate if outcome is 1, otherwise do nothing

    # Measure qubit 1 in X basis
    qc.h(0)

    # Measure the circuit
    qc.measure(0, 0)

    # Draw the circuit
    print(qc.draw())

    # Simulate the circuit
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(qc, simulator, shots=1000)
    result = job.result()

    # Get the counts
    counts = result.get_counts(qc)
    #print("Measurement results:", counts)

    # Plot the outcomes
    plt.bar([list(counts.keys())[1],list(counts.keys())[0]], [list(counts.values())[1],list(counts.values())[0]])
    plt.xlabel('Outcome')
    plt.ylabel('Frequency')
    plt.title('Measurement Outcomes')
    plt.show()

def qc_HTH():
    # Create a quantum circuit with 1 qubit and 1 classical bit
    qc = QuantumCircuit(1, 1)

    # Initialize qubit in |0⟩ state
    # No need to explicitly initialize in |0⟩ state as it's the default state

    # Apply Hadamard gate
    qc.h(0)

    # Apply T gate (T gate is a rotation around Z-axis by π/4)
    qc.p(pi/4, 0)  # Equivalent to T gate

    # Apply Hadamard gate again
    qc.h(0)

    # Measure qubit in Z basis
    qc.measure(0, 0)

    # Draw the circuit
    print(qc.draw())

    # Simulate the circuit
    simulator = Aer.get_backend('qasm_simulator')
    job = execute(qc, simulator, shots=1000)
    result = job.result()

    # Get the counts
    counts = result.get_counts(qc)

    # Plot the outcomes
    plt.bar([list(counts.keys())[1],list(counts.keys())[0]], [list(counts.values())[1],list(counts.values())[0]])
    #plt.bar(counts.keys(),counts.values())
    plt.xlabel('Outcome')
    plt.ylabel('Frequency')
    plt.title('Measurement Outcomes')
    plt.show()