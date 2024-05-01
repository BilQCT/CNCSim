from __future__ import annotations

import io
import os
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from math import isclose
from typing import Sequence, Union

import h5py
import numpy as np
from qutip import Qobj
from qutip.measurement import measure

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
            if not all(isinstance(i, int) for i in identifier):
                raise RuntimeError("Sequence elements must contain only integers")
            if not all([i in [0, 1] for i in identifier]):
                raise RuntimeError("Sequence elements must contain only 0s and 1s")
            bsf = np.array(identifier)
        elif isinstance(identifier, np.ndarray):
            if not all([i in [0, 1] for i in identifier]):
                raise RuntimeError("Numpy Array elements must contain only 0s and 1s")
            bsf = identifier
        else:
            raise RuntimeError("Invalid identifier")

        if not all([i in [0, 1] for i in bsf]):
            raise RuntimeError("Array is not a binary array")

        if len(bsf) < 0 or len(bsf) % 2 == 1:
            raise RuntimeError(
                "Size of a is not correct. It must be an positive even number."
            )

        n = len(bsf) // 2
        self._n = n
        self._bsf = bsf

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
        # TODO: Change it to enumeration that goes like III, IIX, IIY, IIZ, IXI, IXY, ...
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

    @classmethod
    def identity(cls, n: int) -> Pauli:
        """Returns the identity operator for n qubits."""
        return cls(np.zeros(2 * n))

    @classmethod
    def from_basis_order(cls, n: int, basis_order: int) -> Pauli:
        # TODO: Change it to enumeration that goes like III, IIX, IIY, IIZ, IXI, IXY, ...
        """Creates a Pauli with a given number of qubits and basis order."""
        if basis_order < 0 or basis_order >= 4**n:
            raise RuntimeError(
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

        result = 1j ** self.bsf[: self.n].dot(self.bsf[self.n :]) * result
        return result

    def calculate_gamma(self, other: Pauli) -> int:
        """Calculates the gamma value that arises when two Pauli operators are multiplied.

        Note:
            .. math::
                T_a T_b = (-1)^{\\gamma} T_{a + b}

        Args:
            other (Pauli): The other Pauli operator.

        Returns:
            int: gamma value (in modulo 4).
        """
        if self.n != other.n:
            raise RuntimeError("Size of the operators is not correct")
        return (
            self.bsf[self.n :].dot(other.bsf[: self.n])
            - self.bsf[: self.n].dot(other.bsf[self.n :])
        ) % 4

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
            raise RuntimeError("The operators are not commuting")

        # TODO: Write a neater algorithm
        pauli_str = pauli_bsf_to_str(self.bsf)
        other_pauli_str = pauli_bsf_to_str(other.bsf)
        count = 0
        for paulis in zip(pauli_str, other_pauli_str):
            if paulis == ("X", "Y") or paulis == ("Y", "Z") or paulis == ("Z", "X"):
                count += 1
            elif paulis == ("Y", "X") or paulis == ("Z", "Y") or paulis == ("X", "Z"):
                count -= 1

        count = count % 4
        return count // 2

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
            raise RuntimeError("Size of the operators is not correct")
        return (
            self.bsf[self.n :].dot(other.bsf[: self.n])
            - self.bsf[: self.n].dot(other.bsf[self.n :])
        ) % 2

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

    zip_file = f"../maximal_cnc_matrices/all_maximal_cncs_matrix_{n}.zip"
    jld_file = f"all_maximal_cncs_matrix_{n}.jld"

    if not os.path.exists(zip_file):
        raise RuntimeError(
            "There is no precomputed matrix of all maximal CNC atates for the given number of qubits."
        )

    with zipfile.ZipFile(zip_file, "r") as zip_ref:
        with zip_ref.open(jld_file) as file:
            with h5py.File(io.BytesIO(file.read()), "r") as f:
                # cast to numpy array
                return np.array(f["matrix"]).T


def decimal_to_binary_array(decimal: int, width: int) -> np.ndarray:
    """Converts a decimal number to a numpy array of 0s and 1s. For example 5 in width 4 is converted to [0, 1, 0, 1].
    Width is required to fill the leading 0s.

    Args:
        decimal (int): Decimal number.
        width (int): Width of the binary array.

    Returns:
        np.ndarray: Binary array of the decimal number.
    """
    if decimal < 0:
        raise RuntimeError("Decimal number cannot be negative")
    if decimal >= 2**width:
        raise RuntimeError("Decimal number is too large for the given width")

    binary_str = np.binary_repr(decimal, width=width)
    return np.array(list(map(int, binary_str)))


def binary_array_to_decimal(binary_array: np.ndarray) -> int:
    """Converts a numpy array of 0s and 1s to a decimal number. For example, [0, 1, 0, 1] is converted to 5.

    Args:
        binary_array (np.ndarray): Numpy array of 0s and 1s.

    Returns:
        int: Decimal number of the binary array when it is read as a binary number.
    """

    for i in binary_array:
        if i != 0 and i != 1:
            raise RuntimeError("Array is not a binary array")

    binary_str = "".join(map(str, binary_array))
    return int(binary_str, 2)


def pauli_str_to_bsf(pauli_string: str) -> np.ndarray:
    """Converts a string of Pauli operators to a array of 0s and 1s which is its Binary Symplectic Form.
    For example "XZ" is converted to [1, 0, 0, 1]. The string is case insensitive.

    Args:
        pauli_string (str): String representation of the Pauli operator.

    Returns:
        np.ndarray: Binary Symplectic Form of the Pauli operator.
    """

    a_x = np.zeros(len(pauli_string))
    a_z = np.zeros(len(pauli_string))

    pauli_string = pauli_string.upper()

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
            raise RuntimeError("Invalid Pauli string. It must contain only X, Y, Z, I.")

    pauli_array = np.concatenate((a_x, a_z)).astype(int)

    return pauli_array


def pauli_bsf_to_str(pauli_bsf: np.ndarray) -> str:
    """Converts Binary Symplectic Form of a Pauli operator to its string representation.
    For example [1, 0, 0, 1] is converted to "XZ".

    Args:
        pauli_bsf (np.ndarray): Binary Symplectic Form of the Pauli operator.

    Returns:
        str: String representation of the Pauli operator.
    """
    if len(pauli_bsf) % 2 == 1 or len(pauli_bsf) <= 0:
        raise RuntimeError(
            "Size of the given Binary Symplectic Form is not correct. It must be a positive even number."
        )

    if not all([i in [0, 1] for i in pauli_bsf]):
        raise RuntimeError("Array is not a binary array")

    pauli_str = ""

    n = len(pauli_bsf) // 2
    for i in range(n):
        if pauli_bsf[i] == 1 and pauli_bsf[i + n] == 1:
            pauli_str += "Y"
        elif pauli_bsf[i + n] == 1:
            pauli_str += "Z"
        elif pauli_bsf[i] == 1:
            pauli_str += "X"
        else:
            pauli_str += "I"

    return pauli_str


def qutip_simuation(
    initial_state: Qobj, measurements: list[Qobj], num_simulations: int
) -> dict[str, int]:
    """Runs a simulation with the given initial state and measurements using QuTiP. The simulation is repeated
    num_simulations times and the counts of the measurement outcomes are returned.

    Args:
        initial_state (Qobj): Initial state of the system.
        measurements (list[Qobj]): List of measurement operators.
        num_simulations (int): Number of simulations.

    Returns:
        dict[str, int]: Dictionary of measurement outcomes and their counts.
    """
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


def get_n_from_pauli_basis_representation(
    pauli_basis_representation: np.ndarray,
) -> int:
    """Calculates the number of qubits from the Pauli basis representation of a state.

    Args:
        pauli_basis_representation (np.ndarray): Pauli basis representation of a state.

    Returns:
        int: Number of qubits.
    """
    n = 1
    l = 4
    while l < len(pauli_basis_representation):
        n += 1
        l *= 4

    if l != len(pauli_basis_representation):
        raise RuntimeError("The size of the basis representation must be a power of 4.")

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
