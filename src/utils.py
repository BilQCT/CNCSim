from __future__ import annotations

import itertools
from math import isclose
from typing import Sequence, Union

import numpy as np
from qutip import Qobj
from qutip.measurement import measure

# Quantum Gates
PAULI_X = np.array([[0, 1], [1, 0]])
PAULI_Y = np.array([[0, -1j], [1j, 0]])
PAULI_Z = np.array([[1, 0], [0, -1]])
H_GATE = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
T_GATE = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])


def decimal_to_binary_array(decimal: int, width: int) -> np.ndarray:
    """Converts a decimal number to a numpy array of 0s and 1s. For example 5 in width 4 is converted to [0, 1, 0, 1].
    Width is required to fill the leading 0s.

    Args:
        decimal (int): Decimal number.
        width (int): Width of the binary array.

    Returns:
        np.ndarray: Numpy array of 0s and 1s.
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
        int: Decimal number.
    """

    for i in binary_array:
        if i != 0 and i != 1:
            raise RuntimeError("Array is not a binary array")

    binary_str = "".join(map(str, binary_array))
    return int(binary_str, 2)


def pauli_str_to_pauli_array(pauli_string: str) -> np.ndarray:
    """Converts a string of Pauli operators to a numpy array of 0s and 1s.
    For example "XZ" is converted to [1, 0, 0, 1]. The string is case insensitive.

    Args:
        pauli_string (str): String of Pauli operators.

    Returns:
        np.ndarray: Numpy array of 0s and 1s.
    """
    if not all([i in ["X", "Y", "Z", "I"] for i in pauli_string]):
        raise RuntimeError("Invalid Pauli string. It must contain only X, Y, Z, and I.")

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
            raise RuntimeError("Invalid Pauli string")

    pauli_array = np.concatenate((a_x, a_z)).astype(int)

    return pauli_array


def qutip_simuation(
    initial_state: Qobj, measurements: list[Qobj], num_simulations: int
) -> dict[str, int]:
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
            bsf = pauli_str_to_pauli_array(identifier)
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
    def associated_integer(self) -> int:
        # TODO: Change it to enumeration that goes like III, IIX, IIY, IIZ, IXI, IXY, ...
        """Associated integer of the Pauli which is the decimal value of bsf of the Pauli when
        it is read as a binary number. Assigning a unique integer to each Pauli operator is useful. This can be seen
        as enumerating the Pauli basis. So we can write operators in Pauli basis using this enumeration.
        """
        return binary_array_to_decimal(self.bsf)

    @classmethod
    def identity(cls, n: int) -> Pauli:
        """Returns the identity operator for n qubits."""
        return cls(np.zeros(2 * n))

    @classmethod
    def from_integer(cls, n: int, associated_integer: int) -> Pauli:
        # TODO: Change it to enumeration that goes like III, IIX, IIY, IIZ, IXI, IXY, ...
        """Creates a Pauli with a given number of qubits and an associated integer."""
        return cls(decimal_to_binary_array(associated_integer, 2 * n))

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
        return self.calculate_gamma(other) % 2

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
            + self.bsf[: self.n].dot(other.bsf[self.n :])
        ) % 2

    def __str__(self) -> str:
        pauli_str = "Pauli Operator: "
        for i in range(self.n):
            if self.bsf[i] == 1 and self.bsf[i + self.n] == 1:
                pauli_str += "Y"
            elif self.bsf[i + self.n] == 1:
                pauli_str += "Z"
            elif self.bsf[i] == 1:
                pauli_str += "X"
            else:
                pauli_str += "I"
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
