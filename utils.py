from __future__ import annotations
import numpy as np
from typing import Union
import itertools

# Quantum Gates
PAULI_X = np.array([[0, 1], [1, 0]])
PAULI_Y = np.array([[0, -1j], [1j, 0]])
PAULI_Z = np.array([[1, 0], [0, -1]])
H_GATE = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
T_GATE = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])


def decimal_to_binary_array(decimal: int, width: int) -> np.ndarray:
    """
    Converts a decimal number to a numpy array of 0s and 1s. For example 5 in width 4 is converted to np.array([0, 1, 0, 1]). Width is required to fill the leading 0s.

    Parameters:
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
    """
    Converts a numpy array of 0s and 1s to a decimal number. For example np.array([0, 1, 0, 1]) is converted to 5.

    Parameters:
    binary_array (np.ndarray): Numpy array of 0s and 1s.

    Returns:
    int: Decimal number.
    """

    for i in binary_array:
        if i != 0 and i != 1:
            raise RuntimeError("Array is not a binary array")

    binary_str = "".join(map(str, binary_array))
    return int(binary_str, 2)


class PauliOperator:
    def __init__(self, identifier: Union[np.ndarray, str]) -> None:
        """
        Initializes the PauliOperator.

        Parameters:
        identifier (Union[np.ndarray, str]): Identifier of the Pauli operator.
            It can be a numpy array of 0s and 1s or a string of Pauli operators.
            For example, "XZ" is equivalent to np.array([1, 0, 0, 1]).
        """

        if isinstance(identifier, np.ndarray):
            a = identifier
        elif isinstance(identifier, str):
            a = self._pauli_string_to_pauli_array(identifier)
        else:
            raise RuntimeError("Invalid identifier")

        if len(a) < 0 or len(a) % 2 == 1:
            raise RuntimeError("Size of a is not correct")

        n = len(a) // 2
        self._n = n
        self._a = a
        self._a_x = a[:n]
        self._a_z = a[n:]

    @property
    def n(self) -> int:
        return self._n

    @property
    def a(self) -> np.ndarray:
        return self._a

    @property
    def a_x(self) -> np.ndarray:
        return self._a_x

    @property
    def a_z(self) -> np.ndarray:
        return self._a_z

    @property
    def order(self) -> int:
        """
        Order of the Pauli operator is the corresponding decimal value of the binary array a when it is read as a binary number.
        """
        return binary_array_to_decimal(self.a)

    def get_operator(self) -> np.ndarray:
        """
        Creates the matrix representation of the Pauli operator.

        Returns:
        np.ndarray: Matrix representation of the Pauli operator.
        """
        result = np.eye(1)
        for i in range(self.n):
            temp = np.eye(2)
            if self.a_x[i] == 1:
                temp = temp @ PAULI_X
            if self.a_z[i] == 1:
                temp = temp @ PAULI_Z
            result = np.kron(result, temp)

        result = 1j ** self.a_x.dot(self.a_z) * result
        return result

    def calculate_gamma(self, other: PauliOperator) -> int:
        """
        Calculates the gamma value that arises when two Pauli operators are multiplied. T_a T_b = (i)^gamma T_(a+b).

        Parameters:
        other (PauliOperator): The other Pauli operator.

        Returns:
        int: gamma value (in modulo 4).
        """
        if self.n != other.n:
            raise RuntimeError("Size of the operators is not correct")
        return (self.a_z.dot(other._a_x) - self._a_x.dot(other.a_z)) % 4

    def calculate_beta(self, other: PauliOperator) -> int:
        """
        Calculates the gamma value that arises when two commuting Pauli operators are multiplied. T_a T_b = (-1)^beta T_(a+b).

        Parameters:
        other (PauliOperator): The other Pauli operator.

        Returns:
        int: beta value (in modulo 2).
        """
        if self.calculate_omega(other) != 0:
            raise RuntimeError("The operators are not commuting")

        return self.calculate_gamma(other) % 2

    def calculate_omega(self, other: PauliOperator) -> int:
        """
        Calculates the omega value that determines if two Pauli operators are commuting or anticommuting. T_a T_b = (-1)^omega T_b T_a.

        Parameters:
        other (PauliOperator): The other Pauli operator.

        Returns:
        int: omega value (in modulo 2).
        """
        if self.n != other.n:
            raise RuntimeError("Size of the operators is not correct")
        return (self.a_z.dot(other.a_x) + self.a_x.dot(other.a_z)) % 2
    
    def find_commuting_paulis(self) -> list[PauliOperator]:
        """
        Finds the Pauli operators that commutes with the instance Pauli operator.

        Returns:
        list[PauliOperator]: List of commuting Pauli operators.
        """
        result = []
        for p in itertools.product([0, 1], repeat=2 * self.n):
            pauli = PauliOperator(np.array(p))
            if pauli.calculate_omega(self) == 0:
                result.append(pauli)

        return result

    def _pauli_string_to_pauli_array(self, pauli_string: str) -> np.ndarray:
        """
        Converts a string of Pauli operators to a numpy array of 0s and 1s. For example "XZ" is converted to np.array([1, 0, 0, 1]). The string is case insensitive.

        Parameters:
        pauli_string (str): String of Pauli operators.

        Returns:
        np.ndarray: Numpy array of 0s and 1s.
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
                raise RuntimeError("Invalid Pauli string")

        pauli_array = np.concatenate((a_x, a_z)).astype(int)

        return pauli_array

    def __str__(self) -> str:
        """
        String representation of the Pauli operator in the form of "PauliOperator: XZII".

        Returns:
        str: String representation of the Pauli operator.
        """
        pauli_str = "PauliOperator: "
        for i in range(self.n):
            if self.a_x[i] == 1 and self.a_z[i] == 1:
                pauli_str += "Y"
            elif self.a_z[i] == 1:
                pauli_str += "Z"
            elif self.a_x[i] == 1:
                pauli_str += "X"
            else:
                pauli_str += "I"
        return pauli_str

    def __repr__(self) -> str:
        return self.__str__()

    def __eq__(self, other: PauliOperator) -> bool:
        return np.array_equal(self.a, other.a)

    def __add__(self, other: PauliOperator) -> PauliOperator:
        """
        Addition of the arrays that represent the Pauli operators. The addition is done modulo 2. For given a representing T_a and b representing T_b
        it gives (a+b) representing T_(a+b).

        Parameters:
        other (PauliOperator): The other Pauli operator.

        Returns:
        PauliOperator: Result of the addition.
        """
        added_a = (self.a + other.a) % 2
        return PauliOperator(added_a)

    def __hash__(self) -> int:
        return hash(self.a.tobytes())
    

def get_canonical_isotropic_generators(n: int) -> list[PauliOperator]:
    """
    Returns the canonical isotropic generators for the given number of qubits.

    Parameters:
    n (int): Number of qubits.

    Returns:
    list[PauliOperator]: List of canonical isotropic generators. The generators are
        X_1, X_2, ..., X_n
    """
    if n < 0:
        raise RuntimeError("Number of qubits cannot be negative")

    if n == 0:
        return []

    result = []
    for i in range(n):
        a = np.zeros(2 * n)
        a[i] = 1
        result.append(PauliOperator(a))

    return result





if __name__ == '__main__':
    pass
         

