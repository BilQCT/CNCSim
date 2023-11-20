from __future__ import annotations
import numpy as np

# Quantum Gates
PAULI_X = np.array([[0, 1], [1, 0]])
PAULI_Y = np.array([[0, -1j], [1j, 0]])
PAULI_Z = np.array([[1, 0], [0, -1]])
H_GATE = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
T_GATE = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]])


def decimal_to_binary_array(decimal: int, width: int) -> np.ndarray:
    binary_str = np.binary_repr(decimal, width=width)
    return np.array(list(map(int, binary_str)))


def binary_array_to_decimal(binary_array: np.ndarray) -> int:
    binary_str = "".join(map(str, binary_array))
    return int(binary_str, 2)


class PauliOperator:
    def __init__(self, a: np.ndarray) -> None:
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
        return binary_array_to_decimal(self.a)

    def get_operator(self) -> np.ndarray:
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

    def calculate_beta(self, other: PauliOperator) -> int:
        if self.n != other.n:
            raise RuntimeError("Size of the operators is not correct")
        return (self.a_z.dot(other._a_x) - self._a_x.dot(other.a_z)) % 4

    def calculate_omega(self, other: PauliOperator) -> int:
        if self.n != other.n:
            raise RuntimeError("Size of the operators is not correct")
        return (self.a_z.dot(other.a_x) + self.a_x.dot(other.a_z)) % 2

    def __repr__(self) -> str:
        return f"PauliOperator({self.a_x}, {self.a_z})"

    def __eq__(self, other: PauliOperator) -> bool:
        return np.array_equal(self.a, other.a)
    
    def __add__(self, other: PauliOperator) -> PauliOperator:
        added_a = (self.a + other.a) % 2
        return PauliOperator(added_a)

    def __hash__(self) -> int:
        return hash(self.a.tobytes())