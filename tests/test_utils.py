import numpy as np
import pytest

from src.utils import PAULI_X, PAULI_Y, PAULI_Z, Pauli


def test_pauli_init():
    # Initialization by Pauli String 1
    pauli = Pauli("X")
    assert pauli.n == 1
    assert np.array_equal(pauli.bsf, np.array([1, 0]))

    # Initialization by Pauli String 2
    pauli = Pauli("IXYZ")
    assert pauli.n == 4
    assert np.array_equal(pauli.bsf, [0, 1, 1, 0, 0, 0, 1, 1])

    # Initialization by Pauli String 3
    pauli = Pauli("ixyz")
    assert pauli.n == 4
    assert np.array_equal(pauli.bsf, [0, 1, 1, 0, 0, 0, 1, 1])

    # Initialization by Binary Symplectic form as List 1
    pauli = Pauli([1, 0])
    assert pauli.n == 1
    assert np.array_equal(pauli.bsf, np.array([1, 0]))

    # Initialization by Binary Symplectic form as List 2
    pauli = Pauli([0, 1, 1, 0, 0, 0, 1, 1])
    assert pauli.n == 4
    assert np.array_equal(pauli.bsf, [0, 1, 1, 0, 0, 0, 1, 1])

    # Initialization by Binary Symplectic form as Tuple 1
    pauli = Pauli((1, 0))
    assert pauli.n == 1
    assert np.array_equal(pauli.bsf, np.array([1, 0]))

    # Initialization by Binary Symplectic form as Tuple 2
    pauli = Pauli((0, 1, 1, 0, 0, 0, 1, 1))
    assert pauli.n == 4
    assert np.array_equal(pauli.bsf, [0, 1, 1, 0, 0, 0, 1, 1])

    # Initialization by Binary Symplectic form as Numpy array 1
    pauli = Pauli(np.array([1, 0]))
    assert pauli.n == 1
    assert np.array_equal(pauli.bsf, np.array([1, 0]))

    # Initialization by Binary Symplectic form as Numpy array 2
    pauli = Pauli(np.array([0, 1, 1, 0, 0, 0, 1, 1]))
    assert pauli.n == 4
    assert np.array_equal(pauli.bsf, [0, 1, 1, 0, 0, 0, 1, 1])

    # Array elemenet must be integers
    with pytest.raises(ValueError):
        Pauli([1, "X"])

    # Array elemenet must be 0, 1
    with pytest.raises(ValueError):
        Pauli([0, 1, 3])

    # Identifier for Pauli must be one of the following: str, Sequence[int], np.ndarray
    with pytest.raises(RuntimeError):
        Pauli({"X"})

    # Identifier for Pauli must be one of the following: str, Sequence[int], np.ndarray
    with pytest.raises(RuntimeError):
        Pauli(1)

    # When Identifier is given in Binary Symplectic form, its length must be a positive even number
    with pytest.raises(ValueError):
        Pauli(np.array([1, 0, 1]))


def test_pauli_equality():
    # Positive tests
    pauli_1 = Pauli("I")
    pauli_2 = Pauli("I")
    assert pauli_1 == pauli_2

    pauli_1 = Pauli("X")
    pauli_2 = Pauli([1, 0])
    assert pauli_1 == pauli_2

    pauli_1 = Pauli([0, 1, 0, 1, 0, 1])
    pauli_2 = Pauli((0, 1, 0, 1, 0, 1))
    pauli_3 = Pauli(np.array([0, 1, 0, 1, 0, 1]))
    assert pauli_1 == pauli_2 == pauli_3

    pauli_1 = Pauli("IXYZ")
    pauli_2 = Pauli([0, 1, 1, 0, 0, 0, 1, 1])
    assert pauli_1 == pauli_2

    # Negative tests
    pauli_1 = Pauli("I")
    pauli_2 = Pauli("X")
    assert pauli_1 != pauli_2

    pauli_1 = Pauli("I")
    pauli_2 = Pauli("II")
    assert pauli_1 != pauli_2


def test_pauli_basis_order():
    # Check the basis order for Pauli strings
    assert Pauli("I").basis_order == 0
    assert Pauli("X").basis_order == 1
    assert Pauli("Y").basis_order == 2
    assert Pauli("Z").basis_order == 3

    assert Pauli("II").basis_order == 0
    assert Pauli("IX").basis_order == 1
    assert Pauli("YY").basis_order == 10

    assert Pauli("IIII").basis_order == 0
    assert Pauli("XYZI").basis_order == 64 + 2 * 16 + 3 * 4

    # Check that the basis order is unique and in the range [0, 256)
    local_paulis_str = ["I", "X", "Y", "Z"]
    found_basis_order = set()
    for p1 in local_paulis_str:
        for p2 in local_paulis_str:
            for p3 in local_paulis_str:
                for p4 in local_paulis_str:
                    pauli_str = p1 + p2 + p3 + p4
                    pauli = Pauli(pauli_str)
                    assert pauli.basis_order not in found_basis_order
                    assert pauli.basis_order >= 0
                    assert pauli.basis_order < 256
                    found_basis_order.add(pauli.basis_order)


def test_pauli_identity():
    # Check the convenience method for the identity operator
    for n in range(1, 10):
        identity_n = Pauli.identity(n)
        assert identity_n.basis_order == 0
        assert identity_n.n == n
        assert np.array_equal(identity_n.bsf, np.zeros(2 * n))


def test_pauli_from_basis_order():
    # 1-qubit Paulis
    assert Pauli.from_basis_order(1, 0) == Pauli("I")
    assert Pauli.from_basis_order(1, 1) == Pauli("X")

    # 2-qubit Paulis
    assert Pauli.from_basis_order(2, 0) == Pauli("II")
    assert Pauli.from_basis_order(2, 7) == Pauli("XZ")

    # 3-qubit Paulis
    assert Pauli.from_basis_order(3, 0) == Pauli("III")
    assert Pauli.from_basis_order(3, 22) == Pauli("XXY")


def test_pauli_get_operator():
    # 1 - qubit Paulis
    assert np.array_equal(Pauli("I").get_operator(), np.eye(2))
    assert np.array_equal(Pauli("X").get_operator(), PAULI_X)
    assert np.array_equal(Pauli("Y").get_operator(), PAULI_Y)
    assert np.array_equal(Pauli("Z").get_operator(), PAULI_Z)

    # 2 - qubit Paulis
    assert np.array_equal(Pauli("II").get_operator(), np.eye(4))

    ix = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    assert np.array_equal(Pauli("IX").get_operator(), ix)
    assert np.array_equal(Pauli("IX").get_operator(), np.kron(np.eye(2), PAULI_X))

    zx = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, -1, 0]])
    assert np.array_equal(Pauli("ZX").get_operator(), zx)
    assert np.array_equal(Pauli("ZX").get_operator(), np.kron(PAULI_Z, PAULI_X))

    # 3 - qubit Paulis
    assert np.array_equal(Pauli("III").get_operator(), np.eye(8))

    xyz = np.array(
        [
            [0, 0, 0, 0, 0, 0, -1j, 0],
            [0, 0, 0, 0, 0, 0, 0, 1j],
            [0, 0, 0, 0, 1j, 0, 0, 0],
            [0, 0, 0, 0, 0, -1j, 0, 0],
            [0, 0, -1j, 0, 0, 0, 0, 0],
            [0, 0, 0, 1j, 0, 0, 0, 0],
            [1j, 0, 0, 0, 0, 0, 0, 0],
            [0, -1j, 0, 0, 0, 0, 0, 0],
        ]
    )
    assert np.array_equal(Pauli("XYZ").get_operator(), xyz)
    assert np.array_equal(
        Pauli("XYZ").get_operator(), np.kron(np.kron(PAULI_X, PAULI_Y), PAULI_Z)
    )


def test_pauli_calculate_gamma():
    # TODO
    # # 1 - qubit Paulis
    # pauli_1 = Pauli("I")
    # pauli_2 = Pauli("I")
    # assert pauli_1.calculate_gamma(pauli_2) == 0

    # pauli_1 = Pauli("X")
    # pauli_2 = Pauli("Y")
    # assert pauli_1.calculate_gamma(pauli_2) == 1

    # pauli_1 = Pauli("Z")
    # pauli_2 = Pauli("Y")
    # assert pauli_1.calculate_gamma(pauli_2) == 3

    # # 2 - qubit Paulis
    # pauli_1 = Pauli("II")
    # pauli_2 = Pauli("II")
    # assert pauli_1.calculate_gamma(pauli_2) == 0

    # pauli_1 = Pauli("IX")
    # pauli_2 = Pauli("YI")
    # assert pauli_1.calculate_gamma(pauli_2) == 0

    # pauli_1 = Pauli("IX")
    # pauli_2 = Pauli("IY")
    # assert pauli_1.calculate_gamma(pauli_2) == 1

    # pauli_1 = Pauli("XX")
    # pauli_2 = Pauli("YY")
    # assert pauli_1.calculate_gamma(pauli_2) == 2

    # pauli_1 = Pauli("XY")
    # pauli_2 = Pauli("YX")
    # assert pauli_1.calculate_gamma(pauli_2) == 0

    # # 3 - qubit Paulis
    # pauli_1 = Pauli("III")
    # pauli_2 = Pauli("III")
    # assert pauli_1.calculate_gamma(pauli_2) == 0

    # pauli_1 = Pauli("IXY")
    # pauli_2 = Pauli("YXI")
    # assert pauli_1.calculate_gamma(pauli_2) == 0

    # pauli_1 = Pauli("IXY")
    # pauli_2 = Pauli("IYX")
    # assert pauli_1.calculate_gamma(pauli_2) == 0

    # pauli_1 = Pauli("XYZ")
    # pauli_2 = Pauli("III")
    # assert pauli_1.calculate_gamma(pauli_2) == 0

    # pauli_1 = Pauli("XYZ")
    # pauli_2 = Pauli("YZX")
    # assert pauli_1.calculate_gamma(pauli_2) == 3

    # pauli_1 = Pauli("XYZ")
    # pauli_2 = Pauli("ZXY")
    # assert pauli_1.calculate_gamma(pauli_2) == 1
    pass


def test_pauli_calculate_beta():
    # TODO
    pass


def test_pauli_calculate_omega():
    # TODO
    pass


def test_pauli_str():
    # TODO
    pass


def test_pauli_addition():
    # TODO
    pass


def test_pauli_hash():
    # TODO
    pass


def test_load_all_maximal_cncs_matrix():
    # TODO
    pass


def test_pauli_str_to_bsf():
    # TODO
    pass


def test_pauli_bsf_to_str():
    # TODO
    pass


def test_qutip_simulation():
    # TODO
    pass


def get_n_from_pauli_basis_representation():
    # TODO
    pass


def test_decomposition_element():
    # TODO
    pass


if __name__ == "__main__":
    test_pauli_init()
    test_pauli_equality()
    test_pauli_basis_order()
    test_pauli_identity()
    test_pauli_from_basis_order()
    test_pauli_get_operator()
    test_pauli_calculate_gamma()
    test_pauli_calculate_beta()
    test_pauli_calculate_omega()
    test_pauli_str()
    test_pauli_addition()
    test_pauli_hash()
    test_load_all_maximal_cncs_matrix()
    test_pauli_str_to_bsf()
    test_pauli_bsf_to_str()
    test_qutip_simulation()
    get_n_from_pauli_basis_representation()
    test_decomposition_element()
