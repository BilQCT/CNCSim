import numpy as np
import pytest

from src.utils import (
    PAULI_X,
    PAULI_Y,
    PAULI_Z,
    DecompositionElement,
    Pauli,
    PhasePointOperator,
    get_n_from_pauli_basis_representation,
    load_all_maximal_cncs_matrix,
    pauli_bsf_to_str,
    pauli_str_to_bsf,
    qutip_simuation,
)

from qutip import Qobj, identity, sigmaz, sigmay, tensor


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
    pauli = Pauli([0, 1])
    assert pauli.n == 1
    assert np.array_equal(pauli.bsf, np.array([0, 1]))

    # Initialization by Binary Symplectic form as List 2
    pauli = Pauli([0, 1, 1, 0, 0, 0, 1, 1])
    assert pauli.n == 4
    assert np.array_equal(pauli.bsf, [0, 1, 1, 0, 0, 0, 1, 1])

    # Initialization by Binary Symplectic form as Tuple 1
    pauli = Pauli((0, 1))
    assert pauli.n == 1
    assert np.array_equal(pauli.bsf, np.array([0, 1]))

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


def test_phase():
    # 1 - qubit Paulis
    pauli = Pauli("I")
    assert pauli.phase == 0

    pauli = Pauli("X")
    assert pauli.phase == 0

    pauli = Pauli("Y")
    assert pauli.phase == 1

    pauli = Pauli("Z")
    assert pauli.phase == 0

    # 2 - qubit Paulis
    pauli = Pauli("II")
    assert pauli.phase == 0

    pauli = Pauli("IX")
    assert pauli.phase == 0

    pauli = Pauli("XY")
    assert pauli.phase == 1

    pauli = Pauli("YZ")
    assert pauli.phase == 1

    pauli = Pauli("YY")
    assert pauli.phase == 2

    # More qubit Paulis
    pauli = Pauli("XYXXYZ")
    assert pauli.phase == 2

    pauli = Pauli("XYZXYY")
    assert pauli.phase == 3


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
    # 1 - qubit Paulis
    pauli_1 = Pauli("I")
    pauli_2 = Pauli("I")
    summed_pauli = pauli_1 + pauli_2
    gamma = pauli_1.calculate_gamma(pauli_2)
    assert gamma == 0
    assert np.array_equal(
        pauli_1.get_operator() @ pauli_2.get_operator(),
        (1j**gamma) * summed_pauli.get_operator(),
    )

    pauli_1 = Pauli("X")
    pauli_2 = Pauli("Y")
    summed_pauli = pauli_1 + pauli_2
    gamma = pauli_1.calculate_gamma(pauli_2)
    assert gamma == 1
    assert np.array_equal(
        pauli_1.get_operator() @ pauli_2.get_operator(),
        (1j**gamma) * summed_pauli.get_operator(),
    )

    pauli_1 = Pauli("Z")
    pauli_2 = Pauli("Y")
    summed_pauli = pauli_1 + pauli_2
    gamma = pauli_1.calculate_gamma(pauli_2)
    assert gamma == 3
    assert np.array_equal(
        pauli_1.get_operator() @ pauli_2.get_operator(),
        (1j**gamma) * summed_pauli.get_operator(),
    )

    # 2 - qubit Paulis
    pauli_1 = Pauli("II")
    pauli_2 = Pauli("II")
    summed_pauli = pauli_1 + pauli_2
    gamma = pauli_1.calculate_gamma(pauli_2)
    assert gamma == 0
    assert np.array_equal(
        pauli_1.get_operator() @ pauli_2.get_operator(),
        (1j**gamma) * summed_pauli.get_operator(),
    )

    pauli_1 = Pauli("IX")
    pauli_2 = Pauli("YI")
    summed_pauli = pauli_1 + pauli_2
    gamma = pauli_1.calculate_gamma(pauli_2)
    assert gamma == 0
    assert np.array_equal(
        pauli_1.get_operator() @ pauli_2.get_operator(),
        (1j**gamma) * summed_pauli.get_operator(),
    )

    pauli_1 = Pauli("IX")
    pauli_2 = Pauli("IY")
    summed_pauli = pauli_1 + pauli_2
    gamma = pauli_1.calculate_gamma(pauli_2)
    assert gamma == 1
    assert np.array_equal(
        pauli_1.get_operator() @ pauli_2.get_operator(),
        (1j**gamma) * summed_pauli.get_operator(),
    )

    pauli_1 = Pauli("XX")
    pauli_2 = Pauli("YY")
    summed_pauli = pauli_1 + pauli_2
    gamma = pauli_1.calculate_gamma(pauli_2)
    assert gamma == 2
    assert np.array_equal(
        pauli_1.get_operator() @ pauli_2.get_operator(),
        (1j**gamma) * summed_pauli.get_operator(),
    )

    pauli_1 = Pauli("XY")
    pauli_2 = Pauli("YX")
    summed_pauli = pauli_1 + pauli_2
    gamma = pauli_1.calculate_gamma(pauli_2)
    assert gamma == 0
    assert np.array_equal(
        pauli_1.get_operator() @ pauli_2.get_operator(),
        (1j**gamma) * summed_pauli.get_operator(),
    )

    # 3 - qubit Paulis
    pauli_1 = Pauli("III")
    pauli_2 = Pauli("III")
    summed_pauli = pauli_1 + pauli_2
    gamma = pauli_1.calculate_gamma(pauli_2)
    assert gamma == 0
    assert np.array_equal(
        pauli_1.get_operator() @ pauli_2.get_operator(),
        (1j**gamma) * summed_pauli.get_operator(),
    )

    pauli_1 = Pauli("IXY")
    pauli_2 = Pauli("YXI")
    summed_pauli = pauli_1 + pauli_2
    gamma = pauli_1.calculate_gamma(pauli_2)
    assert gamma == 0
    assert np.array_equal(
        pauli_1.get_operator() @ pauli_2.get_operator(),
        (1j**gamma) * summed_pauli.get_operator(),
    )

    pauli_1 = Pauli("IXY")
    pauli_2 = Pauli("IYX")
    summed_pauli = pauli_1 + pauli_2
    gamma = pauli_1.calculate_gamma(pauli_2)
    assert gamma == 0
    assert np.array_equal(
        pauli_1.get_operator() @ pauli_2.get_operator(),
        (1j**gamma) * summed_pauli.get_operator(),
    )

    pauli_1 = Pauli("XYZ")
    pauli_2 = Pauli("III")
    summed_pauli = pauli_1 + pauli_2
    gamma = pauli_1.calculate_gamma(pauli_2)
    assert gamma == 0
    assert np.array_equal(
        pauli_1.get_operator() @ pauli_2.get_operator(),
        (1j**gamma) * summed_pauli.get_operator(),
    )
    assert pauli_1.calculate_gamma(pauli_2) == 0

    pauli_1 = Pauli("XYZ")
    pauli_2 = Pauli("YZX")
    summed_pauli = pauli_1 + pauli_2
    gamma = pauli_1.calculate_gamma(pauli_2)
    assert gamma == 3
    assert np.array_equal(
        pauli_1.get_operator() @ pauli_2.get_operator(),
        (1j**gamma) * summed_pauli.get_operator(),
    )

    pauli_1 = Pauli("XYZ")
    pauli_2 = Pauli("ZXY")
    summed_pauli = pauli_1 + pauli_2
    gamma = pauli_1.calculate_gamma(pauli_2)
    assert gamma == 1
    assert np.array_equal(
        pauli_1.get_operator() @ pauli_2.get_operator(),
        (1j**gamma) * summed_pauli.get_operator(),
    )

    # Different lengths
    with pytest.raises(ValueError):
        Pauli("I").calculate_gamma(Pauli("XX"))


def test_pauli_calculate_beta():
    # 1 - qubit Paulis
    pauli_1 = Pauli("I")
    pauli_2 = Pauli("I")
    summed_pauli = pauli_1 + pauli_2
    beta = pauli_1.calculate_beta(pauli_2)
    assert beta == 0
    assert np.array_equal(
        pauli_1.get_operator() @ pauli_2.get_operator(),
        ((-1) ** beta) * summed_pauli.get_operator(),
    )

    pauli_1 = Pauli("I")
    pauli_2 = Pauli("X")
    summed_pauli = pauli_1 + pauli_2
    beta = pauli_1.calculate_beta(pauli_2)
    assert beta == 0
    assert np.array_equal(
        pauli_1.get_operator() @ pauli_2.get_operator(),
        ((-1) ** beta) * summed_pauli.get_operator(),
    )

    pauli_1 = Pauli("Z")
    pauli_2 = Pauli("I")
    summed_pauli = pauli_1 + pauli_2
    beta = pauli_1.calculate_beta(pauli_2)
    assert beta == 0
    assert np.array_equal(
        pauli_1.get_operator() @ pauli_2.get_operator(),
        ((-1) ** beta) * summed_pauli.get_operator(),
    )

    # 2 - qubit Paulis
    pauli_1 = Pauli("II")
    pauli_2 = Pauli("II")
    summed_pauli = pauli_1 + pauli_2
    beta = pauli_1.calculate_beta(pauli_2)
    assert beta == 0
    assert np.array_equal(
        pauli_1.get_operator() @ pauli_2.get_operator(),
        ((-1) ** beta) * summed_pauli.get_operator(),
    )

    pauli_1 = Pauli("IX")
    pauli_2 = Pauli("YI")
    summed_pauli = pauli_1 + pauli_2
    beta = pauli_1.calculate_beta(pauli_2)
    assert beta == 0
    assert np.array_equal(
        pauli_1.get_operator() @ pauli_2.get_operator(),
        ((-1) ** beta) * summed_pauli.get_operator(),
    )

    pauli_1 = Pauli("XY")
    pauli_2 = Pauli("YX")
    summed_pauli = pauli_1 + pauli_2
    beta = pauli_1.calculate_beta(pauli_2)
    assert beta == 0
    assert np.array_equal(
        pauli_1.get_operator() @ pauli_2.get_operator(),
        ((-1) ** beta) * summed_pauli.get_operator(),
    )

    pauli_1 = Pauli("XX")
    pauli_2 = Pauli("YY")
    summed_pauli = pauli_1 + pauli_2
    beta = pauli_1.calculate_beta(pauli_2)
    assert beta == 1
    assert np.array_equal(
        pauli_1.get_operator() @ pauli_2.get_operator(),
        ((-1) ** beta) * summed_pauli.get_operator(),
    )

    # 3 - qubit Paulis
    pauli_1 = Pauli("III")
    pauli_2 = Pauli("III")
    summed_pauli = pauli_1 + pauli_2
    beta = pauli_1.calculate_beta(pauli_2)
    assert beta == 0
    assert np.array_equal(
        pauli_1.get_operator() @ pauli_2.get_operator(),
        ((-1) ** beta) * summed_pauli.get_operator(),
    )

    pauli_1 = Pauli("XYZ")
    pauli_2 = Pauli("YZZ")
    summed_pauli = pauli_1 + pauli_2
    beta = pauli_1.calculate_beta(pauli_2)
    assert beta == 1
    assert np.array_equal(
        pauli_1.get_operator() @ pauli_2.get_operator(),
        ((-1) ** beta) * summed_pauli.get_operator(),
    )

    pauli_1 = Pauli("XXX")
    pauli_2 = Pauli("XZZ")
    summed_pauli = pauli_1 + pauli_2
    beta = pauli_1.calculate_beta(pauli_2)
    assert beta == 1
    assert np.array_equal(
        pauli_1.get_operator() @ pauli_2.get_operator(),
        ((-1) ** beta) * summed_pauli.get_operator(),
    )

    # Different lengths
    with pytest.raises(ValueError):
        Pauli("I").calculate_beta(Pauli("XX"))

    # Non-commuting Paulis
    with pytest.raises(ValueError):
        Pauli("X").calculate_beta(Pauli("Y"))

    # Non-commuting Paulis
    with pytest.raises(ValueError):
        Pauli("XXX").calculate_beta(Pauli("ZZZ"))


def test_pauli_calculate_omega():
    # 1 - qubit Paulis
    pauli_1 = Pauli("I")
    pauli_2 = Pauli("I")
    omega = pauli_1.calculate_omega(pauli_2)
    assert omega == 0

    pauli_1 = Pauli("X")
    pauli_2 = Pauli("Y")
    omega = pauli_1.calculate_omega(pauli_2)
    assert omega == 1

    pauli_1 = Pauli("Z")
    pauli_2 = Pauli("Y")
    omega = pauli_1.calculate_omega(pauli_2)
    assert omega == 1

    # 2 - qubit Paulis
    pauli_1 = Pauli("II")
    pauli_2 = Pauli("XI")
    omega = pauli_1.calculate_omega(pauli_2)
    assert omega == 0

    pauli_1 = Pauli("XY")
    pauli_2 = Pauli("YX")
    omega = pauli_1.calculate_omega(pauli_2)
    assert omega == 0

    # 3 - qubit Paulis
    pauli_1 = Pauli("XYZ")
    pauli_2 = Pauli("YZX")
    omega = pauli_1.calculate_omega(pauli_2)
    assert omega == 1

    pauli_1 = Pauli("XYZ")
    pauli_2 = Pauli("XYZ")
    omega = pauli_1.calculate_omega(pauli_2)
    assert omega == 0

    # Different lengths
    with pytest.raises(ValueError):
        Pauli("I").calculate_omega(Pauli("XX"))


def test_pauli_str():
    # 1 - qubit Paulis
    pauli = Pauli("I")
    assert str(pauli) == "Pauli Operator: I"

    pauli = Pauli("X")
    assert str(pauli) == "Pauli Operator: X"

    # More qubit Paulis
    pauli = Pauli("IIII")
    assert str(pauli) == "Pauli Operator: IIII"

    pauli = Pauli("XXYZ")
    assert str(pauli) == "Pauli Operator: XXYZ"


def test_pauli_addition():
    # 1 - qubit Paulis
    pauli_1 = Pauli("I")
    pauli_2 = Pauli("I")
    summed_pauli = pauli_1 + pauli_2
    assert summed_pauli == Pauli("I")

    pauli_1 = Pauli("X")
    pauli_2 = Pauli("Y")
    summed_pauli = pauli_1 + pauli_2
    assert summed_pauli == Pauli("Z")

    pauli_1 = Pauli("Z")
    pauli_2 = Pauli("Y")
    summed_pauli = pauli_1 + pauli_2
    assert summed_pauli == Pauli("X")

    # 2 - qubit Paulis
    pauli_1 = Pauli("II")
    pauli_2 = Pauli("XI")
    summed_pauli = pauli_1 + pauli_2
    assert summed_pauli == Pauli("XI")

    pauli_1 = Pauli("XY")
    pauli_2 = Pauli("YX")
    summed_pauli = pauli_1 + pauli_2
    assert summed_pauli == Pauli("ZZ")

    # 3 - qubit Paulis
    pauli_1 = Pauli("XYZ")
    pauli_2 = Pauli("YZX")
    summed_pauli = pauli_1 + pauli_2
    assert summed_pauli == Pauli("ZXY")

    pauli_1 = Pauli("XYZ")
    pauli_2 = Pauli("XYY")
    summed_pauli = pauli_1 + pauli_2
    assert summed_pauli == Pauli("IIX")

    # Different lengths
    with pytest.raises(ValueError):
        Pauli("I") + Pauli("XX")


def test_pauli_hash():
    # 1 - qubit Paulis
    pauli_1 = Pauli("I")
    assert hash(pauli_1) == hash("Pauli Operator: I")

    pauli_1 = Pauli("X")
    assert hash(pauli_1) == hash("Pauli Operator: X")

    # More qubit Paulis
    pauli_1 = Pauli("IIII")
    assert hash(pauli_1) == hash("Pauli Operator: IIII")

    pauli_1 = Pauli("XXYZ")
    assert hash(pauli_1) == hash("Pauli Operator: XXYZ")


def test_load_all_maximal_cncs_matrix():
    # 1 - qubit CNCs
    cncs_1 = load_all_maximal_cncs_matrix(1)
    assert cncs_1.shape == (4, 14)
    assert all(identity_value == 1 for identity_value in cncs_1[0, :])

    # 2 - qubit CNCs
    cncs_2 = load_all_maximal_cncs_matrix(2)
    assert cncs_2.shape == (16, 492)
    assert all(identity_value == 1 for identity_value in cncs_2[0, :])

    # 3 - qubit CNCs
    cncs_3 = load_all_maximal_cncs_matrix(3)
    assert cncs_3.shape == (64, 72216)
    assert all(identity_value == 1 for identity_value in cncs_3[0, :])

    # More qubit CNCs
    with pytest.raises(ValueError):
        cncs_4 = load_all_maximal_cncs_matrix(4)


def test_pauli_str_to_bsf():
    # 1 - qubit Paulis
    pauli_str = "I"
    assert np.array_equal(pauli_str_to_bsf(pauli_str), np.array([0, 0]))

    pauli_str = "X"
    assert np.array_equal(pauli_str_to_bsf(pauli_str), np.array([1, 0]))

    pauli_str = "Y"
    assert np.array_equal(pauli_str_to_bsf(pauli_str), np.array([1, 1]))

    pauli_str = "Z"
    assert np.array_equal(pauli_str_to_bsf(pauli_str), np.array([0, 1]))

    # 2 - qubit Paulis
    pauli_str = "II"
    assert np.array_equal(pauli_str_to_bsf(pauli_str), np.array([0, 0, 0, 0]))

    pauli_str = "IX"
    assert np.array_equal(pauli_str_to_bsf(pauli_str), np.array([0, 1, 0, 0]))

    pauli_str = "XY"
    assert np.array_equal(pauli_str_to_bsf(pauli_str), np.array([1, 1, 0, 1]))

    # More qubit Paulis
    pauli_str = "XYZXYY"
    assert np.array_equal(
        pauli_str_to_bsf(pauli_str),
        np.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1]),
    )

    # Invalid type
    with pytest.raises(RuntimeError):
        pauli_str_to_bsf(12)

    # Invalid value
    with pytest.raises(ValueError):
        pauli_str_to_bsf("A")

    # Invalid value
    with pytest.raises(ValueError):
        pauli_str_to_bsf("X1")


def test_pauli_bsf_to_str():
    # 1 - qubit Paulis
    pauli_bsf = np.array([0, 0])
    assert pauli_bsf_to_str(pauli_bsf) == "I"

    pauli_bsf = np.array([1, 0])
    assert pauli_bsf_to_str(pauli_bsf) == "X"

    pauli_bsf = np.array([1, 1])
    assert pauli_bsf_to_str(pauli_bsf) == "Y"

    pauli_bsf = np.array([0, 1])
    assert pauli_bsf_to_str(pauli_bsf) == "Z"

    # 2 - qubit Paulis
    pauli_bsf = np.array([0, 0, 0, 0])
    assert pauli_bsf_to_str(pauli_bsf) == "II"

    pauli_bsf = np.array([0, 1, 0, 0])
    assert pauli_bsf_to_str(pauli_bsf) == "IX"

    pauli_bsf = np.array([1, 1, 0, 1])
    assert pauli_bsf_to_str(pauli_bsf) == "XY"

    # More qubit Paulis
    pauli_bsf = np.array([1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1])
    assert pauli_bsf_to_str(pauli_bsf) == "XYZXYY"

    # Other types
    pauli_bsf = [1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1]
    assert pauli_bsf_to_str(pauli_bsf) == "XYZXYY"

    pauli_bsf = (1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1)
    assert pauli_bsf_to_str(pauli_bsf) == "XYZXYY"

    # Invalid type
    with pytest.raises(RuntimeError):
        pauli_bsf_to_str(12)

    # Invalid value
    with pytest.raises(ValueError):
        pauli_bsf_to_str(np.array([0, 0, 0, 2]))

    # Invalid length
    with pytest.raises(ValueError):
        pauli_bsf_to_str(np.array([0, 0, 0]))


def test_qutip_simulation():
    II = tensor(identity(2), identity(2))
    ZI = tensor(sigmaz(), identity(2))
    IZ = tensor(identity(2), sigmaz())
    ZZ = tensor(sigmaz(), sigmaz())
    YY = tensor(sigmay(), sigmay())

    rho = 1 / 4 * (II + ZI + IZ + ZZ)

    measurements = [YY, ZI]

    num_simulations = 2048
    counts_qutip = qutip_simuation(rho, measurements, num_simulations)

    # TODO - Add more tests


def get_n_from_pauli_basis_representation():
    basis_rep = [0, 1, 1, 0]
    assert get_n_from_pauli_basis_representation(basis_rep) == 1

    basis_rep = np.zeros(16)
    assert get_n_from_pauli_basis_representation(basis_rep) == 2

    basis_rep = np.zeros(64)
    assert get_n_from_pauli_basis_representation(basis_rep) == 3

    basis_rep = np.zeros(256)
    assert get_n_from_pauli_basis_representation(basis_rep) == 4

    # Invalid type
    with pytest.raises(RuntimeError):
        get_n_from_pauli_basis_representation("X")

    # Invalid type
    with pytest.raises(RuntimeError):
        get_n_from_pauli_basis_representation(1)

    # Invalid length
    with pytest.raises(ValueError):
        get_n_from_pauli_basis_representation(np.zeros(15))

    pass


def test_decomposition_element():
    # Test initialization with np.ndarray
    operator = np.array([[1, 0], [0, 1]])
    probability = 0.5
    element = DecompositionElement(operator, probability)
    assert np.array_equal(element.operator, operator)
    assert element.probability == probability

    with pytest.raises(ValueError):
        DecompositionElement(np.array([1, 0]), -0.1)

    with pytest.raises(ValueError):
        DecompositionElement(np.array([1, 0]), 0)


if __name__ == "__main__":
    test_pauli_init()
    test_phase()
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
