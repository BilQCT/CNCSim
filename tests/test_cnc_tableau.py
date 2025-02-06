import copy
import numpy as np
import pytest

from src.neater_cnc_tableau import (
    symplectic_inner_product,
    beta,
    symplectic_matrix,
    CncSimulator,
)


def test_symplectic_inner_product():
    xx = np.array([1, 1, 0, 0])
    yy = np.array([1, 1, 1, 1])
    assert symplectic_inner_product(xx, xx) == 0
    assert symplectic_inner_product(yy, yy) == 0
    assert symplectic_inner_product(xx, yy) == 0

    xy = np.array([1, 1, 0, 1])
    assert symplectic_inner_product(xy, xx) == 1
    assert symplectic_inner_product(xy, yy) == 1

    xyz = np.array([1, 1, 0, 0, 1, 1])
    yyy = np.array([1, 1, 1, 1, 1, 1])
    assert symplectic_inner_product(xyz, yyy) == 0


def test_beta():
    xx = np.array([1, 1, 0, 0])
    yy = np.array([1, 1, 1, 1])
    assert beta(xx, xx) == 0
    assert beta(yy, yy) == 0
    assert beta(xx, yy) == 1

    xy = np.array([1, 1, 0, 1])
    with pytest.raises(AssertionError):
        beta(xy, xx)
    yx = np.array([1, 1, 1, 0])
    assert beta(xy, yx) == 0

    xyz = np.array([1, 1, 0, 0, 1, 1])
    yyy = np.array([1, 1, 1, 1, 1, 1])
    assert beta(xyz, yyy) == 0


def test_symplectic_matrix():
    sm_1 = symplectic_matrix(1)
    sm_2 = symplectic_matrix(2)
    sm_3 = symplectic_matrix(3)

    expected_sm1 = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    expected_sm2 = np.array(
        [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]],
        dtype=np.uint8,
    )
    expected_sm3 = np.array(
        [
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ],
        dtype=np.uint8,
    )

    assert np.array_equal(sm_1, expected_sm1)
    assert np.array_equal(sm_2, expected_sm2)
    assert np.array_equal(sm_3, expected_sm3)


def test_initial_tableau():
    # 1-qubit
    tableau_1_0 = CncSimulator.initial_cnc_tableau(n=1, m=0)
    expexted_1_0 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.uint8)
    assert np.array_equal(tableau_1_0, expexted_1_0)

    tableau_1_1 = CncSimulator.initial_cnc_tableau(n=1, m=1)
    expexted_1_1 = np.array([[1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=np.uint8)
    assert np.array_equal(tableau_1_1, expexted_1_1)

    # 2-qubit
    tableau_2_0 = CncSimulator.initial_cnc_tableau(n=2, m=0)
    expected_2_0 = np.array(
        [
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(tableau_2_0, expected_2_0)
    tableau_2_1 = CncSimulator.initial_cnc_tableau(n=2, m=1)
    expected_2_1 = np.array(
        [
            [0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 1, 0],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(tableau_2_1, expected_2_1)
    tableau_2_2 = CncSimulator.initial_cnc_tableau(n=2, m=2)
    expected_2_2 = np.array(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 1, 1, 0],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(tableau_2_2, expected_2_2)

    # 3-qubit
    tableau_3_0 = CncSimulator.initial_cnc_tableau(n=3, m=0)
    expected_3_0 = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(tableau_3_0, expected_3_0)
    tableau_3_1 = CncSimulator.initial_cnc_tableau(n=3, m=1)
    expected_3_1 = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(tableau_3_1, expected_3_1)
    tableau_3_2 = CncSimulator.initial_cnc_tableau(n=3, m=2)
    expected_3_2 = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 1, 0],
            [0, 1, 1, 0, 1, 1, 0],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(tableau_3_2, expected_3_2)
    tableau_3_3 = CncSimulator.initial_cnc_tableau(n=3, m=3)
    expected_3_3 = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 1, 0, 0, 0],
            [1, 0, 0, 1, 1, 0, 0],
            [1, 1, 0, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 0],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(tableau_3_3, expected_3_3)


def test_cnc_init():
    # 1-qubit
    cnc_1_0 = CncSimulator(n=1, m=0)
    assert cnc_1_0.n == 1
    assert cnc_1_0.m == 0
    assert cnc_1_0.isotropic_dim == 1
    expected_sm1 = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    assert np.array_equal(cnc_1_0._symplectic_matrix, expected_sm1)
    expexted_tableau_1_0 = np.array(
        [[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=np.uint8)
    assert np.array_equal(cnc_1_0.tableau, expexted_tableau_1_0)
    expexted_tableau_wo_phase_1_0 = np.array(
        [[0, 1], [1, 0], [0, 0]], dtype=np.uint8
        )
    assert np.array_equal(
        cnc_1_0._tableau_without_phase, expexted_tableau_wo_phase_1_0
        )
    x_cols_1_0 = np.array([[0], [1], [0]], dtype=np.uint8)
    assert np.array_equal(cnc_1_0._x_cols, x_cols_1_0)
    z_cols_1_0 = np.array([[1], [0], [0]], dtype=np.uint8)
    assert np.array_equal(cnc_1_0._z_cols, z_cols_1_0)
    phase_col_1_0 = np.array([0, 0, 0], dtype=np.uint8)
    assert np.array_equal(cnc_1_0._phase_col, phase_col_1_0)
    destabilizer_1_0 = np.array([[0, 1]], dtype=np.uint8)
    assert np.array_equal(cnc_1_0._destabilizer_rows, destabilizer_1_0)
    stabilizer_1_0 = np.array([[1, 0]], dtype=np.uint8)
    assert np.array_equal(cnc_1_0._stabilizer_rows, stabilizer_1_0)
    jw_elements_rows_1_0 = np.array([[0, 0]], dtype=np.uint8)
    assert np.array_equal(cnc_1_0._jw_elements_rows, jw_elements_rows_1_0)

    # 2-qubit
    cnc_2_2 = CncSimulator(n=2, m=2)
    assert cnc_2_2.n == 2
    assert cnc_2_2.m == 2
    assert cnc_2_2.isotropic_dim == 0
    expected_sm2 = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(cnc_2_2._symplectic_matrix, expected_sm2)
    expexted_tableau_2_2 = np.array(
        [
            [1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 1, 1, 0],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(cnc_2_2.tableau, expexted_tableau_2_2)
    expexted_tableau_wo_phase_2_2 = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(
        cnc_2_2._tableau_without_phase, expexted_tableau_wo_phase_2_2
        )
    x_cols_2_2 = np.array(
        [
            [1, 0],
            [1, 1],
            [0, 0],
            [1, 0],
            [1, 1],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(cnc_2_2._x_cols, x_cols_2_2)
    z_cols_2_2 = np.array(
        [
            [0, 0],
            [1, 0],
            [1, 0],
            [1, 1],
            [1, 1],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(cnc_2_2._z_cols, z_cols_2_2)
    phase_col_2_2 = np.zeros(5, dtype=np.uint8)
    assert np.array_equal(cnc_2_2._phase_col, phase_col_2_2)
    assert cnc_2_2._destabilizer_rows.size == 0
    assert cnc_2_2._stabilizer_rows.size == 0
    jw_elements_rows_2_2 = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [0, 0, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(cnc_2_2._jw_elements_rows, jw_elements_rows_2_2)

    # 3-qubit
    cnc_3_1 = CncSimulator(n=3, m=1)
    assert cnc_3_1.n == 3
    assert cnc_3_1.m == 1
    assert cnc_3_1.isotropic_dim == 2
    expected_sm3 = np.array(
        [
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(cnc_3_1._symplectic_matrix, expected_sm3)
    expexted_tableau_3_1 = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1, 0],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(cnc_3_1.tableau, expexted_tableau_3_1)
    expexted_tableau_wo_phase_3_1 = np.array(
        [
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 1],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(
        cnc_3_1._tableau_without_phase, expexted_tableau_wo_phase_3_1
        )
    x_cols_3_1 = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 1],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(cnc_3_1._x_cols, x_cols_3_1)
    z_cols_3_1 = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 1],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(cnc_3_1._z_cols, z_cols_3_1)
    phase_col_3_1 = np.zeros(7, dtype=np.uint8)
    assert np.array_equal(cnc_3_1._phase_col, phase_col_3_1)
    destabilizer_3_1 = np.array(
        [
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(cnc_3_1._destabilizer_rows, destabilizer_3_1)
    stabilizer_3_1 = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(cnc_3_1._stabilizer_rows, stabilizer_3_1)
    jw_elements_rows_3_1 = np.array(
        [
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 0, 1],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(cnc_3_1._jw_elements_rows, jw_elements_rows_3_1)


def test_cliffords():
    cnc_2_1 = CncSimulator(n=2, m=1)
    cnc_2_1.apply_hadamard(0)
    expected = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 1, 0],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(cnc_2_1.tableau, expected)

    cnc_2_1.apply_hadamard(1)
    expected = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(cnc_2_1.tableau, expected)

    cnc_2_1.apply_phase(0)
    expected = np.array(
        [
            [1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 1, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(cnc_2_1.tableau, expected)

    cnc_2_1.apply_phase(1)
    expected = np.array(
        [
            [1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(cnc_2_1.tableau, expected)

    cnc_2_1.apply_cnot(control_qubit=0, target_qubit=1)
    expected = np.array(
        [
            [1, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(cnc_2_1.tableau, expected)

    cnc_2_1.apply_cnot(control_qubit=1, target_qubit=0)
    expected = np.array(
        [
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 0, 0],
            [1, 1, 1, 0, 0],
            [1, 1, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(cnc_2_1.tableau, expected)

    with pytest.raises(AssertionError):
        cnc_2_1.apply_cnot(control_qubit=0, target_qubit=0)

    # Check whether tableau variables are updated correctly
    x_cols_2_1 = np.array(
        [
            [0, 1],
            [0, 0],
            [0, 0],
            [1, 1],
            [1, 1],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(cnc_2_1._x_cols, x_cols_2_1)
    z_cols_2_1 = np.array(
        [
            [1, 1],
            [1, 1],
            [1, 0],
            [1, 0],
            [0, 0],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(cnc_2_1._z_cols, z_cols_2_1)
    destabilizer_2_1 = np.array(
        [
            [0, 1, 1, 1],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(cnc_2_1._destabilizer_rows, destabilizer_2_1)
    stabilizer_2_1 = np.array(
        [
            [0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(cnc_2_1._stabilizer_rows, stabilizer_2_1)
    jw_elements_rows_2_1 = np.array(
        [
            [0, 0, 1, 0],
            [1, 1, 1, 0],
            [1, 1, 0, 0],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(cnc_2_1._jw_elements_rows, jw_elements_rows_2_1)

    # TODO: add 3-qubit tests as well


def test_measure_case_1():
    cnc = CncSimulator(n=4, m=2)
    expected = np.array([
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 0],
    ],
        dtype=np.uint8
    )
    assert np.array_equal(cnc.tableau, expected)

    # For Case 1
    # Choose XIII as the measurement basis
    meas_basis = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
    outcome = cnc.measure(meas_basis)
    assert outcome == 0
    # Tableau should not be changed
    assert np.array_equal(cnc.tableau, expected)

    # Choose XXII as the measurement basis
    meas_basis = np.array([1, 1, 0, 0, 0, 0, 0, 0], dtype=np.uint8)
    outcome = cnc.measure(meas_basis)
    assert outcome == 0
    # Tableau should not be changed
    assert np.array_equal(cnc.tableau, expected)


def test_measure_case_2():
    cnc = CncSimulator(n=4, m=2)
    expected = np.array([
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 0],
    ],
        dtype=np.uint8
    )
    assert np.array_equal(cnc.tableau, expected)

    # Choose XXYY as the measurement basis
    meas_basis = np.array([1, 1, 1, 1, 0, 0, 1, 1], dtype=np.uint8)
    expected_2 = np.array([
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 1],
        [0, 0, 1, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 1, 1, 0],
    ],
        dtype=np.uint8
    )

    count = 0
    expected_equal_found = False
    expected_2_equal_found = False
    while count < 1024:
        outcome = cnc.measure(meas_basis)
        assert outcome == 0

        is_equal_to_expected = np.array_equal(cnc.tableau, expected)
        is_equal_to_expected_2 = np.array_equal(cnc.tableau, expected_2)

        assert is_equal_to_expected or is_equal_to_expected_2

        if is_equal_to_expected:
            expected_equal_found = True
        if is_equal_to_expected_2:
            expected_2_equal_found = True

        # Break when both expected outcomes are found and
        # the last outcome is equal to expected_2 in order to
        # know the table certainly after the loop.
        if expected_equal_found and expected_2_equal_found and (
            is_equal_to_expected_2
        ):
            break

        count += 1

    assert expected_equal_found and expected_2_equal_found
    assert np.array_equal(cnc.tableau, expected_2)

    # Choose XIYX as the measurement basis
    meas_basis = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.uint8)

    expected_3 = np.array([
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 1],
    ],
        dtype=np.uint8
    )

    expected_2_equal_found = False
    expected_3_equal_found = False
    while count < 1024:
        outcome = cnc.measure(meas_basis)
        # Since we fixed the tableau to be as expected_2,
        # the outcome should be 1.
        assert outcome == 1

        is_equal_to_expected_2 = np.array_equal(cnc.tableau, expected_2)
        is_equal_to_expected_3 = np.array_equal(cnc.tableau, expected_3)

        assert is_equal_to_expected_2 or is_equal_to_expected_3

        if is_equal_to_expected_2:
            expected_2_equal_found = True
        if is_equal_to_expected_3:
            expected_3_equal_found = True

        # Break when both expected outcomes are found and
        # the last outcome is equal to expected_3 in order to
        # know the table certainly after the loop.
        if is_equal_to_expected_3 and expected_2_equal_found and (
            expected_3_equal_found
        ):
            break

        count += 1

    assert expected_2_equal_found and expected_3_equal_found
    assert np.array_equal(cnc.tableau, expected_3)

    # Check other tableau variables
    expected_x_cols = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(cnc._x_cols, expected_x_cols)
    expected_z_cols = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(cnc._z_cols, expected_z_cols)
    expected_destabilizer = np.array(
        [
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(cnc._destabilizer_rows, expected_destabilizer)
    expected_stabilizer = np.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    assert np.array_equal(cnc._stabilizer_rows, expected_stabilizer)
    expected_jw_elements = np.array(
        [
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1, 1],
            [0, 0, 1, 1, 0, 0, 1, 1],
        ],
        dtype=np.uint8
    )
    assert np.array_equal(cnc._jw_elements_rows, expected_jw_elements)


def test_measure_case_3():
    # Stabiliers = <XIII, IXII>
    # Destabilizers = <ZIII, IZII>
    # JW elements = IIXI, -IIYX, IIZI, IIZY, -IIYY
    initial_tableau = np.array([
        [0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0, 0, 1, 1, 1],
    ],
        dtype=np.uint8
    )

    cnc = CncSimulator.from_tableau(n=4, m=2, tableau=initial_tableau)
    # Choose IIIX as the measurement basis
    meas_basis = np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=np.uint8)
    assert cnc.m == 2
    assert cnc.isotropic_dim == 2

    outcome_0_found = False
    outcome_1_found = False
    count = 0
    while count < 1024:
        cnc_copy = copy.deepcopy(cnc)
        outcome = cnc_copy.measure(meas_basis)
        assert outcome in [0, 1]

        if outcome == 0:
            outcome_0_found = True
        if outcome == 1:
            outcome_1_found = True

        assert cnc_copy.m == 1
        assert cnc_copy.isotropic_dim == 3

        # Since we do not care about the choice of the anticommuting JW
        # element that will be put in the destabilizer, we check for
        # two possible choices. Either we add IIYZ or IIYY to
        # the destabilizer.
        expected_destabilizer = np.array(
            [
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 1],
            ],
            dtype=np.uint8
        )
        expected_destabilizer_2 = np.array(
            [
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 1, 0, 0, 1, 1],
            ],
            dtype=np.uint8
        )
        assert np.array_equal(
            cnc_copy._destabilizer_rows, expected_destabilizer
            ) or np.array_equal(
            cnc_copy._destabilizer_rows, expected_destabilizer_2
            )

        # b_bar = IIIX so we should add IIIX to the stabilizer
        # with the phase as the outcome of the measurement.
        expected_stabilizer = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0]
            ],
            dtype=np.uint8
        )
        assert np.array_equal(cnc_copy._stabilizer_rows,
                              expected_stabilizer)
        expected_stabilizer_phases = np.array([0, 0, outcome], dtype=np.uint8)
        assert np.array_equal(
            cnc_copy._phase_col[
                cnc_copy.isotropic_dim: 2*cnc_copy.isotropic_dim],
            expected_stabilizer_phases
            )

        # Since the order of the JW elements are not important,
        # we check for all possible permutations using set.
        # Also we include the phases of the JW elements.
        expected_jw_elements_with_phases = set(
            [
                (0, 0, 1, 1, 0, 0, 0, 0, outcome),
                (0, 0, 1, 0, 0, 0, 1, 0, (outcome + 1) % 2),
                (0, 0, 0, 1, 0, 0, 1, 0, outcome),
            ]
        )

        # Convert the JW elements with their phases to set of tuples
        result_jw = set()
        for i in range(3):
            result_jw.add(tuple(cnc_copy.tableau[
                2*cnc_copy.isotropic_dim + i]))

        assert result_jw == expected_jw_elements_with_phases

        if outcome_0_found and outcome_1_found and (outcome == 1):
            break

        count += 1

    assert outcome_0_found and outcome_1_found
    # Notice that we did not modify the original tableau
    # in the loop, so the tableau should be the same as the initial_tableau.
    assert np.array_equal(initial_tableau, cnc.tableau)

    # Choose XIIX as the measurement basis
    # Notice now, b is not equal to b_bar = IIIX
    meas_basis = np.array([1, 0, 0, 1, 0, 0, 0, 0], dtype=np.uint8)
    assert cnc.m == 2
    assert cnc.isotropic_dim == 2

    outcome_0_found = False
    outcome_1_found = False
    count = 0

    while count < 1024:
        cnc_copy = copy.deepcopy(cnc)
        outcome = cnc_copy.measure(meas_basis)
        assert outcome in [0, 1]

        if outcome == 0:
            outcome_0_found = True
        if outcome == 1:
            outcome_1_found = True

        assert cnc_copy.m == 1
        assert cnc_copy.isotropic_dim == 3

        # Destabilizer should be the same as the previous case
        # sinbce b_bar is the same.
        expected_destabilizer = np.array(
            [
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 0, 0, 0, 1, 1],
            ],
            dtype=np.uint8
        )
        expected_destabilizer_2 = np.array(
            [
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
                [0, 0, 1, 1, 0, 0, 1, 1],
            ],
            dtype=np.uint8
        )
        assert np.array_equal(
            cnc_copy._destabilizer_rows, expected_destabilizer
            ) or np.array_equal(
            cnc_copy._destabilizer_rows, expected_destabilizer_2
            )

        # Stabilizer should be the same as the previous case
        # sinbce b_bar is the same.
        expected_stabilizer = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 0]
            ],
            dtype=np.uint8
        )
        assert np.array_equal(cnc_copy._stabilizer_rows,
                              expected_stabilizer)
        expected_stabilizer_phases = np.array([0, 0, outcome], dtype=np.uint8)
        assert np.array_equal(
            cnc_copy._phase_col[
                cnc_copy.isotropic_dim: 2*cnc_copy.isotropic_dim],
            expected_stabilizer_phases
            )

        # Also the JW elements should be the same as the previous case
        expected_jw_elements_with_phases = set(
            [
                (0, 0, 1, 1, 0, 0, 0, 0, outcome),
                (0, 0, 1, 0, 0, 0, 1, 0, (outcome + 1) % 2),
                (0, 0, 0, 1, 0, 0, 1, 0, outcome),
            ]
        )

        # Convert the JW elements with their phases to set of tuples
        result_jw = set()
        for i in range(3):
            result_jw.add(tuple(cnc_copy.tableau[
                2*cnc_copy.isotropic_dim + i]))

        assert result_jw == expected_jw_elements_with_phases

        if outcome_0_found and outcome_1_found and (outcome == 1):
            break

        count += 1

    assert outcome_0_found and outcome_1_found
    # Notice that we did not modify the original tableau
    # in the loop, so the tableau should be the same as the initial_tableau.
    assert np.array_equal(initial_tableau, cnc.tableau)

    # In order to check the t-1 additonal elements for both destabilizer
    # we need to consider a different tableau.
    cnc = CncSimulator(n=4, m=3)

    # Choose XIXI as the measurement basis
    # Again, b is not equal to b_bar = IIXI
    meas_basis = np.array([1, 0, 1, 0, 0, 0, 0, 0], dtype=np.uint8)
    assert cnc.m == 3
    assert cnc.isotropic_dim == 1

    outcome_0_found = False
    outcome_1_found = False
    count = 0
    while count < 1024:
        cnc_copy = copy.deepcopy(cnc)
        outcome = cnc_copy.measure(meas_basis)
        assert outcome in [0, 1]

        if outcome == 0:
            outcome_0_found = True
        if outcome == 1:
            outcome_1_found = True

        assert cnc_copy.m == 1
        assert cnc_copy.isotropic_dim == 3

        # Since we do not care about the choice of the anticommuting JW
        # element that will be put in the destabilizer, we check for
        # all possible choices.
        expected_destabilizers = [
            set(
                [
                    (0, 0, 0, 0, 1, 0, 0, 0),
                    (0, 1, 1, 1, 0, 1, 1, 0),  # IYYX
                    (0, 0, 1, 0, 0, 0, 0, 1)   # IIXZ
                ]
            ),
            set(
                [
                    (0, 0, 0, 0, 1, 0, 0, 0),
                    (0, 1, 1, 1, 0, 1, 1, 0),  # IYYX
                    (0, 0, 1, 1, 0, 0, 0, 1)   # IIXY
                ]
            ),
            set(
                [
                    (0, 0, 0, 0, 1, 0, 0, 0),
                    (0, 1, 1, 1, 0, 1, 1, 0),  # IYYX
                    (0, 0, 0, 0, 0, 0, 0, 1)   # IIIZ
                ]
            ),
            set(
                [
                    (0, 0, 0, 0, 1, 0, 0, 0),
                    (0, 1, 0, 0, 0, 1, 1, 0),  # IYZI
                    (0, 0, 0, 1, 0, 0, 0, 1)   # IIIY
                ]
            ),
            set(
                [
                    (0, 0, 0, 0, 1, 0, 0, 0),
                    (0, 1, 0, 0, 0, 1, 1, 0),  # IYZI
                    (0, 0, 0, 0, 0, 0, 0, 1)   # IIIZ
                ]
            ),
            set(
                [
                    (0, 0, 0, 0, 1, 0, 0, 0),
                    (0, 1, 0, 0, 0, 1, 1, 0),  # IYZI
                    (0, 0, 0, 1, 0, 0, 0, 0)   # IIIX
                ]
            ),
            set(
                [
                    (0, 0, 0, 0, 1, 0, 0, 0),
                    (0, 1, 1, 0, 0, 1, 1, 1),  # IYYZ
                    (0, 0, 1, 1, 0, 0, 0, 0)   # IIXX
                ]
            ),
            set(
                [
                    (0, 0, 0, 0, 1, 0, 0, 0),
                    (0, 1, 1, 0, 0, 1, 1, 1),  # IYYZ
                    (0, 0, 0, 0, 0, 0, 0, 1)   # IIIZ
                ]
            ),
            set(
                [
                    (0, 0, 0, 0, 1, 0, 0, 0),
                    (0, 1, 1, 0, 0, 1, 1, 1),  # IYYZ
                    (0, 0, 1, 1, 0, 0, 0, 1)   # IIXY
                ]
            ),
            set(
                [
                    (0, 0, 0, 0, 1, 0, 0, 0),
                    (0, 1, 1, 1, 0, 1, 1, 1),  # IYYY
                    (0, 0, 1, 1, 0, 0, 0, 0)   # IIXX
                ]
            ),
            set(
                [
                    (0, 0, 0, 0, 1, 0, 0, 0),
                    (0, 1, 1, 1, 0, 1, 1, 1),  # IYYY
                    (0, 0, 0, 1, 0, 0, 0, 1)   # IIIY
                ]
            ),
            set(
                [
                    (0, 0, 0, 0, 1, 0, 0, 0),
                    (0, 1, 1, 1, 0, 1, 1, 1),  # IYYY
                    (0, 0, 1, 0, 0, 0, 0, 1)   # IIXZ
                ]
            ),
        ]

        assert len(cnc_copy._destabilizer_rows) == 3
        result_destabilizer = set(
            tuple(row) for row in cnc_copy._destabilizer_rows
        )
        assert any(
            result_destabilizer == expected
            for expected in expected_destabilizers
        )
        # b_bar = IIIX so we should add IIIX to the stabilizer
        # with the phase as the outcome of the measurement.
        expected_stabilizers = [
            set(
                [
                    (1, 0, 0, 0, 0, 0, 0, 0, 0),
                    (0, 0, 1, 0, 0, 0, 0, 0, outcome),  # b_bar = IIXI
                    (0, 0, 1, 1, 0, 0, 0, 0, 0),  # IIXX
                ]
            ),
            set(
                [
                    (1, 0, 0, 0, 0, 0, 0, 0, 0),
                    (0, 0, 1, 0, 0, 0, 0, 0, outcome),  # b_bar = IIXI
                    (0, 0, 1, 1, 0, 0, 0, 0, 1),  # IIXX
                ]
            ),
            set(
                [
                    (1, 0, 0, 0, 0, 0, 0, 0, 0),
                    (0, 0, 1, 0, 0, 0, 0, 0, outcome),  # b_bar = IIXI
                    (0, 0, 0, 1, 0, 0, 0, 1, 0),  # IIIY
                ]
            ),
            set(
                [
                    (1, 0, 0, 0, 0, 0, 0, 0, 0),
                    (0, 0, 1, 0, 0, 0, 0, 0, outcome),  # b_bar = IIXI
                    (0, 0, 0, 1, 0, 0, 0, 1, 1),  # IIIY
                ]
            ),
            set(
                [
                    (1, 0, 0, 0, 0, 0, 0, 0, 0),
                    (0, 0, 1, 0, 0, 0, 0, 0, outcome),  # b_bar = IIXI
                    (0, 0, 0, 0, 0, 0, 0, 1, 0),  # IIIZ
                ]
            ),
            set(
                [
                    (1, 0, 0, 0, 0, 0, 0, 0, 0),
                    (0, 0, 1, 0, 0, 0, 0, 0, outcome),  # b_bar = IIXI
                    (0, 0, 0, 0, 0, 0, 0, 1, 1),  # IIIZ
                ]
            ),
            set(
                [
                    (1, 0, 0, 0, 0, 0, 0, 0, 0),
                    (0, 0, 1, 0, 0, 0, 0, 0, outcome),  # b_bar = IIXI
                    (0, 0, 1, 0, 0, 0, 0, 1, 0),  # IIXZ
                ]
            ),
            set(
                [
                    (1, 0, 0, 0, 0, 0, 0, 0, 0),
                    (0, 0, 1, 0, 0, 0, 0, 0, outcome),  # b_bar = IIXI
                    (0, 0, 1, 0, 0, 0, 0, 1, 1),  # IIXZ
                ]
            ),
            set(
                [
                    (1, 0, 0, 0, 0, 0, 0, 0, 0),
                    (0, 0, 1, 0, 0, 0, 0, 0, outcome),  # b_bar = IIXI
                    (0, 0, 1, 1, 0, 0, 0, 1, 0),  # IIXY
                ]
            ),
            set(
                [
                    (1, 0, 0, 0, 0, 0, 0, 0, 0),
                    (0, 0, 1, 0, 0, 0, 0, 0, outcome),  # b_bar = IIXI
                    (0, 0, 1, 1, 0, 0, 0, 1, 1),  # IIXY
                ]
            ),
            set(
                [
                    (1, 0, 0, 0, 0, 0, 0, 0, 0),
                    (0, 0, 1, 0, 0, 0, 0, 0, outcome),  # b_bar = IIXI
                    (0, 0, 0, 1, 0, 0, 0, 0, 0),  # IIIX
                ]
            ),
            set(
                [
                    (1, 0, 0, 0, 0, 0, 0, 0, 0),
                    (0, 0, 1, 0, 0, 0, 0, 0, outcome),  # b_bar = IIXI
                    (0, 0, 0, 1, 0, 0, 0, 0, 1),  # IIIX
                ]
            ),
        ]

        assert len(cnc_copy._stabilizer_rows) == 3
        result_stabilizers_with_phase = set(
            tuple(row) for row in cnc_copy.tableau[3: 6]
        )
        assert any(
            result_stabilizers_with_phase == expected
            for expected in expected_stabilizers
        )

        # Since the order of the JW elements are not important,
        # we check for all possible permutations using set.
        # Also we include the phases of the JW elements.
        expected_jw_elements_with_phases = set(
            [
                (0, 1, 1, 0, 0, 0, 0, 0, outcome),
                (0, 1, 0, 0, 0, 1, 0, 0, outcome),
                (0, 0, 1, 0, 0, 1, 0, 0, outcome),
            ]
        )

        # Convert the JW elements with their phases to set of tuples
        result_jw = set()
        for i in range(3):
            result_jw.add(tuple(cnc_copy.tableau[
                2*cnc_copy.isotropic_dim + i]))

        assert result_jw == expected_jw_elements_with_phases

        if outcome_0_found and outcome_1_found and (outcome == 1):
            break

        count += 1

    assert outcome_0_found and outcome_1_found


def test_case_4():
    cnc = CncSimulator(n=4, m=2)
    # Choose ZIII as the measurement basis
    # So we know that the anticommuting stabilizer basis is XIII
    meas_basis = np.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=np.uint8)
    outcome_0_found = False
    outcome_1_found = False
    count = 0

    while count < 1024:
        cnc_copy = copy.deepcopy(cnc)
        outcome = cnc_copy.measure(meas_basis)
        assert outcome in [0, 1]

        if outcome == 0:
            outcome_0_found = True
        elif outcome == 1:
            outcome_1_found = True

        assert cnc_copy.m == 2

        expected_destabilizer = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
            ],
            dtype=np.uint8
        )

        assert np.array_equal(
            cnc_copy._destabilizer_rows, expected_destabilizer
            )

        # b_bar = IIIX so we should add IIIX to the stabilizer
        # with the phase as the outcome of the measurement.
        expected_stabilizer = np.array(
            [
                [0, 0, 0, 0, 1, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint8
        )
        assert np.array_equal(cnc_copy._stabilizer_rows,
                              expected_stabilizer)
        expected_stabilizer_phases = np.array([outcome, 0], dtype=np.uint8)
        assert np.array_equal(
            cnc_copy._phase_col[
                cnc_copy.isotropic_dim: 2*cnc_copy.isotropic_dim],
            expected_stabilizer_phases
            )

        assert np.array_equal(
            cnc_copy._jw_elements_rows, cnc._jw_elements_rows
            )

        if outcome_0_found and outcome_1_found:
            break

        count += 1

    cnc = CncSimulator(n=4, m=2)
    # Choose ZIIZ as the measurement basis
    # So we know that the anticommuting stabilizer basis is XIII
    meas_basis = np.array([0, 0, 0, 0, 1, 0, 0, 1], dtype=np.uint8)
    outcome_0_found = False
    outcome_1_found = False
    count = 0

    while count < 1024:
        cnc_copy = copy.deepcopy(cnc)
        outcome = cnc_copy.measure(meas_basis)
        assert outcome in [0, 1]

        if outcome == 0:
            outcome_0_found = True
        elif outcome == 1:
            outcome_1_found = True

        assert cnc_copy.m == 2

        expected_destabilizer = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 1, 0, 0],
            ],
            dtype=np.uint8
        )

        assert np.array_equal(
            cnc_copy._destabilizer_rows, expected_destabilizer
            )

        # b_bar = IIIX so we should add IIIX to the stabilizer
        # with the phase as the outcome of the measurement.
        expected_stabilizer = np.array(
            [
                [0, 0, 0, 0, 1, 0, 0, 1],
                [0, 1, 0, 0, 0, 0, 0, 0],
            ],
            dtype=np.uint8
        )
        assert np.array_equal(cnc_copy._stabilizer_rows,
                              expected_stabilizer)
        expected_stabilizer_phases = np.array([outcome, 0], dtype=np.uint8)
        assert np.array_equal(
            cnc_copy._phase_col[
                cnc_copy.isotropic_dim: 2*cnc_copy.isotropic_dim],
            expected_stabilizer_phases
            )

        expected_jw_elements = np.array(
            [
                [0, 0, 1, 0, 0, 0, 0, 0],
                [1, 0, 1, 1, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1, 1],
                [1, 0, 1, 1, 0, 0, 1, 1],
            ],
            dtype=np.uint8
        )

        assert np.array_equal(
            cnc_copy._jw_elements_rows, expected_jw_elements
            )

        if outcome_0_found and outcome_1_found:
            break

        count += 1
