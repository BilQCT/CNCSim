from typing import Optional

import numpy as np
from pulp import (GLPK, LpMinimize, LpProblem, LpStatusOptimal, LpVariable,
                  lpDot, lpSum)

from src.cnc import CNC
from src.utils import (DecompositionElement,
                       get_n_from_pauli_basis_representation,
                       load_all_maximal_cncs_matrix)


def find_vertex_decomposition(
    state: np.ndarray, vertices_matrix: np.ndarray
) -> tuple[bool, list[DecompositionElement]]:
    """Finds the vertex decomposition of a given state with respect to the given vertices matrix. The vertices matrix
    is a matrix where each column is a vertex of the polytope. The decomposition is found by modeling the problem as a
    linear program and solving it.

    Args:
        state (np.ndarray): State to be decomposed.
        vertices_matrix (np.ndarray): Vertices matrix of the polytope.

    Returns:
        tuple[bool, list[DecompositionElement]]:
            A tuple where the first element is a boolean indicating whether the
            decomposition is convex or not, and the second element is a list of :py:attr:`~utils.DecompositionElement`
            that stores the vertices of the decomposition and their probabilities.

    """
    d, num_vertices = vertices_matrix.shape

    model = LpProblem("Vertex_Decomposition", LpMinimize)

    x = LpVariable.dicts("x", range(num_vertices), lowBound=0)

    # Define constraints
    for i in range(d):
        model += (
            lpDot(vertices_matrix[i], [x[j] for j in range(num_vertices)]) == state[i],
            f"Equality_contraint_{i}",
        )

    # Sum of x[j] should be 1
    model += lpSum([x[j] for j in range(num_vertices)]) == 1

    # Any feasible solution is an optimal solution
    model += 0

    model.solve(GLPK(msg=False))

    is_convex = True
    if model.status != LpStatusOptimal:
        is_convex = False

    non_zero_column_indices = [j for j in range(num_vertices) if x[j].varValue > 0]

    distribution = [
        DecompositionElement(vertices_matrix[:, j], x[j].varValue)
        for j in non_zero_column_indices
    ]

    return is_convex, distribution


def find_cnc_vertex_decomposition(
    state: np.ndarray, cnc_vertices_matrix: Optional[np.ndarray] = None
) -> tuple[bool, list[DecompositionElement]]:
    """It is CNC specific version of :func:`find_vertex_decomposition` for CNC vertices. It finds the CNC vertex decomposition
    of a given state. If the vertices matrix is not provided, it loads the maximal CNCs matrix from the disk. The decomposition
    is found by modeling the problem as a linear program and solving it.

    Args:
        state (np.ndarray): State to be decomposed.
        cnc_vertices_matrix (Optional[np.ndarray]): Vertices matrix of the CNC polytope. If not provided, it loads the maximal
            CNCs matrix from the disk. Defaults to None.

    Returns:
        tuple[bool, list[DecompositionElement]]:
            A tuple where the first element is a boolean indicating whether the
            decomposition is convex or not, and the second element is a list of :py:attr:`~utils.DecompositionElement`
            that stores the vertices of the decomposition and their probabilities. The operators in the decomposition are
            CNC operators.
    """

    n = get_n_from_pauli_basis_representation(state)

    if cnc_vertices_matrix is None:
        cnc_vertices_matrix = load_all_maximal_cncs_matrix(n)

    is_convex, distribution = find_vertex_decomposition(state, cnc_vertices_matrix)

    cnc_distribution = [
        DecompositionElement(
            operator=CNC.from_pauli_basis_representation(
                decomposition_element.operator
            ),
            probability=decomposition_element.probability,
        )
        for decomposition_element in distribution
    ]

    return is_convex, cnc_distribution
