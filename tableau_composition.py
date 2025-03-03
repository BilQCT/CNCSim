import numpy as np

def left_compose(tableau1, tableau2, m1, m2):
    """
    Compose two CNC tableaus via left composition.

    This function composes two tableaus that follow the canonical CNC tableau
    structure. For a system with n qubits, each tableau is a binary matrix of size
    (2*n + 1) x (2*n + 1) with the following structure:
    
      - Columns:
          * First n columns: X-part.
          * Next n columns: Z-part.
          * Last column: Phase bits.
      - Rows:
          * Destabilizer rows: first (n - m) rows.
          * Stabilizer rows: next (n - m) rows.
          * JW elements: last (2*m + 1) rows.
    
    In left composition, the left tableau (tableau1) must be a pure stabilizer tableau,
    which means its type parameter m1 must be 0. The right tableau (tableau2) can have
    nonzero m2 (i.e. nontrivial JW elements). The function performs the following steps:
    
      1. Compute the number of qubits (n1 and n2) from the shape of each tableau.
      2. Split each tableau into its X and Z blocks (ignoring the phase column) and pad
         with zeros so that the X and Z parts align when the tableaus are combined.
      3. Recombine the padded X and Z parts with the phase column.
      4. Stack the rows in the following order:
         - Destabilizers: first n1 rows from tableau1, then the first (n2 - m2) rows from tableau2.
         - Stabilizers: next n1 rows from tableau1, then the next (n2 - m2) rows from tableau2.
         - JW elements: the remaining 2*m2 rows from tableau2.
    
    Parameters:
        tableau1 (np.ndarray): Left tableau of shape (r1, 2*n1 + 1) with structure:
                               [X-part | Z-part | phase].
        tableau2 (np.ndarray): Right tableau of shape (r2, 2*n2 + 1) with structure:
                               [X-part | Z-part | phase].
        m1 (int): Type parameter for tableau1. Must be 0 (i.e. tableau1 is a stabilizer).
        m2 (int): Type parameter for tableau2, indicating the number of JW rows.

    Returns:
        np.ndarray: A composed tableau with shape ((r1 + r2) x ((2*n1 + 1) + (2*n2 + 1) - 1)).
                    The resulting tableau has merged and padded X and Z parts, a single phase column,
                    and rows arranged in the order: destabilizers, stabilizers, and JW elements.
    
    Raises:
        ValueError: If m1 is not 0.
    """
    if m1 != 0:
        raise ValueError("Composition is not valid. Left tableau must be a stabilizer.")

    # Determine number of qubits from each tableau.
    n1 = int((tableau1.shape[1] - 1) / 2)
    n2 = int((tableau2.shape[1] - 1) / 2)

    # Get dimensions of the input tableaus.
    rows1 = tableau1.shape[0]
    cols1 = tableau1.shape[1]
    rows2 = tableau2.shape[0]
    cols2 = tableau2.shape[1]

    # Initialize the new tableau.
    tableau = np.empty((rows1 + rows2, cols1 + cols2 - 1), dtype=int)

    # Separate tableau1 into X and Z parts, pad with zeros for the qubits in tableau2.
    _tableau1_x = np.concatenate((tableau1[:, :n1], np.zeros((rows1, n2))), axis=1)
    _tableau1_z = np.concatenate((tableau1[:, n1:-1], np.zeros((rows1, n2))), axis=1)
    _tableau1 = np.concatenate((_tableau1_x, _tableau1_z), axis=1)
    _tableau1 = np.concatenate((_tableau1, tableau1[:, -1].reshape(rows1, 1)), axis=1)

    # Separate tableau2 into X and Z parts, pad with zeros for the qubits in tableau1.
    _tableau2_x = np.concatenate((np.zeros((rows2, n1)), tableau2[:, :n2]), axis=1)
    _tableau2_z = np.concatenate((np.zeros((rows2, n1)), tableau2[:, n2:-1]), axis=1)
    _tableau2 = np.concatenate((_tableau2_x, _tableau2_z), axis=1)
    _tableau2 = np.concatenate((_tableau2, tableau2[:, -1].reshape(rows2, 1)), axis=1)

    # Stack the rows:
    # Destabilizers: tableau1 first n1 rows, then tableau2 first (n2 - m2) rows.
    tableau[:n1, :] = _tableau1[:n1, :]
    tableau[n1:(n1 + n2 - m2), :] = _tableau2[:(n2 - m2), :]

    # Stabilizers: tableau1 next n1 rows, then tableau2 next (n2 - m2) rows.
    tableau[(n1 + n2 - m2):(2 * n1 + n2 - m2), :] = _tableau1[n1:, :]
    tableau[(2 * n1 + n2 - m2):(2 * (n1 + n2 - m2)), :] = _tableau2[(n2 - m2):2 * (n2 - m2), :]

    # JW elements: tableau2 remaining 2*m2 rows.
    tableau[(2 * (n1 + n2 - m2)):, :] = _tableau2[2 * (n2 - m2):, :]

    return tableau


def right_compose(tableau1, tableau2, m1, m2):
    """
    Compose two CNC tableaus via right composition.

    This function composes two tableaus that follow the canonical CNC tableau
    structure. Here, the right tableau (tableau2) must be a pure stabilizer tableau,
    which means its type parameter m2 must be 0. The left tableau (tableau1) can have
    nonzero m1 (i.e. nontrivial JW elements). The function performs the following steps:
    
      1. Compute the number of qubits (n1 and n2) from the shape of each tableau.
      2. Split each tableau into its X and Z blocks (ignoring the phase column) and pad
         with zeros so that the X and Z parts align when combined.
      3. Recombine the padded X and Z parts with the phase column.
      4. Stack the rows in the following order:
         - Destabilizers: first n2 rows from tableau2, then the first (n1 - m1) rows from tableau1.
         - Stabilizers: next n2 rows from tableau2, then the next (n1 - m1) rows from tableau1.
         - JW elements: the remaining 2*m1 rows from tableau1.
    
    Parameters:
        tableau1 (np.ndarray): Left tableau of shape (r1, 2*n1 + 1) with structure:
                               [X-part | Z-part | phase].
        tableau2 (np.ndarray): Right tableau of shape (r2, 2*n2 + 1) with structure:
                               [X-part | Z-part | phase].
        m1 (int): Type parameter for tableau1, affecting the number of JW rows.
        m2 (int): Type parameter for tableau2. Must be 0 (i.e. tableau2 is a stabilizer).

    Returns:
        np.ndarray: A composed tableau with shape ((r1 + r2) x ((2*n1 + 1) + (2*n2 + 1) - 1)).
                    The resulting tableau has merged and padded X and Z parts, a single phase column,
                    and rows arranged in the order: destabilizers, stabilizers, and JW elements.
    
    Raises:
        ValueError: If m2 is not 0.
    """
    if m2 != 0:
        raise ValueError("Composition is not valid. Right tableau must be a stabilizer.")

    # Determine number of qubits from each tableau.
    n1 = int((tableau1.shape[1] - 1) / 2)
    n2 = int((tableau2.shape[1] - 1) / 2)

    # Get dimensions of the input tableaus.
    rows1 = tableau1.shape[0]
    cols1 = tableau1.shape[1]
    rows2 = tableau2.shape[0]
    cols2 = tableau2.shape[1]

    # Initialize the new tableau.
    tableau = np.empty((rows1 + rows2, cols1 + cols2 - 1), dtype=int)

    # Separate tableau1 into X and Z parts, pad with zeros for tableau2's qubits.
    _tableau1_x = np.concatenate((tableau1[:, :n1], np.zeros((rows1, n2))), axis=1)
    _tableau1_z = np.concatenate((tableau1[:, n1:-1], np.zeros((rows1, n2))), axis=1)
    _tableau1 = np.concatenate((_tableau1_x, _tableau1_z), axis=1)
    _tableau1 = np.concatenate((_tableau1, tableau1[:, -1].reshape(rows1, 1)), axis=1)

    # Separate tableau2 into X and Z parts, pad with zeros for tableau1's qubits.
    _tableau2_x = np.concatenate((np.zeros((rows2, n1)), tableau2[:, :n2]), axis=1)
    _tableau2_z = np.concatenate((np.zeros((rows2, n1)), tableau2[:, n2:-1]), axis=1)
    _tableau2 = np.concatenate((_tableau2_x, _tableau2_z), axis=1)
    _tableau2 = np.concatenate((_tableau2, tableau2[:, -1].reshape(rows2, 1)), axis=1)

    # Stack the rows:
    # Destabilizers: tableau2 first n2 rows, then tableau1 first (n1 - m1) rows.
    tableau[:n2, :] = _tableau2[:n2, :]
    tableau[n2:(n1 + n2 - m1), :] = _tableau1[:(n1 - m1), :]

    # Stabilizers: tableau2 next n2 rows, then tableau1 next (n1 - m1) rows.
    tableau[(n1 + n2 - m1):(n1 + 2 * n2 - m1), :] = _tableau2[n2:, :]
    tableau[(n1 + 2 * n2 - m1):(2 * (n1 + n2 - m1)), :] = _tableau1[(n1 - m1):2 * (n1 - m1), :]

    # JW elements: tableau1 remaining 2*m1 rows.
    tableau[(2 * (n1 + n2 - m1)):, :] = _tableau1[2 * (n1 - m1):, :]

    return tableau


def compose_tableaus(tableau1, tableau2, m1, m2):
    """
    Compose two CNC tableaus by selecting the appropriate composition method.

    This wrapper function selects between left and right composition based on the type
    parameters m1 and m2:
    
      - If m1 == 0 (and m2 is nonzero), then the left composition is used.
      - Otherwise, m2 must be 0 and the right composition is used.
    
    Note:
      A valid composition requires that one of m1 or m2 be zero. If both m1 and m2 are
      nonzero, a ValueError is raised.

    Parameters:
        tableau1 (np.ndarray): Left tableau matrix with shape (r1, 2*n1 + 1).
        tableau2 (np.ndarray): Right tableau matrix with shape (r2, 2*n2 + 1).
        m1 (int): Type parameter for tableau1.
        m2 (int): Type parameter for tableau2.

    Returns:
        np.ndarray: The composed tableau, resulting from either left or right composition.

    Raises:
        ValueError: If both m1 and m2 are nonzero, with the message:
                    "Composition is not valid. One of m1 or m2 must be 0."
    """
    if (m1 != 0) and (m2 != 0):
        raise ValueError("Composition is not valid. One of m1 or m2 must be 0.")
    elif (m1 == 0) and (m2 != 0):
        return left_compose(tableau1, tableau2, m1, m2)
    else:
        return right_compose(tableau1, tableau2, m1, m2)
