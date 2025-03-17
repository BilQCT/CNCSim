import numpy as np
from itertools import product


def symplectic_inner_product(u: np.ndarray, v: np.ndarray) -> int:
    """Computes the symplectic inner product of two binary vectors over GF(2).

    The symplectic inner product is defined as:
        ω(u, v) = (u_z · v_x - u_x · v_z) mod 2

    If the result is 0, the corresponding Pauli operators commute.
    If the result is 1, they anticommute.

    Args:
        u (np.ndarray): Binary vector of a Pauli operator.
        v (np.ndarray): Binary vector of a Pauli operator.

    Returns:
        int: The symplectic inner product of the two vectors.
    """
    n = len(u) // 2
    ux, uz = u[:n].astype(np.int64), u[n:].astype(np.int64)
    vx, vz = v[:n].astype(np.int64), v[n:].astype(np.int64)

    return (uz.dot(vx) - ux.dot(vz)) % 2


def beta(
    u: np.ndarray, v: np.ndarray, skip_commutation_check: bool = False
) -> int:
    """
    Computes the beta value which determines the sign of the product of two
    commuting Pauli operators.

    Args:
        u (np.ndarray): Binary vector of a Pauli operator.
        v (np.ndarray): Binary vector of a Pauli operator.
        skip_commutation_check (bool, optional): If True, skips the commutation
            check for efficiency. Defaults to False.

    Returns:
        int: The beta value of the two commuting Pauli operators.

    Raises:
        AssertionError: If the Pauli operators do not commute and
            skip_commutation_check is False.
    """
    n = len(u) // 2

    if not skip_commutation_check:
        assert symplectic_inner_product(u, v) == 0, "Vectors do not commute!"

    ux, uz = u[:n], u[n:]
    vx, vz = v[:n], v[n:]

    x_terms = (ux ^ vx) % 2
    z_terms = (uz ^ vz) % 2

    u_phase = ux.dot(uz) % 4
    v_phase = vx.dot(vz) % 4
    combined_phase = x_terms.dot(z_terms) % 4

    tilde_beta = (
        u_phase.astype(np.int32)
        + v_phase.astype(np.int32)
        + 2 * uz.astype(np.int32).dot(vx.astype(np.int32))
        - combined_phase.astype(np.int32)
    ) % 4

    return tilde_beta // 2


def pauli_binary_vec_to_str(u: np.ndarray) -> str:
    """Converts binary vector of a Pauli operator to its string representation.

    For example [1, 0, 0, 1] is converted to "XZ".

    Args:
        u (np.ndarray): Binary vector of a Pauli operator.

    Returns:
        str: String representation of the Pauli operator associated to given
        binary vector (without phase).
    """
    pauli_str = ""

    n = len(u) // 2
    for i in range(n):
        if u[i] == 1 and u[i + n] == 1:
            pauli_str += "Y"
        elif u[i] == 1:
            pauli_str += "X"
        elif u[i + n] == 1:
            pauli_str += "Z"
        else:
            pauli_str += "I"

    return pauli_str


def pauli_str_to_binary_vec(pauli_str: str) -> np.ndarray:
    """
    Converts the string representation of a Pauli operator to its binary
    vector.

    Args:
        pauli_str (str): String representation of a Pauli operator.

    Returns:
        np.ndarray: Binary vector of the Pauli operator.
    """
    x_part = np.zeros(len(pauli_str), dtype=int)
    z_part = np.zeros(len(pauli_str), dtype=int)

    pauli_str = pauli_str.upper().strip()

    for i, op in enumerate(pauli_str):
        if op == "X":
            x_part[i] = 1
        elif op == "Y":
            x_part[i] = 1
            z_part[i] = 1
        elif op == "Z":
            z_part[i] = 1
        elif op == "I":
            pass
        else:
            raise ValueError(
                "Invalid Pauli string. It must contain only X, Y, Z, I."
            )

    return np.concatenate((x_part, z_part))


def get_pauli_vec_from_index(n: int, index: int) -> np.ndarray:
    """
    Returns the binary vector of a n-qubit Pauli operator from its index in the
    lexicographic order.

    Args:
        n (int): Number of qubits.
        index (int): Index of the Pauli operator.

    Returns:
        np.ndarray: Binary vector of the Pauli operator.
    """
    index_to_pauli = {0: "I", 1: "X", 2: "Y", 3: "Z"}
    inc_unit = 4 ** (n - 1)
    pauli_str = ""
    while inc_unit != 0:
        pauli_index, index = divmod(index, inc_unit)
        pauli_str += index_to_pauli[pauli_index]
        inc_unit //= 4

    return pauli_str_to_binary_vec(pauli_str)


def symplectic_matrix(n: int) -> np.ndarray:
    """Generates the 2n × 2n symplectic matrix ω over GF(2).

    The symplectic matrix S is defined as:
        ω = [[ 0  I ],
            [ I  0 ]]

    where `I` is the `n × n` identity matrix, and `0` is the `n × n` zero
    matrix.

    Args:
        n (int): The number of qubits (determines the size of the symplectic
        matrix).

    Returns:
        np.ndarray: A 2n × 2n symplectic matrix over GF(2).
    """
    zeros = np.zeros((n, n), dtype=np.uint8)
    identity = np.eye(n, dtype=np.uint8)

    return np.block([[zeros, identity], [identity, zeros]])


def find_m_from_omega_size(n: int, omega_size: int) -> int:
    """
    Computes the maximum value of m given the number of qubits and omega_size,
    based on the equation:
        omega_size = (2m + 2) * 2^(n - m)

    Args:
        n (int): Number of qubits.
        omega_size (int): Size of the omega set.

    Returns:
        int: The calculated maximum integer m satisfying the equation.

    Raises:
        ValueError: If no valid m (m > n) is found.
    """
    for m in range(n + 1):
        if ((m + 1) * (2 ** (n + 1 - m)) == omega_size):
            return m

    raise ValueError("Not a valid size for Omega")


def find_commuting_elements(
    vectors: list[np.ndarray]
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Seperates the list of vectors to the ones that commute with the all
    vectors (isotropic) and the others(non_isotropic). Either list can be
    empty.

    Args:
        vectors (list[np.ndarray]): List of binary vectors representing
            Pauli operators.

    Returns:
        tuple[list[np.ndarray], list[np.ndarray]]: A tuple containing two lists
            of binary vectors. The first list contains the isotropic vectors
            and the second list contains the non-isotropic vectors.
    """
    # Convert to tuples for faster lookup and set operations
    # Note: numpy arrays are not hashable
    vector_set = {tuple(v.tolist()) for v in vectors}
    isotropic_set = set()

    # Find vectors that commute with all other vectors
    for v in vectors:
        # A vector is isotropic if it commutes with every vector in the set
        if all(
            symplectic_inner_product(v, np.array(w)) == 0 for w in vector_set
        ):
            isotropic_set.add(tuple(v.tolist()))

    # Find the complementary elements (non-isotropic)
    non_isotropic_set = vector_set - isotropic_set

    # Convert back to numpy arrays
    isotropic_elements = [np.array(v) for v in isotropic_set]
    non_isotropic_elements = [np.array(w) for w in non_isotropic_set]

    return isotropic_elements, non_isotropic_elements


def find_jw_elements(
    non_stabilizer_vectors: list[np.ndarray]
) -> list[np.ndarray]:
    """
    Finds a set of jw elements from the list of non-stabilizer vectors.

    Args:
        non_stabilizer_vectors (list[np.ndarray]): List of binary vectors.

    Returns:
        list[np.ndarray]: List of jw elements.
    """
    vectors = set(non_stabilizer_vectors)
    jw_elements = []

    while len(vectors) > 0:
        v = vectors.pop()
        commuting_coset = {
            w for w in vectors if symplectic_inner_product(v, w) == 0
        }
        jw_elements.append(v)
        vectors -= commuting_coset

    # To make sure they are in JW elements form we need remove the last element
    # and add the sum of the rest of the elements instead
    jw_elements[-1] = (sum(jw_elements[:-1]) % 2)

    return jw_elements


def gaussian_elimination_mod2(A: np.ndarray) -> np.ndarray:
    """
    Performs Gaussian elimination on a binary matrix A over GF(2).

    Args:
        A (np.ndarray): A binary matrix.

    Returns:
        np.ndarray: A matrix containing a basis of the row space of A, with
        zero rows removed.
    """
    A = A.copy() % 2
    rows, cols = A.shape
    row_idx = 0

    for col in range(cols):
        # Find the first row with a leading one in the current column
        pivot_row = -1
        for i in range(row_idx, rows):
            if A[i, col] == 1:
                pivot_row = i
                break

        # If no pivot found, move to next column
        if pivot_row == -1:
            continue

        # Swap current row with the pivot row
        A[[row_idx, pivot_row]] = A[[pivot_row, row_idx]]

        # Eliminate all other 1s in this column
        for i in range(rows):
            if i != row_idx and A[i, col] == 1:
                A[i] ^= A[row_idx]

        row_idx += 1

    # Return the reduced basis, removing all zero rows
    basis = A[np.any(A, axis=1)]
    return basis


def generate_subspace_efficient(vectors: list[np.ndarray]) -> list[np.ndarray]:
    """
    Generates the subspace spanned by the input vectors over GF(2) efficiently
    using Gaussian elimination.

    Args:
        vectors (List[np.ndarray]): List of binary vectors.

    Returns:
        List[np.ndarray]: List of binary vectors in the generated subspace.
    """
    subspace = set()
    subspace.add(tuple(np.zeros(len(vectors[0]), dtype=int)))

    for v in vectors:
        current_basis = list(subspace)
        for b in current_basis:
            new_vector = (np.array(b) + v) % 2
            subspace.add(tuple(new_vector))

    return [np.array(v) for v in subspace]


def find_independent_subset(vectors: list[np.ndarray]) -> np.ndarray:
    """
    Finds a maximal linearly independent subset of binary vectors over GF(2).

    Args:
        vectors (List[np.ndarray]): List of binary vectors.

    Returns:
        np.ndarray: Array of linearly independent binary vectors over GF(2).
    """
    vectors_matrix = np.array(vectors, dtype=int)
    reduced_matrix = gaussian_elimination_mod2(vectors_matrix)
    independent_subset = []

    # Collect only input vectors that are linearly independent
    for v in vectors:
        if any(np.array_equal(v, row) for row in reduced_matrix):
            independent_subset.append(v)
            # Remove the row to avoid duplicates
            reduced_matrix = reduced_matrix[
                ~np.all(reduced_matrix == v, axis=1)
            ]

    return np.array(independent_subset, dtype=int)

# TODO: Decide if we want to keep the not-used naive approach
def find_complementary_subspace_naive(
    v_basis: list[np.ndarray], n: int
) -> np.ndarray:
    """
    Finds a basis for the complement subspace W such that U = V ⊕ W, using a
    naive approach.

    Args:
        v_basis (List[np.ndarray]): Basis of subspace V (each vector of length
            2n).
        n (int): Half the dimension of the space U (i.e. U is 2n-dimensional
            over GF(2)).

    Returns:
        np.ndarray: Array representing the complement subspace basis.
    """
    # generate all n-dimensional vectors over GF(2):
    u_vectorspace = [
        np.array(v, dtype=int) for v in product([0, 1], repeat=2*n)
    ]
    u_vectorspace = set(tuple(u.tolist()) for u in u_vectorspace)

    # generate all vectors in subspace V:
    v_subspace = generate_subspace_efficient(v_basis)
    print(f"Size of subspace: {len(v_subspace)}")
    v_subspace = set(tuple(v.tolist()) for v in v_subspace)

    # take set difference:
    w_subspace = list(u_vectorspace - v_subspace)

    return np.array(w_subspace, dtype=int)


def find_complementary_subspace(
    v_basis: list[np.ndarray], n: int
) -> np.ndarray:
    """
    Finds a basis for the complement subspace W such that U = V ⊕ W.

    Args:
        v_basis (list[np.ndarray]): Basis of subspace V (each vector of
            length 2n).
        n (int): Half the dimension of the space U (i.e. U is 2n-dimensional
            over GF(2)).

    Returns:
        np.ndarray: Array representing the complement subspace basis.
    """
    # Convert the basis of V into a matrix
    V_matrix = np.array(v_basis, dtype=int)
    rank_v = V_matrix.shape[0]  # This should be n+m
    dim_u = 2 * n

    # Create the canonical basis (identity matrix rows)
    canonical_basis = np.eye(dim_u, dtype=int)

    # Determine the required dimension of the complement basis
    required_complement_dim = dim_u - rank_v  # Should be n-m

    complement_basis = []

    # Find independent vectors
    for v in canonical_basis:
        if len(complement_basis) == required_complement_dim:
            break  # Terminate early when enough vectors are found

        extended_basis = np.vstack([V_matrix, v])
        reduced_basis = gaussian_elimination_mod2(extended_basis)

        # Check if v adds a new dimension
        if reduced_basis.shape[0] > rank_v:
            complement_basis.append(v)
            V_matrix = reduced_basis  # Update the basis with the new dimension
            rank_v = V_matrix.shape[0]  # This should be n+m

    return np.array(complement_basis, dtype=int)

# TODO: Decide if we want to keep the not-used naive approach
def generate_destabilizer_basis_naive(
    d_subspace: list[np.ndarray], w_basis: list[np.ndarray]
) -> list[np.ndarray]:
    """
    Generates a new destabilizer basis from the provided subspace and
    generating basis, using a naive approach.

    Args:
        d_subspace (List[np.ndarray]): List of vectors from the complementary
            subspace.
        w_basis (List[np.ndarray]): List of basis vectors from the generating
            subspace.

    Returns:
        List[np.ndarray]: Updated destabilizer basis.
    """
    new_destabilizer_basis = []

    for v in d_subspace:
        commuting_vectors = [
            w for w in w_basis
            if symplectic_inner_product(v, w) == 0
        ]
        anticommuting_vectors = [
            w for w in w_basis
            if not any(np.array_equal(w, cv) for cv in commuting_vectors)
        ]

        if len(anticommuting_vectors) == 0:
            new_destabilizer_basis.append(v)

    return new_destabilizer_basis


def generate_destabilizer_basis(
    d_basis: list[np.ndarray], w_basis: list[np.ndarray]
) -> list[np.ndarray]:
    """
    Generates a new destabilizer basis from the provided basis vectors.

    Args:
        d_basis (List[np.ndarray]): List of basis vectors from the
            complementary subspace.
        w_basis (List[np.ndarray]): List of basis vectors from the generating
            subspace.

    Returns:
        List[np.ndarray]: Updated destabilizer basis.
    """
    new_destabilizer_basis = []

    for v in d_basis:
        commuting_vectors = [
            w for w in w_basis
            if symplectic_inner_product(v, w) == 0
        ]
        anticommuting_vectors = [
            w for w in w_basis
            if not any(np.array_equal(w, cv) for cv in commuting_vectors)
        ]

        # Check the number of anticommuting vectors is even
        if len(anticommuting_vectors) % 2 == 0:
            v_prime = v.copy()
            for vec in anticommuting_vectors:
                v_prime = (v_prime + vec) % 2
            new_destabilizer_basis.append(v_prime)
        else:
            v_prime = v.copy()
            for vec in commuting_vectors:
                v_prime = (v_prime + vec) % 2
            new_destabilizer_basis.append(v_prime)

    return new_destabilizer_basis


def symplectic_gram_schmidt(
    array1: list[np.ndarray], array2: list[np.ndarray], r: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Performs the Symplectic Gram-Schmidt process on two lists of binary vectors
    over GF(2).

    Based off Symplectic Gram-Schmidt pseudo-code of
        https://arxiv.org/abs/1406.2170

    Args:
        array1 (List[np.ndarray]): List of binary vectors for the first set.
        array2 (List[np.ndarray]): List of binary vectors for the second set.
        r (int): Number of basis pairs to extract.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two numpy arrays representing the
            symplectic basis vectors from the two sets.
    """
    old_basis1 = [np.array(v, dtype=int) for v in array1]
    old_basis2 = [np.array(v, dtype=int) for v in array2]
    new_basis1 = []
    new_basis2 = []

    k = 0

    while k < r:
        found_pair = False

        for i, v in enumerate(old_basis1):
            commutations = np.array([
                symplectic_inner_product(v, w) for w in old_basis2
            ])

            # Check if anticommuting vector exists
            if not np.any(commutations):
                continue

            # Find first anticommuting vector
            j = np.argmax(commutations == 1)
            w = old_basis2[j]

            # Add to new bases and remove from old bases
            new_basis1.append(v)
            new_basis2.append(w)
            old_basis1.pop(i)
            old_basis2.pop(j)

            # Modify remaining vectors
            old_basis1 = [
                (u + symplectic_inner_product(u, w) * v) % 2
                for u in old_basis1
            ]
            old_basis2 = [
                (u + symplectic_inner_product(u, v) * w +
                    symplectic_inner_product(u, w) * v) % 2
                for u in old_basis2
            ]

            k += 1
            found_pair = True
            break

        if not found_pair:
            print("No anticommuting pair found. Terminating early.\n")
            break

    return np.array(new_basis1, dtype=int), np.array(new_basis2, dtype=int)


def is_symplectic(U: np.ndarray, V: np.ndarray, S: np.ndarray) -> bool:
    """
    Check whether two matrices U and V form a symplectic pair with respect to
    S over GF(2).

    A pair of matrices (U, V) is considered symplectic with respect to the
    symplectic form S if they satisfy the symplectic condition:
        (U @ S) @ V^T ≡ I (mod 2)
    where I is the identity matrix, and V^T is the transpose of V.

    Args:
        U (np.ndarray): Matrix of shape (dim, 2n).
        V (np.ndarray): Matrix of shape (dim, 2n).
        S (np.ndarray): Symplectic form matrix of shape (2n, 2n).

    Returns:
        bool: True if (U, V) is symplectic, False otherwise.

    Raises:
        ValueError: If U and V have different dimensions.

    Example:
        >>> n = 2
        >>> U = np.eye(2 * n, dtype=int)
        >>> V = np.eye(2 * n, dtype=int)
        >>> S = np.block([[np.zeros((n, n), dtype=int), np.eye(n, dtype=int)],
                          [np.eye(n, dtype=int), np.zeros((n, n), dtype=int)]])
        >>> is_symplectic(U, V, S)
        False
    """
    dim_U = U.shape[0]
    dim_V = V.shape[0]

    if dim_U != dim_V:
        raise ValueError("Matrices U and V must have the same number of rows.")

    # Check the symplectic condition
    symplectic_check = (U @ S @ V.T) % 2
    return np.all(symplectic_check == np.eye(dim_U, dtype=int))



#################################################################
#################################################################
#################################################################

# TODO: Merge the functions below into the CNCSimulator class
# tableau1: left and stabilizer, tableau2: right and cnc/stabilizer.
def left_compose(tableau1,tableau2,m1,m2):

    # check if composition is valid:
    if m1 != 0:
        raise ValueError("Composition is not valid. Left tableau must be a stabilizer.")

    # qubit numbers:
    n1 = int((tableau1.shape[1]-1)/2)
    n2 = int((tableau2.shape[1]-1)/2)

    # total rows and columns:
    rows1 = tableau1.shape[0]; cols1 = tableau1.shape[1]
    rows2 = tableau2.shape[0]; cols2 = tableau2.shape[1]

    # initialize tableau:
    tableau = np.empty((rows1+rows2,cols1+cols2-1),dtype = int)

    # separate into x-z pieces:
    _tableau1_x = np.concatenate((tableau1[:,:n1],np.zeros((rows1,n2))),axis=1)
    _tableau1_z = np.concatenate((tableau1[:,n1:-1],np.zeros((rows1,n2))),axis=1)
    _tableau1 = np.concatenate((_tableau1_x,_tableau1_z),axis=1)
    _tableau1 = np.concatenate((_tableau1,tableau1[:,-1].reshape(rows1,1)),axis=1)

    _tableau2_x = np.concatenate((np.zeros((rows2,n1)),tableau2[:,:n2]),axis=1)
    _tableau2_z = np.concatenate((np.zeros((rows2,n1)),tableau2[:,n2:-1]),axis=1)
    _tableau2 = np.concatenate((_tableau2_x,_tableau2_z),axis=1)
    _tableau2 = np.concatenate((_tableau2,tableau2[:,-1].reshape(rows2,1)),axis=1)

    # stack: destabilizer1 (n1), destabilizer2 (n2-m2)
    tableau[:n1,:] = _tableau1[:n1,:]
    tableau[n1:(n1+n2-m2),:] = _tableau2[:(n2-m2),:]

    # stack: stabilizer1 (n1), stabilizer2 (n2-m2)
    tableau[(n1+n2-m2):(2*n1+n2-m2),:] = _tableau1[n1:,:]
    tableau[(2*n1+n2-m2):(2*(n1+n2-m2)),:] = _tableau2[(n2-m2):2*(n2-m2),:]

    # stack: jw (2m2)
    tableau[(2*(n1+n2-m2)):,:] = _tableau2[2*(n2-m2):,:]

    return tableau


def right_compose(tableau1,tableau2,m1,m2):

    # check if composition is valid:
    if m2 != 0:
        raise ValueError("Composition is not valid. Right tableau must be a stabilizer.")

    # qubit numbers:
    n1 = int((tableau1.shape[1]-1)/2)
    n2 = int((tableau2.shape[1]-1)/2)

    # total rows and columns:
    rows1 = tableau1.shape[0]; cols1 = tableau1.shape[1]
    rows2 = tableau2.shape[0]; cols2 = tableau2.shape[1]

    # initialize tableau:
    tableau = np.empty((rows1+rows2,cols1+cols2-1),dtype = int)

    # separate tableau1 into x-z pieces and recombine
    _tableau1_x = np.concatenate((tableau1[:,:n1],np.zeros((rows1,n2))),axis=1)
    _tableau1_z = np.concatenate((tableau1[:,n1:-1],np.zeros((rows1,n2))),axis=1)
    _tableau1 = np.concatenate((_tableau1_x,_tableau1_z),axis=1)
    _tableau1 = np.concatenate((_tableau1,tableau1[:,-1].reshape(rows1,1)),axis=1)

    # separate tableau2 into x-z pieces and recombine:
    _tableau2_x = np.concatenate((np.zeros((rows2,n1)),tableau2[:,:n2]),axis=1)
    _tableau2_z = np.concatenate((np.zeros((rows2,n1)),tableau2[:,n2:-1]),axis=1)
    _tableau2 = np.concatenate((_tableau2_x,_tableau2_z),axis=1)
    _tableau2 = np.concatenate((_tableau2,tableau2[:,-1].reshape(rows2,1)),axis=1)

    # stack: destabilizer1 (n2), destabilizer2 (n1-m1)
    tableau[:n2,:] = _tableau2[:n2,:]
    tableau[n2:(n1+n2-m1),:] = _tableau1[:(n1-m1),:]

    # stack: stabilizer1 (n2), stabilizer2 (n1-m1)
    tableau[(n1+n2-m1):(n1+2*n2-m1),:] = _tableau2[n2:,:]
    tableau[(n1+2*n2-m1):(2*(n1+n2-m1)),:] = _tableau1[(n1-m1):2*(n1-m1),:]

    # stack: jw (2m1)
    tableau[(2*(n1+n2-m1)):,:] = _tableau1[2*(n1-m1):,:]
    
    return tableau


def compose_tableaus(tableau1,tableau2,m1,m2):
    # check if composition is valid:
    if (m1!=0) and (m2 !=0):
        raise ValueError("Composition is not valid. One of m1 or m2 must be 0.")
    elif (m1 == 0) and (m2 != 0):
        return left_compose(tableau1,tableau2,m1,m2)
    else:
        return right_compose(tableau1,tableau2,m1,m2)




