import numpy as np
import h5py

def find_m_from_omega(n, omega_size):
    """
    Find the maximum value of m given n and omega_size based on the equation:
        omega_size = (2m+2)*2^(n-m)

    Args:
        n: Number of qubits.
        omega_size: Size of the omega set.

    Returns:
        m: The calculated maximum integer satisfying the condition.
    """
    m = 1
    while True:
        if ((m+1) * (2 ** (n + 1 - m)) == omega_size):
            return m
        m += 1
        if m > n:
            raise ValueError("Not a valid size for Omega")


def find_commuting_elements(vectors):
    """
    Find isotropic generators and their complement from a list of vectors.

    Args:
        vectors: List of binary vectors as numpy arrays.

    Returns:
        isotropic_gens: List of isotropic generators.
        jw_gens: Complementary set of generators.
    """
    N = len(vectors)
    isotropic_elems = []
    
    # Ensure vectors are arrays and make a set for fast lookup
    vector_tuples = {tuple(v) for v in vectors}
    
    for v in vectors:
        K = sum(1 for w in vectors if symplectic_inner_product(v, w) == 0)
        if K == N:  # Commuting with all vectors
            isotropic_elems.append(tuple(v))
    
    # Find generators not in isotropic_gens
    complement_elems = list(set(vector_tuples) - set(isotropic_elems))

    # Convert back to numpy arrays
    isotropic_elems = [np.array(v) for v in isotropic_elems]
    complement_elems = [np.array(w) for w in complement_elems]

    if len(complement_elems) > 0:
        return isotropic_elems, complement_elems
    else:
        return isotropic_elems
    

def find_commuting_coset(v, vectors):
    """
    Find all vectors commuting with a given vector v.
    Args:
        v: Reference vector as a numpy array.
        vectors: List of binary vectors as numpy arrays.
    Returns:
        List of vectors commuting with v.
    """
    return [w for w in vectors if symplectic_inner_product(v, w) == 0]

def find_cosets(vectors):
    """
    Find all commuting cosets from a list of vectors.
    Args:
        vectors: List of binary vectors as numpy arrays.
    Returns:
        List of cosets.
    """
    cosets = []
    vectors = vectors.copy()

    while len(vectors) > 0:
        v = vectors[0]  # Select the first vector
        coset = find_commuting_coset(v, vectors)
        cosets.append(coset)
        # Remove the entire coset from the remaining vectors
        vectors = [w for w in vectors if not any(np.array_equal(w, c) for c in coset)]

    return cosets


import numpy as np

def gaussian_elimination_mod2(A):
    """
    Perform Gaussian elimination on matrix A over Z2.

    Args:
        A: Binary matrix over Z2 (numpy array).

    Returns:
        basis: Linearly independent row vectors of A, ordered by pivot columns.
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


def generate_all_vectors(n):
    """
    Generate all vectors of dimension n over Z2.

    Args:
        n: Dimension of the vector space.
        
    Returns:
        vectors: List of all binary vectors as numpy arrays.
    """
    vectors = [np.array(v, dtype=int) for v in product([0, 1], repeat=n)]
    return vectors

def generate_subspace_efficient(vectors):
    """
    Generate the subspace spanned by the input vectors over Z2 efficiently using Gaussian elimination.

    Args:
        vectors: List of binary vectors as numpy arrays.

    Returns:
        subspace: Set of unique binary vectors forming the subspace.
    """
    subspace = set()
    subspace.add(tuple(np.zeros(len(vectors[0]), dtype=int)))

    for v in vectors:
        current_basis = list(subspace)
        for b in current_basis:
            new_vector = (np.array(b) + v) % 2
            subspace.add(tuple(new_vector))

    return [np.array(v) for v in subspace]


def create_symplectic_form(n):
    """
    Create a 2n x 2n symplectic form matrix with the coordinate convention
    (x1, ..., xn, z1, ..., zn).

    Args:
        n: Dimension parameter (half the total space dimension).
        
    Returns:
        S: 2n x 2n symplectic form matrix.
    """
    # Create nxn identity matrix
    I = np.eye(n, dtype=int)

    # Arrange the symplectic form matrix
    S = np.block([
        [np.zeros((n, n), dtype=int), I],
        [I, np.zeros((n, n), dtype=int)]
    ])

    return S


def find_independent_subset(vectors):
    """
    Find a maximal linearly independent subset of binary vectors.
    
    Args:
        vectors: List of binary vectors as numpy arrays.
        
    Returns:
        A NumPy array of linearly independent binary vectors from the input.
    """
    vectors_matrix = np.array(vectors, dtype=int)
    reduced_matrix = gaussian_elimination_mod2(vectors_matrix)
    independent_subset = []

    # Collect only input vectors that are linearly independent
    for v in vectors:
        if any(np.array_equal(v, row) for row in reduced_matrix):
            independent_subset.append(v)
            # Remove the row to avoid duplicates
            reduced_matrix = reduced_matrix[~np.all(reduced_matrix == v, axis=1)]

    return np.array(independent_subset, dtype=int)



def find_complementary_subspace(v_basis, n):
    """
    Find a basis for the complement subspace W such that U = V ⊕ W.
    
    Args:
        v_basis: Basis of subspace V as a list of numpy arrays. Each vector is of length 2n.
        n: Half the dimension of the space U (2n-dimensional vector space over Z2).
        
    Returns:
        w_basis: Basis of the complement subspace W as a list of numpy arrays.
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



def generate_destabilizer_basis(d_basis, w_basis):
    """
    Generate a new destabilizer basis from given bases.

    Args:
        d_basis: Basis vectors from the complementary subspace.
        w_basis: Basis vectors from the generating subspace.

    Returns:
        new_destabilizer_basis: List of updated destabilizer vectors.
    """
    new_destabilizer_basis = []

    for v in d_basis:
        commuting_vectors = [w for w in w_basis if symplectic_inner_product(v, w) == 0]
        anticommuting_vectors = [w for w in w_basis if not any(np.array_equal(w, cv) for cv in commuting_vectors)]
        
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



import numpy as np

def symplectic_inner_product(v, w):
    """Calculate the symplectic inner product of two vectors over Z2."""
    n = len(v) // 2
    return (np.dot(v[:n], w[n:]) + np.dot(v[n:], w[:n])) % 2

def symplectic_gram_schmidt(array1, array2, r):
    """
    Perform the Symplectic Gram-Schmidt process on a set of vectors over Z2.

    Args:
        array1: A list of binary vectors for the first set.
        array2: A list of binary vectors for the second set.
        r: The number of basis pairs to find.

    Returns:
        Two numpy arrays representing the symplectic basis vectors.
    """
    old_basis1 = [np.array(v, dtype=int) for v in array1]
    old_basis2 = [np.array(v, dtype=int) for v in array2]
    new_basis1 = []
    new_basis2 = []

    k = 0

    while k < r:
        
        found_pair = False

        for i, v in enumerate(old_basis1):
            commutations = np.array([symplectic_inner_product(v, w) for w in old_basis2])

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
            old_basis1 = [(u + symplectic_inner_product(u, w) * v) % 2 for u in old_basis1]
            old_basis2 = [(u + symplectic_inner_product(u, v) * w + symplectic_inner_product(u, w) * v) % 2 for u in old_basis2]

            k += 1
            found_pair = True
            break

        if not found_pair:
            print("No anticommuting pair found. Terminating early.\n")
            break

    return np.array(new_basis1, dtype=int), np.array(new_basis2, dtype=int)



import numpy as np

def is_symplectic(U, V, S):
    """
    Check if two matrices form a symplectic pair over Z2.

    A pair of matrices (U, V) is considered symplectic with respect to the 
    symplectic form S if they satisfy the symplectic condition:
        (U @ S) @ V^T ≡ I (mod 2)
    where I is the identity matrix, and V^T is the transpose of V.

    Args:
        U (np.ndarray): First matrix of shape (dim_U, 2n).
        V (np.ndarray): Second matrix of shape (dim_V, 2n).
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




