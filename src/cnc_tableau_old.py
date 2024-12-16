import galois
import numpy as np
import random

# Define the field GF(2), which is F2
GF2 = galois.GF(2)


def symplectic_inner_product(u:np.ndarray, v:np.ndarray)->galois.GF(2):
    """Computes the symplectic inner product of vectors u and v over GF(2)."""
    n = len(u) // 2
    return (((u[0:n] @ v[n:])-(u[n:] @ v[0:n])) % 2)

def beta(u:np.ndarray, v:np.ndarray):
    n = int(len(u)/2)
    # check if paulis commute:
    if symplectic_inner_product(u,v) != 0:
        raise ValueError("Pauli elements must commute.")
    else:
        return (int(((u[n:] @ v[0:n])-(u[0:n] @ v[n:]))/2) % 2)

def bilinear_form(n:np.int64):
    return np.block([[np.zeros((n,n), dtype=int), np.eye(n, dtype=int)], 
                  [np.eye(n, dtype=int), np.zeros((n,n), dtype=int)]])

def initial_cnc_tableau(n,m)->np.ndarray:
    d = n-m; N = 2*n+1; xi = 2*m+1

    # initialize tableau:
    T = np.zeros([N,2*n],np.int64)

    # Define submatrices:
    A = np.zeros([m,m],np.int64)
    for i in range(m):
        for j in range(m):
            if i >= j:
                A[i,j] = 1
            else:
                A[i,j] = 0

    B = np.zeros([m,m],np.int64)
    for i in range(m):
        for j in range(m):
            if i > j:
                B[i,j] = 1
            else:
                B[i,j] = 0

    F = np.zeros(n,dtype = np.int64); F[n-m:] = 1;
    F = np.concatenate([F,F],axis = 0)

    # Modify tableau:
    T[0:d,n:n+d] = np.identity(d)
    T[d:2*d,0:d] = np.identity(d)
    T[2*d:2*d+m,d:d+m] = A; T[2*d:2*d+m,d+n:d+m+n] = B
    T[2*d+m:2*d+2*m,d:d+m] = B; T[2*d+m:2*d+2*m,d+n:d+m+n] = A
    T[N-1,:] = F

    gamma = np.zeros([N,1],np.int64)
    T = np.concatenate([T,gamma],axis = 1)
    
    return T

def coin_flips(t:int):
    return np.array([random.choice([0, 1]) for _ in range(t)])

class CncSimulator:
    """The bare minimum needed for the CNC simulation.

    Reference:
        "Phase-space-simulation method for quantum computation with magic states on qubits"
        Robert Raussendorf, Juani Bermejo-Vega, Emily Tyhurst, Cihan Okay, and Michael Zurel
        https://arxiv.org/abs/1905.05374
    """

    def __init__(self, n:int, m:int):
        self._n = n
        self._m = m
        self._d = n-m
        self._table = initial_cnc_tableau(n,m)
        self._mat = self._table[:,:-1]
        self._x = self._table[:, :self._n]
        self._z = self._table[:, self._n:-1]
        self._r = self._table[:, -1]
        self._J = bilinear_form(self._n)
        self._D = self._table[:self._d,:]
        self._S = self._table[self._d:2*self._d,:]
        self._W = self._table[2*self._d:,:]


    def is_cnc_(self):
        # remove phase column:
        T = GF2(self._table[:,:-1])
        J = GF2(self._J)

        # compute all symplectic inner products:
        P = T @ J @ np.transpose(T)

        n = self._n; m = self._m; d = n-m; xi = 2*m+1

        # submatrices for symplectic inner product for, e.g., destabilizer-destabilizer (DD), etc.
        DD = P[0:d,0:d]; DW = P[0:d,2*d:]
        SS = P[d:2*d,d:2*d]; SW = P[d:2*d,2*d:]
        WW = P[2*d:,2*d:]
        DS = P[0:2*d,0:2*d]

        WW_target = np.ones([xi,xi],np.int64)
        for i in range(xi): WW_target[i,i] = 0

        # Evaluate submatrices to see if as necessary for CNC tableau.
        bool_DD = np.array_equal(DD,np.zeros([d,d]))
        bool_DW = np.array_equal(DW,np.zeros([d,xi]))
        bool_SS = np.array_equal(SS,np.zeros([d,d]))
        bool_SW = np.array_equal(SW,np.zeros([d,xi]))
        bool_WW = np.array_equal(WW,WW_target)
        bool_DS = np.array_equal(DS,bilinear_form(d))

        bool_all = bool_DD*bool_DW*bool_SS*bool_SW*bool_WW*bool_DS

        if bool_all == True:
            return True
        else:
            raise ValueError("Not a CNC tableau.")

    def cnot(self, control: int, target: int) -> None:
        """Applies a CNOT gate between two qubits.

        Args:
            control: The control qubit of the CNOT.
            target: The target qubit of the CNOT.
        """
        self._r[:] ^= self._x[:, control] & self._z[:, target] & (
                self._x[:, target] ^ self._z[:, control] ^ True)
        self._x[:, target] ^= self._x[:, control]
        self._z[:, control] ^= self._z[:, target]

    def hadamard(self, qubit: int) -> None:
        """Applies a Hadamard gate to a qubit.

        Args:
            qubit: The qubit to apply the H gate to.
        """
        self._r[:] ^= self._x[:, qubit] & self._z[:, qubit]
        # Perform a XOR-swap
        self._x[:, qubit] ^= self._z[:, qubit]
        self._z[:, qubit] ^= self._x[:, qubit]
        self._x[:, qubit] ^= self._z[:, qubit]

    def phase(self, qubit: int) -> None:
        """Applies an S gate to a qubit.

        Args:
            qubit: The qubit to apply the S gate to.
        """
        self._r[:] ^= self._x[:, qubit] & self._z[:, qubit]
        self._z[:, qubit] ^= self._x[:, qubit]

    

    

    def is_phase_clifford_(self,L:tuple)->bool:
        bool1 = (len(L) == 2)
        bool2 = (L[0] == 'p')
        bool3 = (int(L[1]) in range(1,self._n+1))
        if bool1*bool2*bool3 == True:
            return True
        else:
            raise ValueError("Not a valid phase Clifford unitary.")

    def is_hadamard_clifford_(self,L:tuple)->bool:
        bool1 = (len(L) == 2)
        bool2 = (L[0] == 'h')
        bool3 = (int(L[1]) in range(1,self._n+1))
        if bool1*bool2*bool3 == True:
            return True
        else:
            raise ValueError("Not a valid hadamard Clifford unitary.")

    def is_cnot_clifford_(self,L:tuple)->bool:
        bool1 = (len(L) == 3)
        bool2 = (L[0] == 'cnot')
        bool3 = (int(L[1]) in range(1,self._n+1))
        bool4 = (int(L[2]) in range(1,self._n+1))
        if bool1*bool2*bool3*bool4 == True:
            return True
        else:
            raise ValueError("Not a valid cnot Clifford unitary.")

    

    
    def apply_sequence_of_clifford(self, L:list[str])->None:

        for l in L:
            c = tuple(l.split('_'))

            if c[0] == 'p':
                self.is_phase_clifford_(c)
                self.phase(int(c[1])-1)
            elif c[0] == 'h':
                self.is_hadamard_clifford_(c)
                self.hadamard(int(c[1])-1)
            elif c[0] == 'cnot':
                self.is_cnot_clifford_(c)
                self.cnot(int(c[1])-1,int(c[2])-1)
            else:
                raise ValueError("Not a valid Clifford unitary.")
    

    def _row_product_sign(self, i: int, k: int) -> int:
        """Determines the sign of two commuting rows' Pauli Products."""

        return (self._r[i]+self._r[k]+beta(self._mat[i,:],self._mat[k,:])) % 2

    def _row_mult(self, i: int, k: int) -> None:
        """Multiplies row k's Paulis into row i's Paulis."""
        self._r[i] = self._row_product_sign(i, k)
        self._mat[i,:] ^= self._mat[k,:]
        #self._x[i, :self._n] ^= self._x[k, :self._n]
        #self._z[i, :self._n] ^= self._z[k, :self._n]

    def _row_add(self, i: int, k: int) -> None:
        """Bitwise XOR of row k into row i."""
        self._mat[i,:] ^= self._mat[k,:]
        #self._x[i, :self._n] ^= self._x[k, :self._n]
        #self._z[i, :self._n] ^= self._z[k, :self._n]
    

    def measure(self,b:np.ndarray):
        if self.is_cnc_() == False:
            raise ValueError("Must be valid CNC tableau.")
        
        # tableau without phase column:
        T = GF2(self._mat)
        # symplectic form
        J = GF2(self._J)
        # measured Pauli:
        b = GF2(b)

        # Compute symplectic inner product with b:
        A = T @ J @ (b.reshape(2*self._n,1))

        # useful parameters:
        d = self._d; x = 2*self._m+1; N = 2*d+x
        
        # Segment the array
        A_d = tuple([a[0] for a in A[:d]])        # First d elements
        A_2d = tuple([a[0] for a in A[d:2*d]])    # Second d elements
        A_N = tuple([a[0] for a in A[2*d:N]])    # Last x elements

        # Case (1a)
        if all(a == GF2(0) for a in A_2d) and all(a == GF2(0) for a in A_N):
            return self.measure_b_in_stab(A_d)
        
        # Case (1b)
        elif all(a == 0 for a in A_2d) and A_N.count(GF2(0)) == 1:
            i = A_N.index(0)
            W = tuple([i for i in range(2*d, N) if A[i] == 1])
            return self.measure_b_in_coset(i,A_d,W)

    
        # Case (1c)
        elif all(a == 0 for a in A_2d) and 2 <= A_N.count(1) <= 2*self._m:
            # all anticommuting coset rows
            K = ([i for i in range(2*self._d, self._table.shape[0]) if A[i] == 1])
            # all commuting coset rows
            C = ([i for i in range(2*self._d, self._table.shape[0]) if A[i] == 0])
            return self.measure_b_in_norm(b,K,C)

        # Case (2)
        elif not all(a == 0 for a in A_2d):
            # least index anticommuting row in stabilizer:
            p = min([i for i in range(d, 2*d) if A[i] == 1])
            # all anticommuting rows
            K = tuple([i for i in range(N) if ((A[i] == 1) and (i != p))])
            return self.measure_b_not_in_norm(b,p,K)
    
    


    def measure_b_in_stab(self,A:np.ndarray):
        # Print case:
        print("Pauli is in stabilizer")

        # Define "scratch space" for determining outcome:
        outcome = 0; e = (np.zeros([2*self._n],np.int64)); d = self._n-self._m

        # For each anticommuting destabilizer iteratively sum corresponding stabilizer:
        for k in range(len(A)):
            if A[k] != 0:
                f = (self._mat[k+d,:]); s = (self._r[k+d])
                outcome = (outcome + s + beta(e,f)) % 2
                e = e+ f
        
        return outcome

    def measure_b_in_coset(self,i:int,A:np.ndarray,B:tuple[int]):
        print("Pauli is in coset of stabilizer")

        # Define "scratch space" for determining outcome:
        d = self._d; outcome = self._r[2*d+i]; e = self._mat[2*d+i,:]

        # For each anticommuting destabilizer iteratively sum corresponding stabilizer:
        for k in range(len(A)):
            if A[k] != 0:
                f = (self._mat[k+d,:]); s = (self._r[k+d])
                outcome = (outcome + s + beta(e,f)) % 2
                e = e+ f
        
        s = random.randint(0, 1)

        self._r[B] = (self._r[B]+s) % 2

        return outcome

    def measure_b_not_in_norm(self,b:np.ndarray,p:int,K:tuple[int]):
        print("Pauli is not in normalizer of stabilizer")

        # Use rowsum for all anticommuting rows besides p:
        for k in K:
            # If in destabilizer just add rows (phase bits don't matter)
            if k < self._d:
                self._row_add(k,p)
            # Otherwise, use normal rowsum
            else:
                self._row_mult(k,p)
        
        # Replace anticommuting row in stabilizer with measurement b:
        self._table[[p-self._d,p]] = self._table[[p,p-self._d]]

        # Replace anticommuting row in stabilizer with measurement b:
        self._mat[p,:] = b; self._r[p] = random.randint(0, 1)

        return self._r[p]


    def measure_b_in_norm(self,b:np.ndarray,K:list[int],C:list[int]):
        print("Pauli is in normalizer of stabilizer but not in any coset.")

        # Number of anticommuting elements:
        T = len(K); t = int(T/2); kmax = max(K)

        for k in K:
            if k > min(C):
                self._table[[min(C),k]] = self._table[[k,min(C)]]
                K[K.index(k)], C[C.index(min(C))] = C[C.index(min(C))], K[K.index(k)]
        
        # Sort arrays:
        K = sorted(K); C = sorted(C)
        
        # generate new stabilizer/destabilizer:
        for i in range(1,t):
            self._mat[K[T-1-i],:] = self._mat[K[i-1],:]^self._mat[K[T-i-1],:]   # New stabilizer
            self._mat[K[i-1],:] = self._mat[K[i-1],:]^self._mat[K[T-1],:]       # New destabilizer
        
        # Random outcome for new stabilizer generators:
        self._r[K[t:-1]] = coin_flips(t-1)
        
        # define bar_b as the sum of all coset elements that commute with b:
        self._table[K[T-1],:] = np.zeros([1,2*self._n+1],np.int64)
        for c in C:
            self._row_add(K[T-1],c)
        
        
        # Sample from uniform distribution over Z2:
        self._r[K[T-1]] = random.randint(0, 1); outcome = self._r[K[T-1]]

        # Change coset element by multiplying by bar_b:
        for c in C:
            self._row_mult(c,K[T-1])
        
        
        # exchange stabilizer and destabilizer
        source = [i for i in range(self._d,2*self._d)]; target = [i+t for i in range(self._d,2*self._d)]
        self._table[source+target] = self._table[target+source]

        # rearrange to form symplectic basis:
        subarray = self._table[2*self._d+t:2*self._d+2*t,:]
        subarray[:,:] = np.roll(subarray,shift=self._d,axis=0)

        # update m:
        self._m = self._m-t

        return outcome

    


def find_kernel_basis(A):
    """
    Find a basis for the kernel (null space) of the matrix A over F2 using the galois package.
    
    Parameters:
    A (numpy array or list of lists): The matrix for which to find the kernel.
    
    Returns:
    numpy array: A matrix whose columns form a basis for the kernel of A.
    """
    # Define the finite field GF(2)
    GF2 = galois.GF(2)
    
    # Convert the input matrix to a galois array
    A_gf = galois.GF2(A)
    
    # Compute the null space of A_gf
    null_space_matrix = A_gf.null_space()
    
    return null_space_matrix

