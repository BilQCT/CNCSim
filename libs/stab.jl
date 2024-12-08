# Stabilizer theory



"""
Stabilizer Theory

Functions for working with stabilizer states and related operations.
"""

include("symplectic.jl");include("pauli.jl");include("utils.jl")


function tensor_prod(A, n)
    """
    Compute the tensor product of a matrix `A` with itself `n` times.

    Parameters:
    - `A`: The matrix to be tensor producted.
    - `n`: The number of times to tensor product `A` with itself.

    Returns:
    - The resulting tensor product matrix.
    """
    An = A
    for i in 1:(n - 1)
        An = kron(An, A)
    end
    return An
end

function pauli(x)
    """
    Compute the Pauli operator corresponding to a given bit string.

    Parameters:
    - `x`: The bit string.

    Returns:
    - The Pauli operator corresponding to the input bit string.
    """
    if typeof(length(x) // 2) != Rational{Int64}
        return "Must be a bit-string of even length."
    elseif typeof(length(x) // 2) == Rational{Int64}
        n = Int(length(x) // 2)
        X = Matrix([0 1; 1 0])
        Z = Matrix([1 0; 0 -1])
        a = x[1:n]
        b = x[n + 1:end]
        if n == 1
            XaZb = (X^a[1] * Z^b[1])
        else
            XaZb = (X^a[1] * Z^b[1])
            for i in 2:n
                XaZb = kron(XaZb, (X^a[i] * Z^b[i]))
            end
        end
        arg_phase = (transpose(a) * b) % 4
        phase = im ^ (arg_phase)
        return phase * XaZb
    end
end

function sum_bitstrings(x, y)
    """
    Compute the sum of two bit strings.

    Parameters:
    - `x`: The first bit string.
    - `y`: The second bit string.

    Returns:
    - The sum of the two bit strings.
    """
    if length(x) != length(y)
        return "Bit-strings must be the same length"
    else
        n = length(x)
        return [(x[i] + y[i]) % 2 for i in 1:n]
    end
end

function omega(a, b)
    """
    Compute the symplectic inner product of two bit strings.

    Parameters:
    - `a`: The first bit string.
    - `b`: The second bit string.

    Returns:
    - The symplectic inner product of the two bit strings.
    """
    n = Int(length(a) / 2)
    ax = a[1:n]
    az = a[n + 1:end]
    bx = b[1:n]
    bz = b[n + 1:end]
    omega_ab = ((transpose(ax) * bz) + (transpose(az) * bx)) % 2
    return omega_ab
end



################################################################################################


"""
Check if a set of Pauli operators commute with each other.

Parameters:
- `S`: An array of Pauli operators.

Returns:
- `true` if all elements in `S` commute with each other, otherwise `false`.
"""
function commuting_set(S)
    for a in S
        for b in S
            if omega(a, b) == 1
                return false
            end
        end
    end
    return true
end

################################################################################################

using LinearAlgebra

"""
Compute the beta coefficient between two Pauli operators.

Parameters:
- `a`: The first Pauli operator.
- `b`: The second Pauli operator.

Returns:
- The beta coefficient between `a` and `b`, or an error message if they don't commute.
"""
function beta(a, b)
    if omega(a, b) == 1
        return "Must be commuting Paulis"
    else
        n = Int(length(a) / 2)
        Ta = pauli(a)
        Tb = pauli(b)
        Tab = pauli(sum_bitstrings(a, b))
        Id_n = tensor_prod(I(2), n)
        
        # Check beta:
        if Ta * Tb * Tab == Id_n
            return 0
        else
            return 1
        end
    end
end

################################################################################################

"""
Generate canonical generators for an isotropic subspace.

Parameters:
- `n`: The number of qubits.

Returns:
- An array containing the canonical generators.
"""
function canonical_generators(n)
    gens = Vector{Vector{Int}}([])
    
    for i in 1:n
        g = Vector{Int}([0 for j in 1:(2 * n)])
        g[i] = 1
        push!(gens, g)
    end
    
    return gens
end

################################################################################################

"""
Compute the canonical maximal isotropic subspace.

Parameters:
- `n`: The number of qubits.

Returns:
- The canonical maximal isotropic subspace.
"""
function canonical_maximal_isotropic(n)
    gens = canonical_generators(n)
    return isotropic(gens)
end

################################################################################################

"""
Generate elements of an isotropic subspace.

Parameters:
- `N`: The recursion parameter.
- `n`: The current recursion level.
- `gens`: The generators of the isotropic subspace.
- `iso`: The current elements of the isotropic subspace.

Returns:
- All elements of the isotropic subspace.
"""
function generate_elements(N, n, gens, iso)
    if n < N
        m = length(iso)
        a = gens[n]
        iso_prime = [sum_bitstrings(iso[i], a) for i in 1:m]
        iso = vcat(iso, iso_prime)
        n += 1
        generate_elements(N, n, gens, iso)
    else
        return collect(Set(iso))
    end   
end

################################################################################################

"""
Compute the isotropic subspace given its generators.

Parameters:
- `gens`: The generators of the isotropic subspace.

Returns:
- All elements of the isotropic subspace.
"""
function isotropic(gens)
    gens = collect(Set(gens))

    if length(gens) == 1
        L = length(gens[1])
        Id = [0 for i in 1:L]
        return [Id, gens[1]]
    end
    
    n = Int(length(gens[1]) / 2)
    k = length(gens)
    iso = copy(gens)
    
    for i in 1:k
        a = gens[i]
        for j in 1:k
            if j != i
                b = gens[j]
                if omega(a, b) == 1
                    error("All elements must commute")
                end
            end
        end
    end
    
    n = 1
    return collect(generate_elements(k, n, gens, iso))
end




################################################################################################


"""
Converts an array of generators to a matrix.

Parameters:
- `gens`: An array of generators.

Returns:
- A matrix representation of the generators.
"""
function gens_to_check_matrix(gens)
    gens = collect(gens)
    m = length(gens)
    k = length(gens[1])
    M = Array{Int64}(undef, 0, k)
    
    for i in 1:m
        gi = Vector{Int64}(gens[i])
        M = vcat(M, transpose(gi))
    end
    
    return M
end


using Nemo
using Combinatorics

"""
Generate all deterministic vertices for a given set of isotropic subspaces.

Parameters:
- `n`: The number of qubits.
- `isotropic`: The set of isotropic subspaces.

Returns:
- All deterministic vertices.
"""
function find_gens(n, isotropic)
    ZZ2 = residue_ring(ZZ, 2)
    Z2 = ZZ2[1]

    S1 = matrix_space(Z2,2^n,2*n); S2 = matrix_space(Z2,n,2*n)
    M = S1(gens_to_check_matrix(isotropic))
    rnk = rank(M)

    id = [0 for i in 1:(2*n)]; idx = findall(x -> x != id,isotropic)
    combs = combinations(isotropic[idx], rnk)
    for c in combs
        A = S2(gens_to_check_matrix(c))
        r = rank(A)
        if r == rnk
            return c
        end
    end
end


"""
Generate the value assignments for generators.

Parameters:
- `gens`: An array of generators.

Returns:
- A list of value assignments.
"""
function Gamma(gens)
    N = length(gens)
    Z = all_dit_strings(2, N)
    Gamma = []
    
    for z in Z
        gamma = Dict(zip(gens, z))
        identity = [0 for i in 1:length(gens[1])]
        gamma[identity] = 0
        
        for i in 2:N
            combs = collect(combinations(gens, i))
            for j in 1:length(combs)
                a = combs[j][1]
                s = gamma[a]
                for c in combs[j][2:end]
                    bta = beta(a, c)
                    a = sum_bitstrings(a, c)
                    s = (s + gamma[c] + bta) % 2
                end
                
                pair = Pair(a, s)
                if pair ∉ gamma
                    gamma[a] = s
                end
            end
        end
        push!(Gamma, gamma)
    end
    
    return Gamma
end


"""
Generate stabilizer coefficients for given isotropic subspaces.

Parameters:
- `n`: The number of qubits.
- `II`: The set of isotropic subspaces.

Returns:
- Stabilizer coefficients.
"""
function stabilizer_coefficients(n, II)
    PS = PauliString(n)
    En = PS.bit_strings;  N = length(En)
    Pn = PS.pauli_strings;
    dict = PS.bit_to_int

    # initialize array:
    A = Array{Int64}(undef, 0, N)
    
    for iso in II
        iso = collect(iso)
        gens = find_gens(n, iso)
        Gamma_iso = Gamma(gens)
    
        for gamma in Gamma_iso
            a = zeros(Int64, 1, N)
            idx_iso = [dict[a] for a in collect(keys(gamma))]
            coefficients = [(-1)^s for s in collect(values(gamma))]
            a[idx_iso] .= coefficients
            A = vcat(A, a)
        end
    end
    return A
end


function stabilizer_states(n)
    SP = symplectic_tuple = SympPerm(n)
    #sp = SP.Group; fdict = SP.fdict; bdict = SP.bdict;

    # generate canonical S = {I,X} maximal isotropic:
    I = Set(canonical_maximal_isotropic(n))
    #I = Set(Vector{Vector{Int}}(canonical_maximal_isotropic(n)))

    # compute orbit:
    II = collect(SympOrbit(n,SP,I).Orbit)

    return stabilizer_coefficients(n, II)
end


"""
Save isotropic subspaces to a text file.

Parameters:
- `II`: The set of isotropic subspaces.
- `n`: The number of qubits.
"""
function save_isotropics(II, n)
    N = 2 * n
    M = Array{Int64}(undef, N, 0)
    
    for i in 1:length(II)
        for j in II[i]
            cj = Vector{Int64}(j)
            M = hcat(M, cj)
        end
    end
    
    nn = string(n)
    
    open(nn * "_qubit_isotropics.txt", "w") do io
        writedlm(io, M, ',')
    end
end


"""
Load isotropic subspaces from a text file.

Parameters:
- `n`: The number of qubits.

Returns:
- The loaded isotropic subspaces.
"""
function load_isotropics(n)
    filename = string(n) * "_qubit_isotropics.txt"
    A = readdlm(filename, ',', Int64)
    N = size(A)[2]
    K = 2^n
    M = Int(N / K)
    Isotropics = []
    
    for i in 1:M
        isotropic = []
        for j in 1:K
            k = K * (i - 1) + j
            ak = A[:, k]
            push!(isotropic, ak)
        end
        push!(Isotropics, isotropic)
    end
    
    return Isotropics
end


"""
Get local isotropic subspaces from the given isotropic subspaces.

Parameters:
- `II`: The set of isotropic subspaces.
- `n`: The number of qubits.

Returns:
- The local isotropic subspaces.
"""
function local_isotropics(II,n)
    PS = PauliString(n)
    En = PS.bit_strings; pn = PS.pauli_strings
    dict = PS.pauli_to_bit

    # Extract local paulis:
    En_loc = [dict[p] for p in pn if length(findall(x -> x == 'I', p)) == (n - 1)]

    II_loc = []
    for iso in II
        iso_loc = [a for a in iso if a in En_loc]
        if length(iso_loc) == n
            push!(II_loc, iso)
        end
    end
    
    return II_loc
end


"""
Get local inequalities from the given isotropic subspaces.

Parameters:
- `In`: The set of isotropic subspaces.
- `n`: The number of qubits.

Returns:
- The local inequalities.
"""
function local_inequalities(In, n)
    Inloc = local_isotropics(In, n)
    N = Int(2^n)
    all_indices = []
    
    for i in 1:length(In)
        if In[i] in Inloc
            indices = [N * (i - 1) + j for j in 1:N]
            all_indices = vcat(all_indices, indices)
        end
    end
    
    return all_indices
end














"""
Generate all deterministic vertices for a given set of isotropic subspaces.

Parameters:
- `Iloc`: The local isotropic subspaces.
- `n`: The number of qubits.

Returns:
- All deterministic vertices.
"""
function deterministic_vertices(Iloc, n)
    PSn = PauliString(n);
    E = PS.bit_strings; dim = length(E); P = PS.pauli_strings;
    dict1 = PS.pauli_to_bit; dict2 = PS.bit_to_int
    
    Eloc = [dict1[p] for p in P if length(findall(x -> x == 'I', p)) == (n - 1)]
    N = length(Eloc); Z = all_dit_strings(2, N)

    # initialize matrix:
    D = Array{Int}(undef, 4^n, 0)

    for z in Z
        dict3 = Dict(zip(Eloc, z))
        d = [0 for i in 1:dim]
        d[1] = 1
        index_loc = [dict2[a] for a in Eloc]
        values_loc = [(-1)^s for s in z]
        d[index_loc] = values_loc
        
        for iso in Iloc
            g = [a for a in iso if a in Eloc]
            G = length(g)
            values_nloc = []
            
            for i in 2:G
                combs = collect(combinations(g, i))
                
                for j in 1:length(combs)
                    a = [0 for k in 1:(2 * n)]
                    s = 0
                    
                    for c in combs[j][1:end]
                        a = sum_bitstrings(a, c)
                        s = (s + dict3[c]) % 2
                    end
                    
                    d[dict2[a]] = (-1)^s
                end
            end
        end
        
        D = hcat(D, d)
    end
    
    return D
end



function gf_to_pauli_string(x)
    n = div(length(x), 2)
    strg = ""
    for i in 1:n
        if x[i] == 0 && x[i+n] == 0
            strg *= "i"
        elseif x[i] == 1 && x[i+n] == 0
            strg *= "x"
        elseif x[i] == 0 && x[i+n] == 1
            strg *= "z"
        elseif x[i] == 1 && x[i+n] == 1
            strg *= "y"
        else
            return "error"
        end
    end
    return strg
end


