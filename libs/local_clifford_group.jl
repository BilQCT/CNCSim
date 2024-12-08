using GAP
using LinearAlgebra
using Combinatorics

const g = GAP.Globals
const gjl = GAP.gap_to_julia
const jlg = GAP.julia_to_gap;

include("utils.jl"); include("pauli.jl"); include("symplectic.jl");



####################################################################################
####################################################################################
################################# BASIC CLIFFORD GATES #############################
####################################################################################
####################################################################################


R2 = (g.E(8)+(g.E(8))^7)        # root 2 using roots of unity
Id = jlg([1 0; 0 1])            # identity
P = jlg([1 0; 0 g.E(4)])        # Phase gate
H = (1/R2)*jlg([1 1; 1 -1])       # Hadamard gate
CNOT = jlg([1 0 0 0; 0 1 0 0;   # CNOT gate
            0 0 0 1; 0 0 1 0])



function hadamard_j(j::Int, n::Int)

    if j == 1
        Hj = H
    else
        Hj = Id
    end

    if n == 1
        return Hj
    else
        for i in 2:n
            if i != j
                Hj = g.KroneckerProduct(Hj, Id)
            else
                Hj = g.KroneckerProduct(Hj, H)
            end
        end
        return Hj
    end
end

function phase_j(j::Int, n::Int)

    if j == 1
        Pj = P
    else
        Pj = Id
    end

    if n == 1
        return Pj
    else
        for i in 2:n
            if i != j
                Pj = g.KroneckerProduct(Pj, Id)
            else
                Pj = g.KroneckerProduct(Pj, P)
            end
        end
        return Pj
    end
end
function cnot(c::Int, t::Int, n::Int)
    Proj00 = g.DiagonalMat(jlg([1, 0]))
    Proj11 = g.DiagonalMat(jlg([0, 1]))
    NOT = jlg([0 1; 1 0])

    if n == 1
        # Not a valid generator for single qubit
        return Id
    else
        if c == 1
            # If control = 0, apply identity to target
            CX1 = g.KroneckerProduct(Proj00, n_fold_tensor(n-1, Id))

            # If control = 1, apply X (NOT) gate to target
            if (t - c) == 1
                CX2 = g.KroneckerProduct(Proj11, NOT)
                if n == 2
                    return CX1 + CX2
                else
                    CX2 = g.KroneckerProduct(CX2, n_fold_tensor(n-t, Id))
                end
            else
                CX2 = g.KroneckerProduct(Proj11, n_fold_tensor(t-c-1, Id))
                CX2 = g.KroneckerProduct(CX2, NOT)
                CX2 = g.KroneckerProduct(CX2, n_fold_tensor(n-t, Id))
            end

            return CX1 + CX2
        else
            CX1 = g.KroneckerProduct(n_fold_tensor(c-1, Id), Proj00)
            CX1 = g.KroneckerProduct(CX1, n_fold_tensor(n-c, Id))

            CX2 = g.KroneckerProduct(n_fold_tensor(c-1, Id), Proj11)
            if t - c == 1
                CX2 = g.KroneckerProduct(CX2, NOT)
                CX2 = g.KroneckerProduct(CX2, n_fold_tensor(n-t, Id))
            else
                CX2 = g.KroneckerProduct(CX2, n_fold_tensor(t-c-1, Id))
                CX2 = g.KroneckerProduct(CX2, NOT)
                CX2 = g.KroneckerProduct(CX2, n_fold_tensor(n-t, Id))
            end
            #println("Control operation: $(CX1), \n Target operation: $(CX2)")
            return CX1 + CX2
        end
    end
end

function swap(c::Int, t::Int, n::Int)
    #result = nothing
    hc_ht = hadamard_j(c,n)*hadamard_j(t,n)
    if c < t
        cnot_ct = cnot(c,t,n)
        cnot_tc = hc_ht*cnot_ct*hc_ht
        return cnot_ct*cnot_tc*cnot_ct
    else
        cnot_tc = cnot(t,c,n)
        cnot_ct = hc_ht*cnot_tc*hc_ht
        return cnot_tc*cnot_ct*cnot_tc
    end
end


# Helper function to compute tensor product of n matrices
function n_fold_tensor(n::Int, mat)
    result = nothing
    if n == 0
        return g.DiagonalMat(jlg([1]))
    elseif n == 1
        return mat
    elseif n > 1
        result = mat
        for i in 1:n-1
            result = g.KroneckerProduct(result, mat)
        end
    end
    return result
end



# Define the unitary local Clifford group generator
function local_clifford_unitaries(n::Int)
    gens = []

    for i in 1:n
        push!(gens, hadamard_j(i, n))
        push!(gens, phase_j(i, n))
        if i < n
            push!(gens, swap(i, i + 1, n))
        end
    end

    gens = jlg(gens)

    return g.Group(gens)
end



####################################################################################
####################################################################################
############################### LOCAL CLIFFORD GROUP ###############################
####################################################################################
####################################################################################


mutable struct Group
    """
    Structure to store a pauli strings, bit strings, and dictionary mapping between the two 

    Attributes:
    GroupObj: GAP group object
    Order: Order of finite group
    Generators: Generators of Group
    Elements: Elements of Group
    """
    Grp::GapObj
    Order::Int64
    Gens::GapObj
    #Elements::GapObj

    function Group(G::GapObj)
        """
        Constructor for the UnitarySubGroups structure that creates Pauli and Normalizer of Pauli unitary matrix groups.
        """
        Grp = G
        Order = g.Order(G)
        Gens = g.GeneratorsOfGroup(G)
        new(Grp,Order,Gens)
    end
end




mutable struct LocalCliffordGates
    """
    Structure to store a pauli strings, bit strings, and dictionary mapping between the two 

    Attributes:
    PauliGroup: finite subgroup of unitary group generated by Pauli operators
    NormalizerGroup: finite subgroup of unitary generated by Hadamard, Phase, and CNOT
    """
    PauliGates::Group
    NormalizerGates::Group

    function LocalCliffordGates(n::Int)
        """
        Constructor for the CliffordGates structure that creates Pauli and Normalizer of Pauli unitary matrix groups.
        """
        PauliGates = Group(Pauli_group(n))
        NormalizerGates = Group(g.Normalizer(local_clifford_unitaries(n),PauliGates.Grp))
        new(PauliGates,NormalizerGates)
    end
end



################################################################################################

function local_clifford_group(CG::LocalCliffordGates)
    P = CG.PauliGates.Grp; C = CG.NormalizerGates.Grp

    hom = g.ActionHomomorphism(C,P)
    Cl = hom(C)

    return Group(Cl), hom

end


function pauli_to_action_domain_mappings(P::PauliString,G::GapObj)
    E = P.bit_strings; bit_to_int = P.bit_to_int;
    F = [f for f in g.Elements(G)]
    N = length(F); n = Int64(log(4,N)-1);

    X = Vector{Int64}([]); Y = Vector{Int64}([]);
    for a in E
        for s in [0,1]
            Ta = jlg((-1)^s)*pauli_n(a)
            x = bit_to_int[a]+s*2^(2*n); push!(X,x)
            y = findall(w->w == Ta,F)[1]; push!(Y,y)
            #z = findall(w->w == map(E1[y]),E2)[1]; 
        end
    end

    int_to_action_domain = Dict(zip(X,Y))
    action_domain_to_int = Dict(zip(Y,X))


    return int_to_action_domain, action_domain_to_int

end






mutable struct LocalCliffordGroup
    """
    Structure to store Clifford group, normal subgroup used to define action homomorphism, etc.

    Attributes:
    CliffordGroup:      Automorphisms of ActionSubgroup
    PauliGroup:         Isomorphic to ActionSubgroup
    ActionSubgroup:     Normal Subgroup of Clifford group that is isomorphic to PauliGroup
    PauliActionIso:     Isomorphism between Pauli group and normal subgroup used for action homomorphism
    GateGroupHom:       Homomorphism between normalizer of Pauli group and Clifford group
    IntActionDict:      Dictionary mapping Hermitian paulis (indexed 1..4^n/2) and action domain
    ActionIntDict:      Dictionary with reverse mapping

    """
    n::Int
    PauliInfo::PauliString
    NormalizerGates::Group
    PauliGates::Group
    LocalCliffGroup::Group
    GateGroupHom::GapObj
    IntActionDict::Dict{Int,Int}
    ActionIntDict::Dict{Int,Int}

    function LocalCliffordGroup(n::Int)
        """
        Constructor for the CliffordGroup structure
        """
        CG = LocalCliffordGates(n); PauliInfo = PauliString(n)

        NormalizerGates = CG.NormalizerGates
        PauliGates = CG.PauliGates

        LocalCliffGroup, GateGroupHom = local_clifford_group(CG)

        IntActionDict, ActionIntDict = pauli_to_action_domain_mappings(PauliInfo,PauliGates.Grp)

        new(n,PauliInfo,NormalizerGates,PauliGates,LocalCliffGroup,GateGroupHom,IntActionDict,ActionIntDict)
    end
end




####################################################################################
####################################################################################
############################## LOCAL SYMPLECTIC GROUP ##############################
####################################################################################
####################################################################################



function local_symplectic_group(n::Int,LC::Group,P::Group,hom::GapObj)
    """
    local_symplectic_group
    
    Generate the local symplectic group as a quotient of the local Clifford group by the Pauli group and find its matrix representation.
    
    # Arguments
    - `n::Int`: The number of qubits.
    - `LC::Group`: The local Clifford group.
    - `P::Group`: The Pauli group.
    - `hom::GapObj`: The homomorphism from Normalizer to Clifford group.
    
    # Returns
    - `local_spn_action::GapObj`: The local symplectic group action on the specified domain.
    """

    # Define groups:
    local_cln = LC.Grp; paulin = P.Grp; pn = hom(paulin)
    # define local symplectic group as quotient of local_cl by p:
    local_spn = g.FactorGroup(local_cln,pn)

    # The result is a polycyclic group, find matrix representation of size 2n:
    irreps = g.IrreducibleRepresentations(local_spn,g.GF(2)); elem = g.GeneratorsOfGroup(local_spn)[1];
    rep = nothing
    for irrep in irreps
        if g.Size(irrep(elem)) == 2*n
            rep = irrep
            break
        end
    end
    # image of homomorphism:
    local_spn_matrix = rep(local_spn)
    
    # Generate domain of local symplectic group action:
    En = g.ShallowCopy(g.Elements(g.GF(2)^(2*n))); g.Sort(En);

    # Define action homomorphism:
    ahom = g.ActionHomomorphism(local_spn_matrix,En)
    local_spn_action = ahom(local_spn_matrix)
    
    return local_spn_action
    
end



mutable struct LocalSymplectic
    """
    LocalSymplectic

    A mutable struct to represent the local symplectic group and its associated dictionaries for indexing.

    # Fields
    - `n::Int`: The number of qubits.
    - `Group::GapObj`: The local symplectic group as a GAP object.
    - `fdict::Dict{Vector{Int}, Int}`: A dictionary mapping vectors to their indices.
    - `bdict::Dict{Int, Vector{Int}}`: A dictionary mapping indices to their corresponding vectors.

    # Constructor
    - `LocalSymplectic(LC::LocalCliffordGroup)`: Initializes the local symplectic group using the local Clifford group information.
    """
    n::Int
    Grp::Group
    fdict::Dict{Vector{Int},Int}
    bdict::Dict{Int,Vector{Int}}

    function LocalSymplectic(LC::LocalCliffordGroup)
        # Define inputs:
        n = LC.n; H = LC.LocalCliffGroup; N = LC.PauliGates; hom = LC.GateGroupHom

        # Generate local symplectic group:
        G = local_symplectic_group(n,H,N,hom)

        Fn = sort(all_dit_strings(2,2*n));
        Dn = [i for i in 1:length(Fn)];

        fdict = Dict(zip(Fn,Dn))
        bdict = Dict(zip(Dn,Fn))

        new(n,Group(G),fdict,bdict)
    end
end






mutable struct LocalSymplecticOrbit
    """
    SympOrbit

    A mutable struct to represent the orbit of a set of vectors under the action of a symplectic permutation group.

    # Fields
    - `Canonical::Set{Vector{Int}}`: The canonical set of vectors.
    - `Orbit::Set{Set{Vector{Int}}}`: The set of orbits, each orbit being a set of vectors.

    # Constructor
    - `SympOrbit(n::Int, SP::SympPerm, Subset::Set{Vector{Int}})`: Computes the orbits of a given subset under the action of a symplectic permutation group.
    """
    Canonical::Set{Vector{Int}}
    Orbit::Set{Set{Vector{Int}}}

    function LocalSymplecticOrbit(n::Int,SP::LocalSymplectic,Subset::Set{Vector{Int}})
        Subset = [form_conversion(a,n) for a in Subset];
        G = SP.Grp.Grp; fdict = SP.fdict; bdict = SP.bdict

        # generate gap object of subset:
        DomS = g.Set(jlg([g.Set(jlg([jlg(fdict[a]) for a in Subset]))]))

        # generate orbits:
        DomOrbs = gjl(g.Orbits(G,DomS,g.OnSets)[1]); Orbs = Vector{Set{Vector{Int}}}([]);
        
        for DomOrb in DomOrbs
            Orb = [ form_conversion(bdict[d],n) for d in DomOrb];
            push!(Orbs,Set(Orb))
        end
        
        Orbit = Set(Orbs);
        Canonical = Set(Subset)

        new(Canonical,Orbit)
    end
end #added





####################################################################################
####################################################################################
################################### GROUP ORBITS ###################################
####################################################################################
####################################################################################


function grouping(C::Vector{<:Number})
    # Find non-zero coefficients and their indices
    non_zero_indices = findall(!iszero, C)
    non_zero_coefficients = abs.(C[non_zero_indices])

    # Values of non-zero coefficients
    elements = unique(non_zero_coefficients)
    sort!(elements)

    # Map coefficients to element in set and create dictionary
    cdict = Dict()
    idx = 0
    for e in elements
        if !(e in keys(cdict))
            idx += 1
            cdict[e] = idx
        end
    end

    # Create array of sets to store subsets
    subsets = Vector{Set{Int}}(undef, length(elements))
    coefficient_dict = Dict{Real,Int}()

    for (i, coeff) in enumerate(non_zero_coefficients)
        subset_index = cdict[coeff]
        if isassigned(subsets, subset_index)
            if C[non_zero_indices[i]] >= 0
                push!(subsets[subset_index], non_zero_indices[i])
            else
                push!(subsets[subset_index], non_zero_indices[i] + length(C))
            end
        else
            if C[non_zero_indices[i]] >= 0
                subsets[subset_index] = Set([non_zero_indices[i]])
            else
                subsets[subset_index] = Set([non_zero_indices[i] + length(C)])
            end
        end
        coefficient_dict[coeff] = subset_index
    end

    # Sort by increasing order of set
    sorted_subsets, sorted_elements = sortperm(elements), elements[sortperm(elements)]
    subsets = [subsets[i] for i in sorted_subsets]

    return sorted_elements, subsets, coefficient_dict
end



function subset_to_vector(N::Int,S::Vector{Vector{Int64}},dict::Dict{Real,Int})
    map = reverse_map(dict)
    V = Vector{Real}([0 for i in 1:N])
    for i in 1:length(S)
        subset = S[i]; abs_val = map[i]
        for s in subset
            if s <= N
                val = abs_val; idx = s;
            else
                val = -abs_val; idx = s-N;
            end;
            V[idx] = val;
        end
    end
    return V
end




function local_clifford_orbit_of_point(CG::LocalCliffordGroup,C::Vector)
    C = Vector{Real}(C)
    fdict = CG.IntActionDict
    bdict = CG.ActionIntDict
    G = CG.LocalCliffGroup.Grp
    N = length(C); S = g.Set(jlg([]))

    vals, V, map = grouping(C)

    for v in V
        global s = g.Set(jlg([]));
        for e in v
            g.Add(s,jlg(fdict[e])); 
        end
        s = g.Set(s); g.Add(S,s)
    end

    S = g.Set(jlg([S]))
    Orbs = g.Orbits(G,S,g.OnSetsSets)[1]

    #println(Orbs)
    
    VOrb = Vector{Vector{Real}}([])
    for orb in Orbs
        orbit = Vector{Vector{Int}}([])
        for s in orb
            subset = Vector{Int}([bdict[elem] for elem in s])
            push!(orbit,sort(subset))
        end

        vorb = subset_to_vector(N,orbit,map)
        push!(VOrb,vorb)
    end
    

    return VOrb
end




function local_clifford_gate_action_on_point(gens,CG::LocalCliffordGroup,C::Vector)
    C = Vector{Real}(C)
    fdict = CG.IntActionDict
    bdict = CG.ActionIntDict
    G = CG.CliffGroup.Grp
    N = length(C);
    K = g.Set(jlg([]))

    gg = g.GeneratorsOfGroup(G)[gens[1]];
    for i in gens[2:end]; gg = gg*g.GeneratorsOfGroup(G)[i]; end;

    vals, V, map = grouping(C)

    K = Vector{Vector{Int}}([])
    for v in V
        k = Vector{Int}([]);
        for e in v
            x = e; y = fdict[e]; gy = g.OnPoints(jlg(y),gg); gx = bdict[gy];
            #println("$x, $y, $gy, $gx")
            push!(k,gx); 
        end
        push!(K,k)
    end

    return subset_to_vector(N,K,map)
end







####################################################################################
####################################################################################
################################### CLIFFORD GROUP #################################
####################################################################################
####################################################################################




function unitary_clifford_group(n)
    gens = []

    for i in 1:n
        push!(gens, hadamard_j(i, n))
        push!(gens, phase_j(i, n))
    end

    combs = collect(combinations(1:n, 2))

    for c in combs
        push!(gens, cnot(c[1], c[2], n))
    end

    gens = jlg(gens)

    return g.Group(gens)
end



mutable struct CliffordGates
    """
    Structure to store a pauli strings, bit strings, and dictionary mapping between the two 

    Attributes:
    PauliGroup: finite subgroup of unitary group generated by Pauli operators
    NormalizerGroup: finite subgroup of unitary generated by Hadamard, Phase, and CNOT
    """
    PauliGates::Group
    NormalizerGates::Group

    function CliffordGates(n::Int)
        """
        Constructor for the CliffordGates structure that creates Pauli and Normalizer of Pauli unitary matrix groups.
        """
        PauliGates = Group(Pauli_group(n))
        NormalizerGates = Group(g.Normalizer(unitary_clifford_group(n),PauliGates.Grp))
        new(PauliGates,NormalizerGates)
    end
end




function clifford_group(CG::CliffordGates)
    P = CG.PauliGates.Grp; C = CG.NormalizerGates.Grp

    hom = g.ActionHomomorphism(C,P)
    Cl = hom(C)

    return Group(Cl), hom

end





mutable struct CliffordGroup
    """
    Structure to store Clifford group, normal subgroup used to define action homomorphism, etc.

    Attributes:
    CliffordGroup:      Automorphisms of ActionSubgroup
    PauliGroup:         Isomorphic to ActionSubgroup
    ActionSubgroup:     Normal Subgroup of Clifford group that is isomorphic to PauliGroup
    PauliActionIso:     Isomorphism between Pauli group and normal subgroup used for action homomorphism
    GateGroupHom:       Homomorphism between normalizer of Pauli group and Clifford group
    IntActionDict:      Dictionary mapping Hermitian paulis (indexed 1..4^n/2) and action domain
    ActionIntDict:      Dictionary with reverse mapping

    """
    n::Int
    PauliInfo::PauliString
    NormalizerGates::Group
    PauliGates::Group
    CliffGroup::Group
    GateGroupHom::GapObj
    IntActionDict::Dict{Int,Int}
    ActionIntDict::Dict{Int,Int}

    function CliffordGroup(n::Int)
        """
        Constructor for the CliffordGroup structure
        """
        CG = CliffordGates(n); PauliInfo = PauliString(n)

        NormalizerGates = CG.NormalizerGates
        PauliGates = CG.PauliGates

        CliffGroup, GateGroupHom = clifford_group(CG)

        IntActionDict, ActionIntDict = pauli_to_action_domain_mappings(PauliInfo,PauliGates.Grp)

        new(n,PauliInfo,NormalizerGates,PauliGates,CliffGroup,GateGroupHom,IntActionDict,ActionIntDict)
    end
end



####################################################################################





function clifford_orbit_of_point(CG::CliffordGroup,C::Vector)
    C = Vector{Real}(C)
    fdict = CG.IntActionDict
    bdict = CG.ActionIntDict
    G = CG.CliffGroup.Grp
    N = length(C); S = g.Set(jlg([]))

    vals, V, map = grouping(C)

    for v in V
        global s = g.Set(jlg([]));
        for e in v
            g.Add(s,jlg(fdict[e])); 
        end
        s = g.Set(s); g.Add(S,s)
    end

    S = g.Set(jlg([S]))
    Orbs = g.Orbits(G,S,g.OnSetsSets)[1]

    #println(Orbs)
    
    VOrb = Vector{Vector{Real}}([])
    for orb in Orbs
        orbit = Vector{Vector{Int}}([])
        for s in orb
            subset = Vector{Int}([bdict[elem] for elem in s])
            push!(orbit,sort(subset))
        end

        vorb = subset_to_vector(N,orbit,map)
        push!(VOrb,vorb)
    end
    

    return VOrb
end




function clifford_gate_action_on_point(gens,CG::CliffordGroup,C::Vector)
    C = Vector{Real}(C)
    fdict = CG.IntActionDict
    bdict = CG.ActionIntDict
    G = CG.CliffGroup.Grp
    N = length(C);
    K = g.Set(jlg([]))

    gg = g.GeneratorsOfGroup(G)[gens[1]];
    for i in gens[2:end]; gg = gg*g.GeneratorsOfGroup(G)[i]; end;

    vals, V, map = grouping(C)

    K = Vector{Vector{Int}}([])
    for v in V
        k = Vector{Int}([]);
        for e in v
            x = e; y = fdict[e]; gy = g.OnPoints(jlg(y),gg); gx = bdict[gy];
            #println("$x, $y, $gy, $gx")
            push!(k,gx); 
        end
        push!(K,k)
    end

    return subset_to_vector(N,K,map)
end





