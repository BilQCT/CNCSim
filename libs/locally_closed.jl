using Combinatorics
using StatsBase
using LinearAlgebra

include("cnc.jl"); include("pauli.jl")

mutable struct AlmostMaximalTwoDimSet
    n::Int
    m::Int
    chosen_locals::Vector{Vector{Int}}
    independent_nonlocals::Vector{Vector{Int}}

    function AlmostMaximalTwoDimSet(n::Int, m::Int, chosen_locals::Vector{Vector{Int}}, independent_nonlocals::Vector{Vector{Int}})
        new(n, m, chosen_locals, independent_nonlocals)
    end
end

mutable struct AlmostMaximalTwoDim
    almost_maximal_two_dim_set::AlmostMaximalTwoDimSet
    value_assignment::Dict{Vector{Int},Int}

    function AlmostMaximalTwoDim(almost_maximal_two_dim_set::AlmostMaximalTwoDimSet, value_assignment::Dict{Vector{Int},Int})
        new(almost_maximal_two_dim_set, value_assignment)
    end
end

function generate_random_two_dim_almost_maximal_sets(n::Int, m::Int)::AlmostMaximalTwoDimSet
    """
    Generates a random almost maximal set of pauli operators in two dimensions with n local operators in Alice and 
    m local operators in Bob.

    Args:
        n: The number of local operators in Alice.
        m: The number of local operators in Bob.

    Returns:
        An almost maximal paulis set.
    """

    XI = [1, 0, 0, 0]
    YI = [1, 0, 1, 0]
    ZI = [0, 0, 1, 0]
    IX = [0, 1, 0, 0]
    IY = [0, 1, 0, 1]
    IZ = [0, 0, 0, 1]

    all_locals = [XI, YI, ZI, IX, IY, IZ]

    alice_locals = [XI, YI, ZI]
    bob_locals = [IX, IY, IZ]

    random_local_alice = sample(alice_locals, n, replace=false)
    random_local_bob = sample(bob_locals, m, replace=false)

    chosen_locals = vcat(random_local_alice, random_local_bob)

    other_locals = []
    for i in 1:6
        if !(all_locals[i] in chosen_locals)
            push!(other_locals, all_locals[i])
        end
    end

    independent_nonlocals::Vector{Vector{Int}} = []

    for used_locals in combinations(other_locals, 2)
        if do_commute(used_locals[1], used_locals[2])
            push!(independent_nonlocals, used_locals[1] + used_locals[2])
        end
    end

    return AlmostMaximalTwoDimSet(n, m, chosen_locals, independent_nonlocals)

end


function generate_all_two_dim_almost_maximals(almost_maximal::AlmostMaximalTwoDimSet)::Set{AlmostMaximalTwoDim}

    indep_paulis = vcat(almost_maximal.independent_nonlocals, almost_maximal.chosen_locals)
    num_indep = length(indep_paulis)

    almost_maximals = Set{AlmostMaximalTwoDim}()

    for i in 0:(2^num_indep-1)
        identity = [0, 0, 0, 0]

        # Initialize the value assignment by assigning the identity to 1
        value_assignment = Dict{Vector{Int},Int}(identity => 1)

        # Convert the integer to a binary string of length num_indep
        bitstring = string(i, base=2, pad=num_indep)
        value_array = [parse(Int, c) for c in bitstring]

        # Assign the values specified by binary array to the independent Paulis
        for (p, v) in zip(indep_paulis, value_array)
            value_assignment[p] = (-1)^v
        end

        for used_locals in combinations(almost_maximal.chosen_locals, 2)
            if do_commute(used_locals[1], used_locals[2])
                pauli = (used_locals[1] + used_locals[2]) .% 2
                value_assignment[pauli] = value_assignment[used_locals[1]] * value_assignment[used_locals[2]]
            end
        end

        push!(almost_maximals, AlmostMaximalTwoDim(almost_maximal, value_assignment))
    end

    return almost_maximals
end


function almost_maximal_to_pauli_basis(almost_maximal::AlmostMaximalTwoDim, ps::PauliString)
    bit_strings = ps.bit_strings
    N = length(bit_strings); V = []

    gamma = almost_maximal.value_assignment
    gamma_keys = collect(keys(gamma))

    for x in bit_strings
        if x in gamma_keys; push!(V,gamma[x]); else push!(V,0); end;
    end

    return V
end

function generate_isotropic_from_gens(gens::Vector{Vector{Int}})::Set{Vector{Int}}
    """
    Generates the isotropic set from the generators.

    Args:
        gens: The generators of the isotropic set.

    Returns:
        The isotropic set.
    """

    n = length(gens[1]) / 2

    identity = [0 for i in 1:2n]
    isotropic = Set{Vector{Int}}([identity])

    for i in 1:length(gens)
        for used_gens in combinations(gens, i)
            pauli = identity
            for used_gen in used_gens
                pauli = (pauli + used_gen) .% 2
            end
            push!(isotropic, pauli)
        end
    end

    return isotropic
end

function generate_all_three_dim_maximal_local_isotropics()::Set{Set{Vector{Int}}}
    III = [0, 0, 0, 0, 0, 0]

    XII = [1, 0, 0, 0, 0, 0]
    YII = [1, 0, 0, 1, 0, 0]
    ZII = [0, 0, 0, 1, 0, 0]

    alice_locals = [XII, YII, ZII]

    IXI = [0, 1, 0, 0, 0, 0]
    IYI = [0, 1, 0, 0, 1, 0]
    IZI = [0, 0, 0, 0, 1, 0]

    bob_locals = [IXI, IYI, IZI]

    IIX = [0, 0, 1, 0, 0, 0]
    IIY = [0, 0, 1, 0, 0, 1]
    IIZ = [0, 0, 0, 0, 0, 1]

    charlie_locals = [IIX, IIY, IIZ]

    all_maximal_local_isotropics = Set{Set{Vector{Int}}}()

    for alice_local in alice_locals
        for bob_local in bob_locals
            for charlie_local in charlie_locals
                gens = Set{Vector{Int}}([alice_local, bob_local, charlie_local])
                isotropic = generate_isotropic_from_gens(collect(gens))
                push!(all_maximal_local_isotropics, isotropic)
            end
        end
    end

    return all_maximal_local_isotropics
end

function do_locally_commute(pauli1::Vector{Int}, pauli2::Vector{Int})::Bool
    
    if length(pauli1) != length(pauli2)
        return false
    end

    pauli_str_1 = get_pauli_string(pauli1)
    pauli_str_2 = get_pauli_string(pauli2)


    n = length(pauli_str_1)

    for i in 1:n
        if pauli_str_1[i] != pauli_str_2[i]
            if pauli_str_1[i] != 'I' && pauli_str_2[i] != 'I'
                return false
            end
        end
    end

    return true
end

function find_local_closure(omega::Set{Vector{Int}})
    omega = copy(omega)

    identity = [0 for i in 1:length(first(omega))]
    push!(omega, identity)

    is_closed = false

    while !is_closed
        is_closed = true
        for paulis in combinations(collect(omega), 2)
            pauli1 = paulis[1]
            pauli2 = paulis[2]
            if do_locally_commute(pauli1, pauli2) && !((pauli1 + pauli2) .%2 in omega)
                is_closed = false
                push!(omega, (pauli1 + pauli2) .% 2)
            end
        end
    end

    return omega
end

function find_local_extension(value_assignment::Dict{Vector{Int},Int}, new_Pauli::Vector{Int}, value::Int)
    new_value_assignment = copy(value_assignment)
    new_value_assignment[new_Pauli] = value

    is_closed = false

    while !is_closed
        is_closed = true
        for paulis in combinations(collect(keys(new_value_assignment)), 2)
            pauli1 = paulis[1]
            pauli2 = paulis[2]
            if do_locally_commute(pauli1, pauli2) && !((pauli1 + pauli2) .%2 in collect(keys(new_value_assignment)))
                inferred_pauli = (pauli1 + pauli2) .% 2
                is_closed = false
                new_value_assignment[inferred_pauli] = new_value_assignment[pauli1] * new_value_assignment[pauli2]
            end
        end
    end

    locally_value_assignment = true
    for paulis in combinations(collect(keys(new_value_assignment)), 2)
        pauli1 = paulis[1]
        pauli2 = paulis[2]
        if do_locally_commute(pauli1, pauli2)
            pauli = (pauli1 + pauli2) .% 2
            if new_value_assignment[pauli] != new_value_assignment[pauli1] * new_value_assignment[pauli2]
                locally_value_assignment = false
                break
            end
        end
    end

    if !locally_value_assignment
        return false
    end

    return new_value_assignment

end

function is_locally_closed(omega::Set{Vector{Int}})
    if omega == find_local_closure(omega)
        return true
    end

    return false
end

function is_almost_maximal(omega::Set{Vector{Int}})
    l = length(first(omega))
    is_almost_maximal = true

    for i in 0:(2^l - 1)
        bitstring = string(i, base=2, pad=l)
        pauli = [parse(Int, c) for c in bitstring]

        if !(pauli in omega)
            new_omega = copy(omega)
            push!(new_omega, pauli)
            if length(find_local_closure(new_omega)) != 2^l
                println(get_pauli_string(pauli), " is added and closure is not maximal")
                return false
            end
        end
    end

    return true
end

function is_local(pauli::Vector{Int})
    pauli_str = get_pauli_string(pauli)

    identity_count = 0

    for c in pauli_str
        if c == 'I'
            identity_count += 1
        end
    end

    if identity_count < length(pauli_str) - 1
        return false
    end

    return true
end

function find_independent_paulis(omega::Set{Vector{Int}})
    if length(omega) == 0
        return []
    end

    omega = copy(omega)
    n = length(first(omega)) / 2
    identity = [0 for i in 1:2n]

    omega_locals = [pauli for pauli in omega if is_local(pauli)]
    indep_paulis = Set{Vector{Int}}(omega_locals)

    inferred_paulis = find_local_closure(indep_paulis)

    for pauli in inferred_paulis
        delete!(omega, pauli)
    end

    delete!(indep_paulis, identity)

    while length(omega) > 0
        random_indep_pauli = sample(collect(omega), 1)[1]

        push!(indep_paulis, random_indep_pauli)

        closed_inferred_paulis = copy(inferred_paulis)

        push!(closed_inferred_paulis, random_indep_pauli)

        closed_inferred_paulis = find_local_closure(closed_inferred_paulis)

        newly_inferred_paulis = Set{Vector{Int}}()

        for pauli in closed_inferred_paulis
            if !(pauli in inferred_paulis)
                push!(newly_inferred_paulis, pauli)
            end
        end

        for pauli in newly_inferred_paulis
            delete!(omega, pauli)
        end

        inferred_paulis = closed_inferred_paulis
    end

    return indep_paulis
end


function find_all_possible_local_value_assignments(omega::Set{Vector{Int}}, max_num_assignments::Int=-1)
    omega = copy(omega)
    indep_paulis = find_independent_paulis(omega)
    
    identity = [0 for i in 1:length(first(omega))]
    num_indep = length(indep_paulis)

    all_value_assignments = Set{Dict{Vector{Int},Int}}()

    num_repetition = 2^num_indep - 1
    
    if max_num_assignments != -1
        num_repetition = max_num_assignments
    end

    for i in 0:num_repetition

        # Initialize the value assignment by assigning the identity to 1
        value_assignment = Dict{Vector{Int},Int}(identity => 1)

        # Convert the integer to a binary string of length l
        bitstring = string(i, base=2, pad=num_indep)
        value_array = [parse(Int, c) for c in bitstring]

        if max_num_assignments != -1 
            value_array = rand(0:1, num_indep)
        end

        # Assign the values specified by binary array to the independent Paulis
        for (p, v) in zip(collect(indep_paulis), value_array)
            value_assignment[p] = (-1)^v
        end

        while length(value_assignment) < length(omega)
            for paulis in combinations(collect(keys(value_assignment)), 2)
                pauli1 = paulis[1]
                pauli2 = paulis[2]

                if do_locally_commute(pauli1, pauli2)
                    pauli = (pauli1 + pauli2) .% 2
                    value_assignment[pauli] = value_assignment[pauli1] * value_assignment[pauli2]
                    if length(value_assignment) == length(omega)
                        break
                    end
                end
            end
        end


        locally_value_assignment = true
        for paulis in combinations(collect(omega), 2)
            pauli1 = paulis[1]
            pauli2 = paulis[2]
            if do_locally_commute(pauli1, pauli2)
                pauli = (pauli1 + pauli2) .% 2
                if value_assignment[pauli] != value_assignment[pauli1] * value_assignment[pauli2]
                    locally_value_assignment = false
                    break
                end
            end
        end

        if locally_value_assignment
            push!(all_value_assignments, value_assignment)
        end
    end

    return all_value_assignments
end

function value_assignment_to_pauli_basis(value_assignment::Dict{Vector{Int},Int}, n::Int)
    ps = PauliString(n)
    bit_strings = ps.bit_strings
    V = []

    gamma_keys = collect(keys(value_assignment))

    for x in bit_strings
        if x in gamma_keys; push!(V,value_assignment[x]); else push!(V,0); end;
    end

    return V
end

function find_rank_count_for_given_given_value_assignment(value_assignment::Dict{Vector{Int},Int}, n::Int, stab_coeffs)
    A = value_assignment_to_pauli_basis(value_assignment, n)
    H = stab_coeffs * A; println(H)
    Z = findall(x->x==0,H)
    AZ = stab_coeffs[Z,:]
    rank_AZ = rank(AZ)
    return rank_AZ
end

function find_ranks_count_for_given_set_of_value_assignments(value_assignments::Set{Dict{Vector{Int},Int}}, n::Int, stab_coeffs)
    
    # initialize matrix:
    M = Array{Int}(undef, 4^n, 0)

    stab_coeffs = copy(stab_coeffs)

    for value_assignment in value_assignments
        A = value_assignment_to_pauli_basis(value_assignment, n)
        M = hcat(M, A)
    end

    rank_counts = Dict{Int, Int}()
    H = stab_coeffs * M

    for j in 1:size(H)[2]
        Z = findall(x->x==0,H[:,j])
        AZ = stab_coeffs[Z,:]
        rank_AZ = rank(AZ)
        if haskey(rank_counts, rank_AZ)
            rank_counts[rank_AZ] += 1
        else
            rank_counts[rank_AZ] = 1
        end
    end

    return rank_counts
end

function find_random_independent_paulis(n::Int, num_indep::Int, num_examples::Int)
    independent_pauli_examples = Set{Set{Vector{Int}}}()
    identity = [0 for i in 1:2n]
    for i in 1:num_examples
        indep_paulis = Set{Vector{Int}}()
        inferred_paulis = Set{Vector{Int}}([identity])
        for j in 1:num_indep
            indep_pauli_found = false
            while !indep_pauli_found && length(inferred_paulis) < 4^n
                random_pauli = rand(0:1, 2n)
                if !(random_pauli in inferred_paulis)
                    push!(indep_paulis, random_pauli)
                    push!(inferred_paulis, random_pauli)
                    inferred_paulis = find_local_closure(inferred_paulis)
                    indep_pauli_found = true
                end
            end
        end
        push!(independent_pauli_examples, indep_paulis)
    end

    return independent_pauli_examples
end

function is_value_assignment_local(value_assignment::Dict{Vector{Int},Int})
    for paulis in combinations(collect(keys(value_assignment)), 2)
        pauli1 = paulis[1]
        pauli2 = paulis[2]
        if do_locally_commute(pauli1, pauli2)
            pauli = (pauli1 + pauli2) .% 2
            if value_assignment[pauli] != value_assignment[pauli1] * value_assignment[pauli2]
                return false
            end
        end
    end

    return true
end

function find_full_rank_maximal_cnc_examples_for_n_m(n::Int, m::Int, num_examples::Int)
    symptuple_n = symplectic_perm_group(n)
    sp_n = symptuple_n[1] 
    fdict_n = symptuple_n[2]
    bdict_n = symptuple_n[3]

    i_n = canonical_maximal_isotropic(n)
    I_n = (symplectic_orbit(n, sp_n, i_n, fdict_n, bdict_n))

    I_n_loc = local_isotropics(I_n, n)
    A_n_loc = stabilizer_coefficients(n, I_n_loc)

    cnc = get_full_cnc_set(n, m)
    orb = symplectic_orbit(n, sp_n, cnc, fdict_n, bdict_n)

    full_rank_examples = []

    for o in orb
        cnc = MaximalCncSet(Set(o))
        all_cnc = generate_all_cncs_for_given_cnc_set(cnc)
        value_assignments = Set{Dict{Vector{Int},Int}}()
        for max_cnc in all_cnc
            if !is_value_assignment_local(max_cnc.value_assignment)
                error("Value assignment is not local")
            end
            push!(value_assignments, max_cnc.value_assignment)
        end
        ranks = find_ranks_count_for_given_set_of_value_assignments(value_assignments, n, A_n_loc)
        if haskey(ranks, 4^n - 1)
            push!(full_rank_examples, Set(o))
        end
    end

    full_rank_examples = sample(full_rank_examples, num_examples, replace=false)

    return full_rank_examples
end

function find_all_3dim_maximal_isotropics_in_given_set(omega::Set{Vector{Int}})
    all_3dim_maximal_local_isotropics = generate_all_three_dim_maximal_local_isotropics()

    maximal_isotropics = Set{Set{Vector{Int}}}()
    for maximal_isotropic in all_3dim_maximal_local_isotropics
        if maximal_isotropic ⊆ omega
            push!(maximal_isotropics, maximal_isotropic)
        end
    end

    return maximal_isotropics
end

function find_all_3dim_degree_two_local_isotropics_in_given_set(omega::Set{Vector{Int}})
    omega = copy(omega)
    identity = [0, 0, 0, 0, 0, 0]
    delete!(omega, identity)
    degree_two_isotropics = Set{Set{Vector{Int}}}()

    for used_paulis in combinations(collect(omega), 3)
        is_locally_commuting = true
        for pair in combinations(used_paulis, 2)
            if !do_locally_commute(pair[1], pair[2])
                is_locally_commuting = false
                break
            end
        end

        if !is_locally_commuting
            continue
        end

        isotropic = Set{Vector{Int}}(used_paulis)
        push!(isotropic, identity)
        isotropic = find_local_closure(isotropic)
        if length(isotropic) == 4
            push!(degree_two_isotropics, isotropic)
        end
    end

    return degree_two_isotropics
end

function find_local_isotropic_composition(omega::Set{Vector{Int}})::Set{Set{Vector{Int}}}

    composition_elements = Set{Set{Vector{Int}}}()

    maximal_local_isotropics = find_all_3dim_maximal_isotropics_in_given_set(omega)
    degree_two_local_isotropics = find_all_3dim_degree_two_local_isotropics_in_given_set(omega)

    for degree_two_isotropic in degree_two_local_isotropics
        for maximal_isotropic in maximal_local_isotropics
            if degree_two_isotropic ⊆ maximal_isotropic
                delete!(degree_two_local_isotropics, degree_two_isotropic)
            end
        end
    end

    composition_elements = maximal_local_isotropics ∪ degree_two_local_isotropics


    return composition_elements
end

function get_pauli_weight(pauli::Vector{Int})
    pauli_str = get_pauli_string(pauli)
    weight = 0
    for c in pauli_str
        if c != 'I'
            weight += 1
        end
    end

    return weight
end

function get_fillable_weight_3s(omega::Set{Vector{Int}})::Vector{Vector{Int}}
    weight_3s = Vector{Vector{Int}}()
    all_maximal_isotropics = generate_all_three_dim_maximal_local_isotropics()
    for max_isotropic in all_maximal_isotropics
        if length(intersect(omega, max_isotropic)) == 1
            for pauli in max_isotropic
                if get_pauli_weight(pauli) == 3
                    push!(weight_3s, pauli)
                end
            end
        end
    end

    return weight_3s
end


function CZ_action(pauli::Vector{Int}, qubits::Vector{Int})
    n = length(pauli) ÷ 2
    pauli_str = get_pauli_string(pauli)
    qubits = sort(copy(qubits))

    II = [0, 0, 0, 0]
    XI = [1, 0, 0, 0]
    YI = [1, 0, 1, 0]
    ZI = [0, 0, 1, 0]
    IX = [0, 1, 0, 0]
    IY = [0, 1, 0, 1]
    IZ = [0, 0, 0, 1]

    XZ = (XI + IZ) .% 2
    YZ = (YI + IZ) .% 2
    ZX = (ZI + IX) .% 2
    ZY = (ZI + IY) .% 2

    sign = 1
    resulting_paulis = II
    if pauli_str[qubits[1]] == 'X'
        if pauli_str[qubits[2]] == 'Y'
            sign = -1
        end
        resulting_paulis = (resulting_paulis + XZ).% 2
    elseif pauli_str[qubits[1]] == 'Y'
        if pauli_str[qubits[2]] == 'X'
            sign = -1
        end
        resulting_paulis = (resulting_paulis + YZ).% 2
    elseif pauli_str[qubits[1]] == 'Z'
        resulting_paulis = (resulting_paulis + ZI).% 2
    end

    if pauli_str[qubits[2]] == 'X'
        resulting_paulis = (resulting_paulis + ZX).% 2
    elseif pauli_str[qubits[2]] == 'Y'
        resulting_paulis = (resulting_paulis + ZY).% 2
    elseif pauli_str[qubits[2]] == 'Z'
        resulting_paulis = (resulting_paulis + IZ).% 2
    end

    resulting_paulis_str = get_pauli_string(resulting_paulis)

    pauli_arr = collect(pauli_str)
    pauli_arr[qubits[1]] = resulting_paulis_str[1]
    pauli_arr[qubits[2]] = resulting_paulis_str[2]
    pauli_str = join(pauli_arr)

    return get_pauli_from_pauli_string(pauli_str), sign
end


function find_local_cnc_generators(omega::Set{Vector{Int}})::Vector{Vector{Int}}
    if length(omega) == 0
        return Set{Vector{Int}}()
    end

    omega = copy(omega)

    n = length(first(omega)) / 2
    identity = [0 for i in 1:2n]

    weighted_paulis = Dict{Int, Set{Vector{Int}}}()
    for pauli in omega
        weight = get_pauli_weight(pauli)
        if haskey(weighted_paulis, weight)
            push!(weighted_paulis[weight], pauli)
        else
            weighted_paulis[weight] = Set{Vector{Int}}([pauli])
        end
    end
    
    generators = Vector{Vector{Int}}()
    inferred_paulis = Set{Vector{Int}}()

    cur_weight = 1

    while cur_weight <= n
        if !haskey(weighted_paulis, cur_weight) || length(weighted_paulis[cur_weight]) == 0
            cur_weight += 1
            continue
        end

        random_indep_pauli = sample(collect(weighted_paulis[cur_weight]), 1)[1]

        push!(generators, random_indep_pauli)

        closed_inferred_paulis = copy(inferred_paulis)

        push!(closed_inferred_paulis, random_indep_pauli)

        closed_inferred_paulis = find_local_closure(closed_inferred_paulis)

        for pauli in closed_inferred_paulis
            if !(pauli in inferred_paulis)
                weight = get_pauli_weight(pauli)
                delete!(weighted_paulis[weight], pauli)
            end
        end

        inferred_paulis = closed_inferred_paulis
    end

    return generators
end


function calculate_point_operator_basis(value_assignment::Dict{Vector{Int},Int}, n::Int)
    ps = PauliString(n)

    point_operator_coeffs = Vector{Int}()
    for u in ps.bit_strings
        coeff = 0
        for (pauli, value) in value_assignment
            coeff += (-1)^(omega(u, pauli)) * value
        end
        push!(point_operator_coeffs, coeff)
    end

    return point_operator_coeffs

end