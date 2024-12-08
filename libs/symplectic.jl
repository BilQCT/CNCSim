using GAP
const g = GAP.Globals
const gjl = GAP.gap_to_julia
const jlg = GAP.GapObj

include("pauli.jl");

# convert from default GAP convention to physics convention for symplectic form:
function form_conversion(a,n)
    """
    Convert from default GAP convention to physics convention for symplectic form.
    
    # Arguments
    - `a::Vector{Int}`: Input vector in default GAP convention.
    - `n::Int`: Half the length of the input vector.
    
    # Returns
    - `Vector{Int}`: Output vector in physics convention.
    """
    b = [a[i] for i in 1:n];
    for i in 1:n; push!(b,a[2*n-i+1]);; end
    return b
end

mutable struct SympPerm
    """
    SympPerm

    A mutable struct to represent a symplectic permutation group and its associated dictionaries for indexing.

    # Fields
    - `Group::GapObj`: The symplectic group as a GAP object.
    - `fdict::Dict{Vector{Int}, Int}`: A dictionary mapping vectors to their indices.
    - `bdict::Dict{Int, Vector{Int}}`: A dictionary mapping indices to their corresponding vectors.

    # Constructor
    - `SympPerm(n::Int)`: Generates the symplectic group of degree `2n` and the associated action on a domain of length `2n`.
    """
    Group::GapObj
    fdict::Dict{Vector{Int},Int}
    bdict::Dict{Int,Vector{Int}}

    function SympPerm(n::Int)
        # Generate symplectic group:
        spn = g.SymplecticGroup(2*n,2)

        # Generate domain of symplectic group action:
        En = g.ShallowCopy(g.Elements(g.GF(2)^(2*n))); g.Sort(En);
        hom = g.ActionHomomorphism(spn,En)

        Fn = sort(all_dit_strings(2,2*n));
        Dn = [i for i in 1:length(Fn)];

        Group = hom(spn)
        fdict = Dict(zip(Fn,Dn))
        bdict = Dict(zip(Dn,Fn))

        new(Group,fdict,bdict)
    end
end



mutable struct SympOrbit
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

    function SympOrbit(n::Int,SP::SympPerm,Subset::Set{Vector{Int}})
        Subset = [form_conversion(a,n) for a in Subset];
        G = SP.Group; fdict = SP.fdict; bdict = SP.bdict

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
end

#added