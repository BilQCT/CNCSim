
########################################################
# Generate group objects:
using Polymake
const pm = Polymake;
using GAP

# Polymake uses 0-based indexing, but GAP and Julia use 1-based indexing:
to_one_based_indexing(n::Number) = n + one(n)
to_zero_based_indexing(n::Number) = (n > zero(n) ? n - one(n) : throw(ArgumentError("Can't use negative index")))

# Not sure what this does (yet)
for f in [:to_one_based_indexing, :to_zero_based_indexing]
    @eval begin
        $f(itr) = $f.(itr)
        $f(s::S) where S<:AbstractSet = S($f.(s))
    end
end

#input: polymake polytope object P
#output: GAP group
function combinatorial_automorphism_group(P)
    G = pm.group.automorphism_group(P.VERTICES_IN_FACETS)
    gens_polymake = G.PERMUTATION_ACTION.GENERATORS # acting on the facets
    gens_julia = Vector{Int64}.(pm.to_one_based_indexing(gens_polymake))
    gens_gap = GAP.Globals.PermList.(GAP.julia_to_gap.(gens_julia))
    return GAP.Globals.Group(gens_gap...)
end


# input: polymake polytope:
# output: GAP group for action on VERTICES:
function vertex_action_GAP(P)
    # embed group attribute in P:
    pm.polytope.combinatorial_symmetries(P);
    pm.give(P, "GROUP");
    # permutation group acting on vertices:
    gens_poly = P.GROUP.VERTICES_ACTION.GENERATORS;
    # convert to julia array (1-based indexing):
    gens_julia = Vector{Int64}.(pm.to_one_based_indexing(gens_poly));
    # create GAP generators:
    gens_gap = GAP.Globals.PermList.(GAP.julia_to_gap.(gens_julia));
    return GAP.Globals.Group(gens_gap...)
end

# input: polymake polytope:
# output: GAP group for action on FACETS:
function facets_action_GAP(P)
    # embed group attribute in P:
    pm.polytope.combinatorial_symmetries(P);
    pm.give(P, "GROUP");
    # permutation group acting on vertices:
    gens_poly = P.GROUP.FACETS_ACTION.GENERATORS;
    # convert to julia array (1-based indexing):
    gens_julia = Vector{Int64}.(pm.to_one_based_indexing(gens_poly));
    # create GAP generators:
    gens_gap = GAP.Globals.PermList.(GAP.julia_to_gap.(gens_julia));
    return GAP.Globals.Group(gens_gap...)
end

#input: polymake polytope P:
#output: array of arrays: each element is index of orbit under action of Aut(P):
function vertex_orbit(P)
    # create GAP group:
    G = vertex_action_GAP(P)
    # use polymake to generate representative vertices of P:
    rep = representative_vertices(P);
    # find vertex identifiers: representative vertices:
    rep_idx = [find_vertex(rep[i,:],P.VERTICES) for i in 1:size(rep)[1]];
    # create array of arrays: each array is an orbit:
    orbit_arry = [Vector{Int64}(GAP.Globals.Orbit(G,i)) for i in rep_idx];

    return orbit_arry
end

#input: polymake polytope P:
#output: array of arrays: each element is index of orbit under action of Aut(P):
function facets_orbit(P)
    # create GAP group:
    G = facets_action_GAP(P)
    # use polymake to generate representative vertices of P:
    rep = representative_facets(P);
    # find vertex identifiers: representative vertices:
    rep_idx = [find_vertex(rep[i,:],P.FACETS) for i in 1:size(rep)[1]];
    # create array of arrays: each array is an orbit:
    orbit_arry = [Vector{Int64}(GAP.Globals.Orbit(G,i)) for i in rep_idx];

    return orbit_arry
end

#INPUT: Polymake polytope object:
#OUTPUT: representative vertices
function representative_vertices(polytope)
    pm.polytope.combinatorial_symmetries(polytope)
    pm.give(polytope, "GROUP.REPRESENTATIVE_VERTICES");
    return polytope.GROUP.REPRESENTATIVE_VERTICES
end

#INPUT: Polymake polytope object:
#OUTPUT: representative facets
function representative_facets(polytope)
    pm.polytope.combinatorial_symmetries(polytope)
    pm.give(polytope, "GROUP.REPRESENTATIVE_FACETS");
    return polytope.GROUP.REPRESENTATIVE_FACETS
end