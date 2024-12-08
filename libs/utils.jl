function reverse_map(map)
    in_key = collect(keys(map)); out_value = collect(values(map))
    return Dict(zip(out_value,in_key))
end


function composition(map1,map2)
    Yin = Set(values(map1)); Yout = Set(keys(map2))

    if isempty(intersect!(Yin,Yout)) == true
        error("Codomain of first map have elements in domain of second.")
    else
        Y = intersect!(Set(values(map1)), Set(keys(map2)))
        in_key = [reverse_map(map1)[y] for y in Y]
        out_value = [map2[y] for y in Y]
        return Dict(zip(in_key,out_value))
    end
end



"""
Generate all dit strings recursively.

Parameters:
- `d`: The number of outcomes.
- `N`: The number of generators.
- `n`: The current iteration.
- `s`: The current set of dit strings.

Returns:
- All dit strings.
"""
function generate_dit_strings(d, N, n, s)
    if n < N
        Zd = [(i - 1) for i in 1:d]
        dit_strings = []
        for i in 1:length(s)
            for x in Zd
                push!(dit_strings, push!(copy(s[i]), x))
            end
        end
        s = dit_strings
        n = n + 1
        generate_dit_strings(d, N, n, s)
    else
        return s
    end
end


"""
Generate all possible dit strings.

Parameters:
- `d`: The number of outcomes.
- `N`: The number of generators.

Returns:
- All dit strings.
"""
function all_dit_strings(d, N)
    s = [[i - 1] for i in 1:d]
    n = 1
    return generate_dit_strings(d, N, n, s)
end


using LinearAlgebra

function rank_of_point(V::Vector{Number},M::Matrix{Number})
    """
    V: Vector: (dx1)
    M: Matrix: (mxd)
    """
    Z = findall(x->x==-1,M*V)
    return rank(M[Z,:])
end



using DelimitedFiles

function pm_to_panda(A)
    M = copy(A[:,2:end]);
    M = hcat(M,A[:,1]);
    return -M
end



function save_input_panda(M,filename,delimiter,input)
    if input == "H"
        input = "Inequalities \n"
    elseif input == "V"
        input == "Vertices \n"
    else
        error("Not valid.")
    end
    open(filename, "w") do io
        write(io,input)
           writedlm(io, M, delimiter)
       end
end


# example usage: save_input_panda(M2,"L2","  ","H")




function save_input_lrs(M,filename,name,comment,delimiter,input)
    if input == "H"
        input = "H-representation \n"
    elseif input == "V"
        input == "V-representation \n"
    else
        error("Not valid.")
    end
    
    # size of input:
    m = size(M)[1]; n = size(M)[2];
    
    strg = name*"\n"*"* "*comment*"\n"*input*"begin \n"*string(m)*" "*string(n)*" rational \n";
    
    open(filename, "w") do io
        write(io,strg);
        writedlm(io, M, delimiter);
        write(io,"end");
    end
end

function array_to_matrix(A)
    n = length(A[1]); N = length(A);
    R = Array{Rational{Int64}}(undef,n,0)
    for i in 1:N
        R = hcat(R,A[i]);
    end
    return R
end

# example usage: save_input_lrs(A2,"L2.ine","Lambda2","Lambda2"," ","H")