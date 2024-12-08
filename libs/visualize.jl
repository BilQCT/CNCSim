using PlotlyJS
using GraphRecipes, Plots

include("cnc.jl")


function create_3dim_paulis_table(omega::Set{Vector{Int}})
    
    header_paulis = ["III", "IXI", "IYI", "IZI", "", "IIX", "IXX", "IYX", "IZX"]

    cell_paulis = [
        ["XII", "YII", "ZII", "", "IIY", "XIY", "YIY", "ZIY"],
        ["XXI", "YXI", "ZXI", "", "IXY", "XXY", "YXY", "ZXY"],
        ["XYI", "YYI", "ZYI", "", "IYY", "XYY", "YYY", "ZYY"],
        ["XZI", "YZI", "ZZI", "", "IZY", "XZY", "YZY", "ZZY"],
        ["", "", "", "", "", "", "", ""],
        ["XIX", "YIX", "ZIX", "", "IIZ", "XIZ", "YIZ", "ZIZ"],
        ["XXX", "YXX", "ZXX", "", "IXZ", "XXZ", "YXZ", "ZXZ"],
        ["XYX", "YYX", "ZYX", "", "IYZ", "XYZ", "YYZ", "ZYZ"],
        ["XZX", "YZX", "ZZX", "", "IZZ", "XZZ", "YZZ", "ZZZ"]
    ]

    
    cell_paulis_values = [["<b>" * string(pauli) * "</b>" for pauli in row]  for row in cell_paulis]

    header_paulis_values = ["<b>" * string(pauli) * "</b>" for pauli in header_paulis]

    header_colors = []
    cell_colors = []

    in_set_color = "#51A664"
    out_set_color = "#BA4C55"

    blank_color = "white"

    for pauli_string in header_paulis
        if pauli_string == ""
            push!(header_colors, blank_color)
            continue
        end
        pauli = get_pauli_from_pauli_string(pauli_string)
        if pauli in omega
            push!(header_colors, in_set_color)
        else
            push!(header_colors, out_set_color)
        end
    end

    for col in cell_paulis
        col_colors = []
        for pauli_string in col
            if pauli_string == ""
                push!(col_colors, blank_color)
                continue
            end
            pauli = get_pauli_from_pauli_string(pauli_string)
            if pauli in omega
                push!(col_colors, in_set_color)
            else
                push!(col_colors, out_set_color)
            end
        end

        push!(cell_colors, col_colors)
        
    end

    line_colors = ["darkslategray" for _ in 1:8]
    line_colors[4] = "black"

    paulis_table = table(
        header=attr(
            values=header_paulis_values,
            line_color="darkslategray",
            fill_color=header_colors,
            align=["center"],
            font=attr(color="white", size=12)
        ),
        cells=attr(
            values=cell_paulis_values,
            line_color=["darkslategray"],
            fill_color=cell_colors,
            align=["center"],
            font=attr(color="white", size=12)
        ),
        columnwidth=[3, 3, 3, 3, 1, 3, 3, 3, 3]
    )

    return paulis_table

end

function create_3dim_paulis_with_value_assignment_table(value_assignment::Dict{Vector{Int}, Int})
    
    header_paulis = ["III", "IXI", "IYI", "IZI", "", "IIX", "IXX", "IYX", "IZX"]

    cell_paulis = [
        ["XII", "YII", "ZII", "", "IIY", "XIY", "YIY", "ZIY"],
        ["XXI", "YXI", "ZXI", "", "IXY", "XXY", "YXY", "ZXY"],
        ["XYI", "YYI", "ZYI", "", "IYY", "XYY", "YYY", "ZYY"],
        ["XZI", "YZI", "ZZI", "", "IZY", "XZY", "YZY", "ZZY"],
        ["", "", "", "", "", "", "", ""],
        ["XIX", "YIX", "ZIX", "", "IIZ", "XIZ", "YIZ", "ZIZ"],
        ["XXX", "YXX", "ZXX", "", "IXZ", "XXZ", "YXZ", "ZXZ"],
        ["XYX", "YYX", "ZYX", "", "IYZ", "XYZ", "YYZ", "ZYZ"],
        ["XZX", "YZX", "ZZX", "", "IZZ", "XZZ", "YZZ", "ZZZ"]
    ]

    
    cell_paulis_values = [["<b>" * string(pauli) * "</b>" for pauli in row]  for row in cell_paulis]

    header_paulis_values = ["<b>" * string(pauli) * "</b>" for pauli in header_paulis]

    header_colors = []
    cell_colors = []

    plus_one_color = "#51A664"
    minus_one_color = "#5A90BF"
    zero_color = "#BA4C55"

    blank_color = "white"

    non_zero_paulis = Set(keys(value_assignment))

    for pauli_string in header_paulis
        if pauli_string == ""
            push!(header_colors, blank_color)
            continue
        end
        pauli = get_pauli_from_pauli_string(pauli_string)
        if pauli in non_zero_paulis
            value = value_assignment[pauli]
            if value == 1
                push!(header_colors, plus_one_color)
            elseif value == -1
                push!(header_colors, minus_one_color)
            else
                push!(header_colors, zero_color)
            end
        else
            push!(header_colors, zero_color)
        end
    end

    for col in cell_paulis
        col_colors = []
        for pauli_string in col
            if pauli_string == ""
                push!(col_colors, blank_color)
                continue
            end
            pauli = get_pauli_from_pauli_string(pauli_string)
            if pauli in non_zero_paulis
                value = value_assignment[pauli]
                if value == 1
                    push!(col_colors, plus_one_color)
                elseif value == -1
                    push!(col_colors, minus_one_color)
                else
                    push!(col_colors, zero_color)
                end
            else
                push!(col_colors, zero_color)
            end
        end

        push!(cell_colors, col_colors)
        
    end

    line_colors = ["darkslategray" for _ in 1:8]
    line_colors[4] = "black"

    paulis_table = table(
        header=attr(
            values=header_paulis_values,
            line_color="darkslategray",
            fill_color=header_colors,
            align=["center"],
            font=attr(color="white", size=12)
        ),
        cells=attr(
            values=cell_paulis_values,
            line_color=["darkslategray"],
            fill_color=cell_colors,
            align=["center"],
            font=attr(color="white", size=12)
        ),
        columnwidth=[3, 3, 3, 3, 1, 3, 3, 3, 3]
    )

    return paulis_table

end

function draw_3dim_paulis(omega::Set{Vector{Int}}, title::String="3-dimensional Pauli Operators") 
    pauli_table = create_3dim_paulis_table(omega)
    p = PlotlyJS.plot(pauli_table)
    relayout!(p, showlegend=false, title_text=title)

    return p
end

function draw_3_dim_paulis_with_value_assignment(value_assignment::Dict{Vector{Int}, Int})
    pauli_table = create_3dim_paulis_with_value_assignment_table(value_assignment)
    p = PlotlyJS.plot(pauli_table)
    relayout!(p, showlegend=false, title_text="3-dimensional Pauli Operators")

    return p
end

function find_3dim_degree_two_isotropic_type(isotropic::Set{Vector{Int}})
    identity = [0, 0, 0, 0, 0, 0]

    weight_set = []
    for pauli in isotropic
        if pauli == identity
            continue
        end
        weight = 0
        pauli_str = get_pauli_string(pauli)
        for op in pauli_str
            if op != 'I'
                weight += 1
            end
        end
        push!(weight_set, weight)
    end
    
    sort!(weight_set)
    return "(" * join(weight_set, ", ") * ")"
end

function draw_isotropic_composition_graph(composition_elements::Set{Set{Vector{Int}}}, title::String="Isotropic Composition", node_size=0.04)
    composition_elements = collect(composition_elements)
    num_elements = length(composition_elements)
    isotropic_gens_list = [find_isotropic_gens(isotropic) for isotropic in composition_elements]
    
    node_names = ["<" * join(map(get_pauli_string, collect(isotropic_gens)), ", ")* ">" for isotropic_gens in isotropic_gens_list]

    edges = [[0 for _ in 1:num_elements] for _ in 1:num_elements]
    is_no_edge = true
    for i in 1:num_elements
        isotropic = composition_elements[i]
        if length(isotropic) == 4
            node_names[i] = node_names[i] * "\n" * find_3dim_degree_two_isotropic_type(isotropic)
        end

        for j in i+1:num_elements
            isotropic2 = composition_elements[j]
            if length(intersect(isotropic, isotropic2)) > 1
                is_no_edge = false
                edges[i][j] = 1
                edges[j][i] = 1
            end
        end
    end

    if is_no_edge
        error("Graph has no edges so it is not possible to draw")
    end

    edges = hcat(edges...)'

    graphplot(edges, curves=false, names=node_names, self_edge_size=0.3, nodesize=node_size, title=title)
end