"""
Mesh structure and methods.
"""


"""
    struct GenericMesh

Data structure representing a generic 3D mesh. 
"""
Base.@kwdef struct GenericMesh
    # Dictionary of mesh elements. Keys correspond to Element 
    # types (e.g. FlatTriangleElement) and values are vectors
    # of elements.
    etype2elements::OrderedDict{DataType, Vector{<:AbstractElement}} = 
                        OrderedDict{DataType, Vector{<:AbstractElement}}() 
end

"""
    get_etypes_and_elements(mesh::GenericMesh)

Returns an iterator that contains the element types and the elements 
of the `mesh`. Each entry is a tuple `(etype, elist)`, where `etype`
is an element type and `elist` is the list of elements associated
with that type.
"""
function get_etypes_and_elements(mesh::GenericMesh)
    return pairs(mesh.etype2elements)
end

"""
    getelements(mesh::GenericMesh)

Utility function for getting an iterator with all mesh elements.
"""
function getelements(mesh::GenericMesh)
    return Iterators.flatten(values(mesh.etype2elements))
end

"""
    getelement(mesh::GenericMesh, element_index)

Returns the `element_index`th element of the mesh.
CAUTION, this assumes that the order of the elements 
have not changed.
"""
function getelement(mesh::GenericMesh, element_index)
    @assert 1 ≤ element_index ≤ get_number_of_elements(mesh)
    # A better way of doing this?
    elementiterator = getelements(mesh)
    index = 1
    element = nothing
    for e in elementiterator
        element = e
        if index == element_index
            break
        end
        index += 1
    end
    return element
end

"""
    get_number_of_elements(mesh::GenericMesh)

Returns the total number of elements in the mesh.
"""
function get_number_of_elements(mesh::GenericMesh)
    return sum(length(v) for v in values(mesh.etype2elements))
end

"""
    get_number_of_lnodes(mesh::GenericMesh)

Returns the total number of lagrangian nodes in the mesh.
"""
function get_number_of_lnodes(mesh::GenericMesh)
    return sum(get_number_of_lnodes(el) for el in getelements(mesh))
end

"""
    compute_hmax(mesh::GenericMesh)

Returns the maximum element size `hmax` in `mesh`.
"""
function compute_hmax(mesh::GenericMesh)
    hmax = 0.0
    for element in getelements(mesh)
        hmax = max(hmax, _compute_hmax_element(element))
    end
    return hmax
end
function _compute_hmax_element(element)
    domain = getdomain(element)
    vertices = getvertices(domain)
    n_vertices = length(vertices)
    hmax = 0.0
    for i in 1:n_vertices
        v1 = element(vertices[i])
        for j in i+1:n_vertices
            v2 = element(vertices[j])
            hmax = max(hmax, norm(v1-v2))
        end
    end
    return hmax
end

"""
    find_neighboring_elements(mesh::GenericMesh)

Returns `[l₁, ..., lₙ]`, where `lᵢ` is the list of element indices that are neighbors to element `i`.
"""
function find_neighboring_elements(mesh::GenericMesh)
    n_elements = get_number_of_elements(mesh)
    n_lnodes = get_number_of_lnodes(mesh)
    # Find common-tags elements
    data = Dict{Int64, Vector{Int64}}()
    for (el_index, el) in enumerate(getelements(mesh))
        for tag in el.nodetags
            if haskey(data, tag)
                append!(data[tag], el_index)
            else
                data[tag] = [el_index]
            end
        end
    end
    # Generate neighbor list
    neighbor_list = [Int64[] for _ in 1:n_elements]
    for el_index_list in values(data)
        for i in el_index_list
            for j in el_index_list
                if i != j && j ∉ neighbor_list[i]
                    push!(neighbor_list[i], j)
                end
            end
        end
    end
    return neighbor_list
end



