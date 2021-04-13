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
    etype2elements :: OrderedDict{Any, Vector{<:AbstractElement}} = 
                          OrderedDict{Any, Vector{<:AbstractElement}}()    
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
    get_number_of_elements(mesh::GenericMesh)

Returns the total number of elements in the mesh.
"""
function get_number_of_elements(mesh::GenericMesh)
    return sum(length(v) for v in values(mesh.etype2elements))
end

"""
    get_number_of_nodes(mesh::GenericMesh)

Returns the total number of quadrature nodes in the mesh.
"""
function get_number_of_nodes(mesh::GenericMesh)
    return sum(get_number_of_nodes(el) for el in getelements(mesh))
end

