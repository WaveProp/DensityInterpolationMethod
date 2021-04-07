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
    etype2elements :: OrderedDict{DataType, Vector{<:AbstractElement}} = 
                          OrderedDict{DataType, Vector{<:AbstractElement}}()    
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

