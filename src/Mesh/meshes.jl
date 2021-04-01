"""
Mesh structure and methods.
"""


"""
    struct GenericMesh

Data structure representing a generic 3D mesh. 
"""
struct GenericMesh
    elements::Vector{AbstractElement}
end