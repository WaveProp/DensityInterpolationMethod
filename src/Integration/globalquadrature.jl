"""
    Global Quadrature structure and methods.
"""

"""
    const JacMatrix

Concrete type that defines a jacobian matrix 
of size 3x2 using StaticArrays.
"""
const JacMatrix = SMatrix{DIMENSION3, DIMENSION2, 
                          Float64, DIMENSION3*DIMENSION2}


"""
    struct GlobalQuadrature

Structure that contains the concrete information of the `N`-point quadrature,
i.e., quadrature nodes, weigths, normals, jacobians and elements.
"""                        
Base.@kwdef struct GlobalQuadrature{N}
    nodes::Vector{Point3D} = zeros(Point3D, N)          # List of quadrature nodes
    weigths::Vector{Float64} = zeros(Float64, N)        # List of *lifted* quadrature weigths
    jacobians::Vector{JacMatrix} = zeros(JacMatrix, N)  # List of jacobian matrices at qnodes
    normals::Vector{Point3D} = zeros(Point3D, N)        # List of normal vectors at qnodes
    
    # List of elements.
    # Each entry is an SVector `[a, b]` that correpond 
    # to the initial index and the final index for each 
    # element, inclusive.
    el2indices::Vector{SVector{2, Int64}} = zeros(SVector{2, Int64}, N)            
end

function generate_globalquadrature(mesh::GenericMesh, order)
    n_nodes = get_number_of_nodes(mesh)
    gquad = GlobalQuadrature{n_nodes}

    index = 0
    for (etype, elements) in get_etypes_and_elements(mesh)
        qrule = get_qrule_for_element(etype, order)
    end

end
function _add_elementlist_to_gquad(gquad::GlobalQuadrature, elements)
    for el in elements
    end
end
function _add_element_to_gquad(gquad::GlobalQuadrature, el, initial_index, final_index)
    


end