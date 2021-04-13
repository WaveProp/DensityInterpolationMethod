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
    el2indices::Vector{SVector{2, Int64}} = SVector{2, Int64}[]            
end

"""
    generate_globalquadrature(mesh::GenericMesh, order)

Generates a GlobalQuadrature struct from a GenericMesh using a quadrature
rule of order `order`.
"""
function generate_globalquadrature(mesh::GenericMesh; order=2)
    n_nodes = get_number_of_lnodes(mesh)
    gquad = GlobalQuadrature{n_nodes}()

    index = 1  # for computing the element indices
    for (etype, elements) in get_etypes_and_elements(mesh)
        qrule = get_qrule_for_element(etype, order)
        index = _add_elementlist_to_gquad(gquad, elements, qrule, index)
    end
    #@assert index == n_nodes+1
    return gquad
end
function _add_elementlist_to_gquad(gquad::GlobalQuadrature, elements, qrule, index)
    for el in elements
        index = _add_element_to_gquad(gquad, el, qrule, index)
    end
    return index
end
function _add_element_to_gquad(gquad::GlobalQuadrature, el, qrule, index)
    qdata = get_quadrature_data(qrule, el)
    initial_index = index
    for (xᵢ, wᵢ, jacᵢ, nᵢ) in qdata
        gquad.nodes[index] = xᵢ
        gquad.weigths[index] = wᵢ
        gquad.jacobians[index] = jacᵢ
        gquad.normals[index] = nᵢ
        index += 1
    end
    final_index = index-1
    push!(gquad.el2indices, SVector(initial_index, final_index))
    return index
end

"""
    get_number_of_nodes(gquad::GlobalQuadrature{N})

Returns the total number of quadrature nodes of the GlobalQuadrature.
"""
function get_number_of_nodes(gquad::GlobalQuadrature{N}) where N
    return N
end

"""
    get_number_of_elements(gquad::GlobalQuadrature)

Returns the total number of elements of the GlobalQuadrature.
"""
function get_number_of_elements(gquad::GlobalQuadrature)
    return length(gquad.el2indices)
end