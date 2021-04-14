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

Structure that contains the concrete information of the quadrature,
i.e., quadrature nodes, weigths, normals, jacobians and elements.
"""                        
struct GlobalQuadrature
    nodes::Vector{Point3D}          # List of *lifted* quadrature nodes
    weigths::Vector{Float64}        # List of *lifted* quadrature weigths
    jacobians::Vector{JacMatrix}    # List of jacobian matrices at qnodes
    normals::Vector{Point3D}        # List of normal vectors at qnodes
    
    # List of elements.
    # Each element correspond to a vector `[i₁, i₂, ..., iₙ]` which 
    # contains the indices of the `n` qnodes of the element.
    el2indices::Vector{Vector{Int64}}     
    
    # Mapping from qnode indices to element indices.
    index2element::Vector{Int64}

    # Constructor
    function GlobalQuadrature(n_qnodes, n_elements)
        nodes = Vector{Point3D}(undef, n_qnodes)
        weigths = Vector{Float64}(undef, n_qnodes)
        jacobians = Vector{JacMatrix}(undef, n_qnodes)
        normals = Vector{Point3D}(undef, n_qnodes)
        el2indices = Vector{Vector{Int64}}(undef, n_elements)
        index2element = Vector{Int64}(undef, n_qnodes)
        return new(nodes, weigths, jacobians, 
                   normals, el2indices, index2element)
    end
end

"""
    generate_globalquadrature(mesh::GenericMesh, order)

Generates a GlobalQuadrature struct from a GenericMesh using a quadrature
rule of order `order`.
"""
function generate_globalquadrature(mesh::GenericMesh; order=2)
    n_qnodes = get_number_of_qnodes(mesh, order)
    n_elements = get_number_of_elements(mesh)
    gquad = GlobalQuadrature(n_qnodes, n_elements)

    index = 1  # for computing the element indices
    for (etype, elements) in get_etypes_and_elements(mesh)
        qrule = get_qrule_for_element(etype, order)
        index = _add_elementlist_to_gquad(gquad, elements, qrule, index)
    end
    @assert index == n_qnodes+1
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

    # Get element index
    if index > 1
        element_index = gquad.index2element[index-1]+1
    else
        element_index = 1
    end

    # Save data in GlobalQuadrature
    for (xᵢ, wᵢ, jacᵢ, nᵢ) in qdata
        gquad.nodes[index] = xᵢ
        gquad.weigths[index] = wᵢ
        gquad.jacobians[index] = jacᵢ
        gquad.normals[index] = nᵢ
        gquad.index2element[index] = element_index
        index += 1
    end
    final_index = index-1

    # Save element data
    gquad.el2indices[element_index] = collect(initial_index:final_index)
    return index
end

"""
    get_number_of_qnodes(gquad::GlobalQuadrature)

Returns the total number of quadrature nodes of the GlobalQuadrature.
"""
function get_number_of_qnodes(gquad::GlobalQuadrature)
    return length(gquad.nodes)
end

"""
    get_number_of_elements(gquad::GlobalQuadrature)

Returns the total number of elements of the GlobalQuadrature.
"""
function get_number_of_elements(gquad::GlobalQuadrature)
    return length(gquad.el2indices)
end