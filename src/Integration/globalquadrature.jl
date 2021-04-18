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
    # Get element index
    if index > 1
        element_index = gquad.index2element[index-1]+1
    else
        element_index = 1
    end

    # Save data in GlobalQuadrature
    qdata = get_quadrature_data(qrule, el)
    initial_index = index
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

"""
    get_inelement_qnode_indices(gquad::GlobalQuadrature, qnode_index)

Given a `qnode_index`, returns the indices of the qnodes that lie on the same
element.
"""
function get_inelement_qnode_indices(gquad::GlobalQuadrature, qnode_index)
    element_index = gquad.index2element[qnode_index]
    inelement_qnode_indices = gquad.el2indices[element_index]
    return inelement_qnode_indices
end

"""
    get_outelement_qnode_indices(gquad::GlobalQuadrature, qnode_index)

Given a `qnode_index`, returns an iterator with the indices of the qnodes 
that do not lie on the same element. This assumes that the qnodes indices of
an element are stored contiguously (not necessarily in order).
"""
function get_outelement_qnode_indices(gquad::GlobalQuadrature, qnode_index)
    inelement_qnode_indices = get_inelement_qnode_indices(gquad, qnode_index)
    min_index, max_index = extrema(inelement_qnode_indices)
    n_qnodes = get_number_of_qnodes(gquad)
    iterator = Iterators.flatten((1:min_index-1, 
                                  max_index+1:n_qnodes))
    return iterator
end

"""
    get_nodedata_from_element(gquad::GlobalQuadrature, element)

Returns the node data `(nodes, normals, jacobians)` in `element`.
"""
function get_nodedata_from_element(gquad::GlobalQuadrature, element)
    nodes = gquad.nodes[element]
    normals = gquad.normals[element]
    jacobians = gquad.jacobians[element]
    return nodes, normals, jacobians
end

""" 
    compute_bounding_box(gquad::GlobalQuadrature)

Returns the bounding box `[low_corner, high_corner]` of the
quadrature nodes in `gquad`.
"""
function compute_bounding_box(gquad::GlobalQuadrature)
    low_corner = first(gquad.nodes)
    high_corner = first(gquad.nodes)
    for pt in gquad.nodes
        low_corner = min.(low_corner, pt)
        high_corner = max.(high_corner, pt)
    end
    return SVector(low_corner, high_corner)
end