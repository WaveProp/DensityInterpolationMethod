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
    struct QNode

Structure that contains the concrete information of a single 
quadrature node, i.e., the quadrature point, weigth, normal and
jacobian.
"""                        
struct QNode
    index::Int64         # QNode global index
    element_index::Int64 # Index of the element to which QNode belongs
    qnode::Point3D       # *Lifted* quadrature nodes
    weigth::Float64      # *Lifted* quadrature weigth
    jacobian::JacMatrix  # Jacobian matrices at qnode
    normal::Point3D      # Normal vectors at qnodes
end

"""
    get_qnode_data(qnode::QNode)

Returns `(x, w, jac, n)` of `qnode`, where `x` is the *lifted* quadrature node, 
`w` is the *lifted* quadrature weight, `jac` is the jacobian at `x` 
and `n` is the normal at `x`.
"""
function get_qnode_data(qnode::QNode)
    return qnode.qnode, qnode.weigth, qnode.jacobian, qnode.normal
end

"""
    dual_jacobian(qnode::QNode)

Returns the dual jacobian `djac` of `qnode`. Basically, `djac = pinv(qnode.jacobian)`.
"""
function dual_jacobian(qnode::QNode)
    t1 = qnode.jacobian[:,1]
    t2 = qnode.jacobian[:,2]
    n = qnode.normal
    V = norm(cross(t1, t2))
    e1 = transpose(cross(t2, n) / V)
    e2 = transpose(cross(n, t1) / V)
    djac = vcat(e1, e2)
    #return transpose(qnode.jacobian)
    return djac
end

"""
    struct GlobalQuadrature

Structure that contains the concrete information of the quadrature,
i.e., quadrature nodes (`QNode`) and elements.
"""                        
struct GlobalQuadrature
    # List of QNodes.
    qnodes::Vector{QNode}    
    # List of elements.
    # Each element correspond to a vector `[i₁, i₂, ..., iₙ]` which 
    # contains the indices of the `n` qnodes of the element.
    elements::Vector{Vector{Int64}}     
    # Constructor
    function GlobalQuadrature(n_qnodes, n_elements)
        qnodes = Vector{QNode}(undef, n_qnodes)
        elements = Vector{Vector{Int64}}(undef, n_elements)
        return new(qnodes, elements)
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

    qnode_index = 1  # for computing the qnode indices
    for (etype, elements) in get_etypes_and_elements(mesh)
        qrule = get_qrule_for_element(etype, order)
        qnode_index = _add_elementlist_to_gquad(gquad, elements, qrule, qnode_index)
    end
    @assert qnode_index == n_qnodes+1 # sanity check
    return gquad
end
function _add_elementlist_to_gquad(gquad::GlobalQuadrature, elements, qrule, qnode_index)
    for el in elements
        qnode_index = _add_element_to_gquad(gquad, el, qrule, qnode_index)
    end
    return qnode_index
end
function _add_element_to_gquad(gquad::GlobalQuadrature, el, qrule, qnode_index)
    # Get element index
    if qnode_index > 1
        last_qnode = gquad.qnodes[qnode_index-1]
        element_index = last_qnode.element_index+1
    else
        element_index = 1
    end
    # Save data in GlobalQuadrature
    qdata = get_quadrature_data(qrule, el)
    initial_index = qnode_index
    for (xᵢ, wᵢ, jacᵢ, nᵢ) in qdata
        gquad.qnodes[qnode_index] = 
            QNode(qnode_index, element_index, xᵢ, wᵢ, jacᵢ, nᵢ)
        qnode_index += 1
    end
    final_index = qnode_index-1

    # Save element data
    gquad.elements[element_index] = collect(initial_index:final_index)
    return qnode_index
end

"""
    get_qnode(gquad::GlobalQuadrature, qnode_index::Integer)

Returns the `QNode` with index `qnode_index`.
"""
function get_qnode(gquad::GlobalQuadrature, qnode_index::Integer)
    qnode = gquad.qnodes[qnode_index]
    return qnode
end

"""
    get_qnodes(gquad::GlobalQuadrature)
    get_qnodes(gquad::GlobalQuadrature, element_index::Integer)

Returns all `QNodes` in `gquad`. Otherwise, returns the `QNodes` in
the element `element_index`.
"""
function get_qnodes(gquad::GlobalQuadrature)
    return gquad.qnodes
end
function get_qnodes(gquad::GlobalQuadrature, element_index::Integer)
    qnode_indices = get_inelement_qnode_indices(gquad, element_index)
    return gquad.qnodes[qnode_indices]
end

"""
    get_number_of_qnodes(gquad::GlobalQuadrature)

Returns the total number of quadrature nodes of the GlobalQuadrature.
"""
function get_number_of_qnodes(gquad::GlobalQuadrature)
    return length(gquad.qnodes)
end

"""
    get_number_of_elements(gquad::GlobalQuadrature)

Returns the total number of elements of the GlobalQuadrature.
"""
function get_number_of_elements(gquad::GlobalQuadrature)
    return length(gquad.elements)
end

"""
    get_qnode_indices(gquad::GlobalQuadrature)

Returns an `UnitRange` containing the indices of all qnodes.
"""
function get_qnode_indices(gquad::GlobalQuadrature)
    return eachindex(gquad.qnodes)
end

"""
    get_element_indices(gquad::GlobalQuadrature)

Returns an `UnitRange` containing the indices of all elements.
"""
function get_element_indices(gquad::GlobalQuadrature)
    return eachindex(gquad.elements)
end

"""
    get_element_index(gquad::GlobalQuadrature, qnode_index::Integer)

Returns the element index where `qnode_index` belongs.
"""
function get_element_index(gquad::GlobalQuadrature, qnode_index::Integer)
    qnode = gquad.qnodes[qnode_index]
    return qnode.element_index
end

"""
    get_inelement_qnode_indices(gquad::GlobalQuadrature, element_index::Integer)

Returns the indices of the qnodes that lie on the element `element_index`.
"""
function get_inelement_qnode_indices(gquad::GlobalQuadrature, element_index::Integer)
    inelement_qnode_indices = gquad.elements[element_index]
    return inelement_qnode_indices
end

"""
    get_outelement_qnode_indices(gquad::GlobalQuadrature, element_index::Integer)

Returns an iterator with the indices of the qnodes that do not lie 
on the element `element_index`. This assumes that the qnodes of
an element are stored (in `gquad.qnodes`) contiguously (not necessarily in order).
"""
function get_outelement_qnode_indices(gquad::GlobalQuadrature, element_index::Integer)
    inelement_qnode_indices = get_inelement_qnode_indices(gquad, element_index)
    min_index, max_index = extrema(inelement_qnode_indices)
    n_qnodes = get_number_of_qnodes(gquad)
    iterator = Iterators.flatten((1:min_index-1, 
                                  max_index+1:n_qnodes))
    return iterator
end

"""
    is_qnode_in_element(gquad::GlobalQuadrature, qnode_index, element_index)

Returns `true` if `qnode_index` belongs to element `element_index`, otherwise
returns `false`.
"""
function is_qnode_in_element(gquad::GlobalQuadrature, qnode_index, element_index)
    inelement_qnodes = get_inelement_qnode_indices(gquad, element_index)
    return qnode_index ∈ inelement_qnodes
end

""" 
    compute_bounding_box(gquad::GlobalQuadrature)

Returns the bounding box `[low_corner, high_corner]`, its center and radius,
for the quadrature nodes in `gquad`.
"""
function compute_bounding_box(gquad::GlobalQuadrature)
    first_qnode = first(gquad.qnodes)
    low_corner = first_qnode.qnode
    high_corner = first_qnode.qnode
    for qnode_obj in gquad.qnodes
        pt = qnode_obj.qnode
        low_corner = min.(low_corner, pt)
        high_corner = max.(high_corner, pt)
    end
    bbox = SVector(low_corner, high_corner)
    center = (low_corner + high_corner) / 2
    radius = norm(low_corner - high_corner) / 2
    return bbox, center, radius
end