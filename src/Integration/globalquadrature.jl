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
    # to the initial qnode index and the final qnode index for each 
    # element, inclusive.
    el2indices::Vector{SVector{2, Int64}} = SVector{2, Int64}[]       
    
    # Mapping from qnode indices to element indices.
    index2element::Vector{Int64} = zeros(Int64, N) 
end

"""
    generate_globalquadrature(mesh::GenericMesh, order)

Generates a GlobalQuadrature struct from a GenericMesh using a quadrature
rule of order `order`.
"""
function generate_globalquadrature(mesh::GenericMesh; order=2)
    n_qnodes = get_number_of_qnodes(mesh, order)
    gquad = GlobalQuadrature{n_qnodes}()

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
    push!(gquad.el2indices, SVector(initial_index, final_index))
    return index
end

"""
    get_number_of_qnodes(gquad::GlobalQuadrature{N})

Returns the total number of quadrature nodes of the GlobalQuadrature.
"""
function get_number_of_qnodes(gquad::GlobalQuadrature{N}) where N
    return N
end

"""
    get_number_of_elements(gquad::GlobalQuadrature)

Returns the total number of elements of the GlobalQuadrature.
"""
function get_number_of_elements(gquad::GlobalQuadrature)
    return length(gquad.el2indices)
end