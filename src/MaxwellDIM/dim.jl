"""
Structures and methods for solving Maxwell's equations using 
Density Interpolation Method.
"""

"""
    struct DimData

Structure that contains all necessary data
for the Density Interpolation Method.
"""
struct DimData
    mesh::GenericMesh
    gquad::GlobalQuadrature
end

# In construction...
function compute_density_interpolant(qguad::GlobalQuadrature, element, src_points)
    α = 1
    β = 1
    τvec = SVector(1, 1)

    # Get data
    node_list, normal_list, jac_list = get_nodedata_from_element(qguad, element)
    n_qnodes = length(qnodes)
    n_src = length(src_points)

    # Initialize matrices
    # 6 equations per qnode
    # 3 unknowns per src point
    Mmatrix = zeros(ComplexF64, 6*n_qnodes, 3*n_src)
    Bvector = zeros(ComplexF64, 3*n_src)

    # Assemble system
    for r in 1:n_qnodes
        node = node_list[r]
        normal = normal_list[r]
        jac = jac_list[r]
        _assemble_rhs!(Bvector, α, β, τvec, jac, n_qnodes, r)
        _assemble_matrix!(Mmatrix, node, normal, src_points, n_qnodes, n_src, r)
    end

    # Solve system
    # Direct solver (for the moment...)
    Ccoeff = Mmatrix \ Bvector
    return Ccoeff
end
function _assemble_rhs!(Bvector, α, β, τvec, jacobian, n_qnodes, r) 
    rhs = jacobian * τvec
    for i in 1:3
        index1 = r+i-1 #check
        index2 = r+i+3*n_qnodes-1 #check
        Bvector[index1] = α * rhs[i]
        Bvector[index2] = β * rhs[i]
    end
end
function _assemble_matrix!(Mmatrix, node, normal, src_points, n_qnodes, n_src, r) 
    for l in 1:n_src
        src = src_points[l]
        _assemble_submatrix!(Mmatrix, node, normal, src, n_qnodes, r, l) 
    end
end
function _assemble_submatrix!(Mmatrix, node, normal, src, n_qnodes, r, l) 
end






end