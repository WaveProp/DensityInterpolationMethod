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
    mesh::GenericMesh              # Contains the elements parametrization
    gquad::GlobalQuadrature        # Contains quadrature data
    k::Float64                     # Wavenumber

    # Surface density ϕ at quadrature nodes yⱼ.
    # Each entry (ϕ₁, ϕ₂) correspond to the coefficients
    # in ϕ(yⱼ) = ϕ₁τ₁ + ϕ₂τ₂, where τ₁,τ₂ are the tangential 
    # vectors at the quadrature node yⱼ.
    ϕ::Vector{Point2D}
end

# In construction...
function construct_density_interpolant(dimdata::DimData, element, src_points)
    α = 1
    β = 1

    # Get data
    node_list, normal_list, jac_list = get_nodedata_from_element(dimdata.gquad, element)
    n_qnodes = length(qnodes)
    n_src = length(src_points)

    # Initialize matrices
    # 6 equations per qnode
    # 3 unknowns per src point
    Mmatrix = zeros(ComplexF64, 6*n_qnodes, 3*n_src)
    Bvector = zeros(ComplexF64, 6*n_qnodes)

    # Assemble system
    for r_index in 1:n_qnodes
        node = node_list[r_index]
        normal = normal_list[r_index]
        jacobian = jac_list[r_index]
        ϕcoeff = dimdata.ϕ[r_index]
        _assemble_rhs!(Bvector, jacobian, ϕcoeff, r_index)
        for l_index in 1:n_src
            src = src_points[l_index]
            _assemble_matrix!(Mmatrix, node, normal, src, dimdata.k, 
                              n_qnodes, r_index, l_index) 
        end
    end

    # Solve system
    # Direct solver (for the moment...)
    _apply_scaling_to_rhs!(Bvector, α, β)
    Ccoeff = Mmatrix \ Bvector
    return Ccoeff
end
function _assemble_rhs!(Bvector, jacobian, ϕcoeff, r_index) 
    rhs = jacobian * ϕcoeff   # ϕ₁τ₁ + ϕ₂τ₂, where τ₁,τ₂ are the tangential vectors
    index = 3*r_index - 2
    for i in 1:3
        Bvector[index] = rhs[i]
        index += 1
    end
end
function _assemble_matrix!(Mmatrix, node, normal, src, k, n_qnodes, r_index, l_index) 
    # γ₀G
    M0submatrix = single_layer_kernel(node, src, k, normal)  
    # -n x γ₁G
    M1submatrix = -cross_product_matrix(normal) * 
                  double_layer_kernel(node, src, k, normal)  

    # Initial indices (i, j)
    initial_i0 = 3*r_index - 2              # for M0
    initial_i1 = initial_i0 + 3*n_qnodes    # for M1
    initial_j = 3*l_index - 2               # for both M0 and M1

    index_j = initial_j
    for j in 1:3
        index_i0 = initial_i0
        index_i1 = initial_i1
        for i in 1:3
            Mmatrix[index_i0, index_j] = M0submatrix[i, j]
            Mmatrix[index_i1, index_j] = M1submatrix[i, j]
            index_i0 += 1
            index_i1 += 1
        end 
        index_j += 1
    end
end
function _apply_scaling_to_rhs!(Bvector, α, β)
    # Transform a vector [b₁, ..., bₙ, 0, ..., 0]ᵗ
    # into [α*b₁, ..., α*bₙ, β*b₁, ..., β*bₙ]ᵗ
    @assert iseven(length(Bvector))
    n = length(Bvector) ÷ 2
    for i in 1:n
        Bvector[n+i] = β*Bvector[i]
        Bvector[i] = α*Bvector[i]
    end
end

