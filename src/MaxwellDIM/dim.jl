"""
Methods for solving Maxwell's equations using 
Density Interpolation Method.
"""

# In construction...
function construct_density_interpolant(dimdata::DimData, element_index)
    # Get data
    k, α, β = getparameters(dimdata)
    src_list = dimdata.src_list
    qnode_list, normal_list, jac_list = get_nodedata_from_element(dimdata.gquad, 
                                                                  element_index)
    n_qnodes = length(qnode_list)    # number of qnodes
    n_src = length(src_list)         # number of src points

    # Initialize matrices
    # 6 equations per qnode
    # 3 unknowns per src point
    Mmatrix = zeros(ComplexF64, 6*n_qnodes, 3*n_src)
    Bvector = zeros(ComplexF64, 6*n_qnodes)

    # Assemble system
    for r_index in 1:n_qnodes
        qnode = qnode_list[r_index]         # quadrature node
        normal = normal_list[r_index]       # normal at qnode
        jacobian = jac_list[r_index]        # jacobian at qnode
        ϕcoeff = dimdata.ϕcoeff[r_index]    # density coefficients at qnode
        _assemble_rhs!(Bvector, jacobian, ϕcoeff, r_index)
        for l_index in 1:n_src
            src = src_list[l_index]       # src point
            _assemble_matrix!(Mmatrix, qnode, normal, src, k, 
                              n_qnodes, r_index, l_index) 
        end
    end

    # Solve system
    # Direct solver (for the moment...)
    _apply_scaling_to_rhs!(Bvector, α, β)
    Ccoeff = Mmatrix \ Bvector
    println("size $(size(Mmatrix)), rank $(rank(Mmatrix))")
    println("norm $(norm(Mmatrix * Ccoeff - Bvector))")
    save_dimcoeff!(dimdata, element_index, Ccoeff)
end
function _assemble_rhs!(Bvector, jacobian, ϕcoeff, r_index) 
    rhs = jacobian * ϕcoeff   # ϕ₁τ₁ + ϕ₂τ₂, where τ₁,τ₂ are the tangential vectors
    index = 3*r_index - 2
    for i in 1:3
        Bvector[index] = rhs[i]
        index += 1
    end
end
function _assemble_matrix!(Mmatrix, qnode, normal, src, k, n_qnodes, r_index, l_index) 
    # γ₀G
    M0submatrix = single_layer_kernel(qnode, src, k, normal)  
    # -n x γ₁G
    M1submatrix = -cross_product_matrix(normal) * 
                  double_layer_kernel(qnode, src, k, normal)  

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

