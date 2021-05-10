"""
Methods for solving Maxwell's equations using 
Density Interpolation Method.
"""

"""
    assemble_dim_matrices(dimdata::DimData)
    assemble_dim_matrices(dimdata::DimData, element_index::Integer)

Assembles the matrix for computing the Density Interpolant and
stores its LQ decomposition in `dimdata`, for each element in `dimdata`.
"""
function assemble_dim_matrices(dimdata::DimData)
    # Compute DIM matrices for each element
    for element_index in get_element_indices(dimdata.gquad) 
        assemble_dim_matrices(dimdata, element_index)
    end
end
function assemble_dim_matrices(dimdata::DimData, element_index)
    # Get data
    qnode_list = get_qnodes(dimdata.gquad, element_index)   # list of qnodes in element
    src_list = dimdata.src_list      # list of interpolant source points
    n_qnodes = length(qnode_list)    # number of qnodes
    n_src = length(src_list)         # number of src points

    # Initialize matrix
    # 4 equations per qnode
    # 3 unknowns per src point
    Mmatrix = Matrix{ComplexF64}(undef, 4*n_qnodes, 3*n_src) 

    # Assemble system
    # `r_index` is the (local) qnode index
    #  `l_index` is the src index
    for r_index in 1:n_qnodes
        qnode = qnode_list[r_index]       # quadrature node object
        for l_index in 1:n_src
            src = src_list[l_index]       # src point
            _assemble_submatrix!(dimdata, Mmatrix, qnode, src, n_qnodes, r_index, l_index) 
        end
    end

    # Compute LQ and save matrices
    lqobject = lq!(Mmatrix)
    dimdata.Lmatrices[element_index] = LowerTriangular(lqobject.L)
    dimdata.Qmatrices[element_index] = Matrix(lqobject.Q)
end
function _assemble_submatrix!(dimdata::DimData, Mmatrix, qnode, src, n_qnodes, r_index, l_index) 
    k, _, _ = getparameters(dimdata)
    x, _, jac, n = get_qnode_data(qnode)
    # Jᵗγ₀G, size=2×3
    M0submatrix = transpose(jac) *
                  single_layer_kernel(x, src, k, n)  
    # Jᵗ(-n x γ₁G), size=2×3
    M1submatrix = -transpose(jac) *
                  cross_product_matrix(n) * 
                  double_layer_kernel(x, src, k, n)  

    # Initial indices (i, j)
    initial_i0 = 2*r_index - 1              # for M0
    initial_i1 = initial_i0 + 2*n_qnodes    # for M1
    initial_j = 3*l_index - 2               # for both M0 and M1

    index_j = initial_j
    for j in 1:3
        index_i0 = initial_i0
        index_i1 = initial_i1
        for i in 1:2
            Mmatrix[index_i0, index_j] = M0submatrix[i, j]
            Mmatrix[index_i1, index_j] = M1submatrix[i, j]
            index_i0 += 1
            index_i1 += 1
        end 
        index_j += 1
    end
end

"""
    compute_density_interpolant(dimdata::DimData)
    compute_density_interpolant(dimdata::DimData, element_index)

Computes the Density Interpolant coefficients, for each element
in `dimdata`. This assumes that [`assemble_dim_matrices`](@ref) has already
been called.
"""
function compute_density_interpolant(dimdata::DimData)
    # Compute DIM matrices for each element
    for element_index in get_element_indices(dimdata.gquad) 
        compute_density_interpolant(dimdata, element_index)
    end
end
function compute_density_interpolant(dimdata::DimData, element_index)
    # Get data
    qnode_list = get_qnodes(dimdata.gquad, element_index)   # list of qnodes in element
    n_qnodes = length(qnode_list)  # number of qnodes in element

    # Initialize RHS vector
    # 4 equations per qnode
    Bvector = Vector{ComplexF64}(undef, 4*n_qnodes)

    # Assemble RHS
    # `r_index` is the (local) qnode index
    for r_index in 1:n_qnodes
        qnode = qnode_list[r_index]
        _assemble_rhs!(dimdata, Bvector, qnode, r_index)
    end
    _apply_scaling_to_rhs!(dimdata, Bvector)

    # Solve system using LQ decomposition
    # and save solution
    _solve_dim_lq!(dimdata, Bvector, element_index)
end
function _assemble_rhs!(dimdata::DimData, Bvector, qnode, r_index) 
    # RHS = [τ₁ τ₂]ᵗϕ, size=2×1,
    # where ϕ is the surface density
    _, _, jacobian, _ = get_qnode_data(qnode)
    ϕ = get_surface_density(dimdata, qnode)
    rhs = transpose(jacobian) * ϕ   
    index = 2*r_index - 1
    for i in 1:2
        Bvector[index] = rhs[i]
        index += 1
    end
end
function _apply_scaling_to_rhs!(dimdata::DimData, Bvector)
    # Transform a vector [b₁, ..., bₙ, x, ..., x]ᵗ
    # into [α*b₁, ..., α*bₙ, β*b₁, ..., β*bₙ]ᵗ
    @assert iseven(length(Bvector))
    _, α, β = getparameters(dimdata)
    n = length(Bvector) ÷ 2
    for i in 1:n
        Bvector[n+i] = β*Bvector[i]
        Bvector[i] = α*Bvector[i]
    end
end
function _solve_dim_lq!(dimdata::DimData, Bvector, element_index)
    # Solves the density interpolant system
    # using LQ decomposition and saves result
    ldiv!(dimdata.Lmatrices[element_index], 
          Bvector)    # Solves Ly=b, store result in b
    mul!(dimdata.interpolant_coeff_data[element_index], 
         adjoint(dimdata.Qmatrices[element_index]), 
         Bvector)     # interpolant_coeff = adjoint(Q)*y
end

"""
    compute_integral_operator(dimdata::DimData)

Computes the integral operator `C̃_{α,β}[ϕ]` at all quadrature points,
using the density interpolation method.
"""
function compute_integral_operator(dimdata::DimData)
    n_nodes = get_number_of_qnodes(dimdata)
    # Set integral operator value to zero
    reset_integral_operator_value(dimdata)
    # Compute density interpolant for all elements
    compute_density_interpolant(dimdata)

    # Loop for computing the integral operator.
    # (i, j) correspond to the indices of the 
    # observation and source qnodes, respectively.
    Threads.@threads for i in get_qnode_indices(dimdata.gquad)
        element_index_i = get_element_index(dimdata.gquad, i)
        _compute_integral_operator_innerloop(dimdata, element_index_i, i)
    end
    # Return integral op. value
    # at all quadrature points
    return dimdata.integral_op
end
function _compute_integral_operator_innerloop(dimdata::DimData, element_index_i, i)
    qnode_i = get_qnode(dimdata.gquad, i)     # qnode i object
    for j in get_outelement_qnode_indices(dimdata.gquad, element_index_i)
        qnode_j = get_qnode(dimdata.gquad, j)     # qnode j object
        # Update integral op. value at qnode i
        dimdata.integral_op[i] +=
            _compute_integral_operator_integrand(dimdata, element_index_i, qnode_i, qnode_j)
    end
    # Interpolant γ₀Φ at qnode i
    γ₀Φi = evaluate_γ₀interpolant(dimdata, element_index_i, qnode_i)   
    # Update integral op. value at qnode i
    dimdata.integral_op[i] += -0.5*γ₀Φi
end
function _compute_integral_operator_integrand(dimdata::DimData, element_index_i, qnode_i, qnode_j)
    k, α, β = getparameters(dimdata)
    # qnode i data
    yi, _, _, ni = get_qnode_data(qnode_i)
    # qnode j data
    yj, wj, _, nj = get_qnode_data(qnode_j)
    ϕj = get_surface_density(dimdata, qnode_j)           # surf. dens. ϕ at qnode j
    γ₀Φj = evaluate_γ₀interpolant(dimdata, element_index_i, qnode_j)   # interpolant γ₀Φ at qnode j
    γ₁Φj = evaluate_γ₁interpolant(dimdata, element_index_i, qnode_j)   # interpolant γ₁Φ at qnode j

    K_input = α*ϕj - γ₀Φj               # Double layer input vector
    T_input = β*cross(nj, ϕj) - γ₁Φj    # Single layer input vector
    K = double_layer_kernel(yi, yj, k, ni, K_input)   # Double layer operator
    T = single_layer_kernel(yi, yj, k, ni, T_input)   # Single layer operator
    return wj*(K + T)
end

"""
    compute_potencial(dimdata::DimData, xlist::AbstractArray{Point3D})
    compute_potencial(dimdata::DimData, x)

Computes the potential `C_{α,β}[ϕ]` at all points `x` in
`xlist`.
"""
function compute_potencial(dimdata::DimData, xlist::AbstractArray{Point3D})
    result = similar(xlist, ComplexPoint3D)
    for i in eachindex(xlist)
        x = xlist[i]
        result[i] = compute_potencial(dimdata, x)
    end
    return result
end
function compute_potencial(dimdata::DimData, x)
    return sum(get_qnodes(dimdata.gquad)) do qnode
        _compute_potencial_integrand(dimdata, qnode, x)
    end
end
function _compute_potencial_integrand(dimdata::DimData, qnode::QNode, x)
    k, α, β = getparameters(dimdata)
    yj, wj, _, nj = get_qnode_data(qnode)   # qnode j data
    ϕj = get_surface_density(dimdata, qnode)    # surf. dens. ϕ at qnode j
    # Double layer potencial
    K_input = α * ϕj
    Kpot = double_layer_potential_kernel(x, yj, k, K_input)  
    # Single layer potencial
    T_input = β * cross(nj, ϕj)
    Tpot = single_layer_potential_kernel(x, yj, k, T_input) 
    return wj*(Kpot + Tpot)
end



