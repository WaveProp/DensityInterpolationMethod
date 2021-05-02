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
    k, _, _ = getparameters(dimdata)
    src_list = dimdata.src_list
    qnode_list, normal_list, jac_list = get_nodedata_from_element(dimdata.gquad, 
                                                                  element_index)
    n_qnodes = length(qnode_list)    # number of qnodes
    n_src = length(src_list)         # number of src points

    # Initialize matrix
    # 4 equations per qnode
    # 3 unknowns per src point
    Mmatrix = Matrix{ComplexF64}(undef, 4*n_qnodes, 3*n_src) 

    # Assemble system
    for r_index in 1:n_qnodes
        qnode = qnode_list[r_index]         # quadrature node
        normal = normal_list[r_index]       # normal at qnode
        jacobian = jac_list[r_index]        # jacobian at qnode
        for l_index in 1:n_src
            src = src_list[l_index]       # src point
            _assemble_submatrix!(Mmatrix, qnode, normal, jacobian, 
                                 src, k, n_qnodes, r_index, l_index) 
        end
    end

    # Compute LQ and save matrices
    lqobject = lq!(Mmatrix)
    dimdata.Lmatrices[element_index] = LowerTriangular(lqobject.L)
    dimdata.Qmatrices[element_index] = Matrix(lqobject.Q)
end
function _assemble_submatrix!(Mmatrix, qnode, normal, jacobian, src, k, 
                              n_qnodes, r_index, l_index) 
    # Jᵗγ₀G, size=2×3
    M0submatrix = transpose(jacobian) *
                  single_layer_kernel(qnode, src, k, normal)  
    # Jᵗ(-n x γ₁G), size=2×3
    M1submatrix = -transpose(jacobian) *
                  cross_product_matrix(normal) * 
                  double_layer_kernel(qnode, src, k, normal)  

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
    # Bvector: pre-allocated RHS vector
    # Get data
    _, α, β = getparameters(dimdata)
    qnode_list, _, jac_list = get_nodedata_from_element(dimdata.gquad, 
                                                        element_index)
    n_qnodes = length(qnode_list)  # number of qnodes in element

    # Initialize RHS vector
    # 4 equations per qnode
    Bvector = Vector{ComplexF64}(undef, 4*n_qnodes)

    # Assemble RHS
    for r_index in 1:n_qnodes
        jacobian = jac_list[r_index]        # jacobian at qnode
        ϕcoeff = dimdata.ϕcoeff[r_index]    # density coefficients at qnode
        _assemble_rhs!(Bvector, jacobian, ϕcoeff, r_index)
    end
    _apply_scaling_to_rhs!(Bvector, α, β)

    # Solve system using LQ decomposition
    # and save solution
    _solve_dim_lq!(dimdata, Bvector, element_index)
end
function _assemble_rhs!(Bvector, jacobian, ϕcoeff, r_index) 
    # RHS = [τ₁ τ₂]ᵗ(ϕ₁τ₁ + ϕ₂τ₂), size=2×1,
    # where J = [τ₁ τ₂] is the jacobian
    rhs = transpose(jacobian) * jacobian * ϕcoeff   
    index = 2*r_index - 1
    for i in 1:2
        Bvector[index] = rhs[i]
        index += 1
    end
end
function _apply_scaling_to_rhs!(Bvector, α, β)
    # Transform a vector [b₁, ..., bₙ, x, ..., x]ᵗ
    # into [α*b₁, ..., α*bₙ, β*b₁, ..., β*bₙ]ᵗ
    @assert iseven(length(Bvector))
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
    mul!(dimdata.ccoeff[element_index], 
         adjoint(dimdata.Qmatrices[element_index]), 
         Bvector)     # ccoef = adjoint(Q)*y
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
    for i in get_qnode_indices(dimdata.gquad)
        element_index_i = get_element_index(dimdata.gquad, i)
        _compute_integral_operator_innerloop(dimdata, element_index_i, i)
    end
    # Return integral op. value
    # at all quadrature points
    return dimdata.integral_op
end
function _compute_integral_operator_innerloop(dimdata::DimData, element_index_i, i)
    yi = dimdata.gquad.nodes[i]     # qnode i
    ni = dimdata.gquad.normals[i]   # qnormal at qnode i
    for j in get_outelement_qnode_indices(dimdata.gquad, element_index_i)
        # Update integral op. value at qnode i
        dimdata.integral_op[i] +=
            _compute_integral_operator_integrand(dimdata, element_index_i,
                                                 yi, ni, j)
    end
    # Interpolant γ₀Φ at qnode i
    γ₀Φi = evaluate_γ₀dim(dimdata, element_index_i, i)   
    # Update integral op. value at qnode i
    dimdata.integral_op[i] += -0.5*γ₀Φi
end
function _compute_integral_operator_integrand(dimdata::DimData, element_index_i, 
                                              yi, ni, j)
    k, α, β = getparameters(dimdata)
    yj = dimdata.gquad.nodes[j]                          # qnode j
    nj = dimdata.gquad.normals[j]                        # qnormal at qnode j
    wj = dimdata.gquad.weigths[j]                        # qweigth at qnode j
    ϕj = get_surface_density(dimdata, j)                # surf. dens. ϕ at qnode j
    γ₀Φj = evaluate_γ₀dim(dimdata, element_index_i, j)   # interpolant γ₀Φ at qnode j
    γ₁Φj = evaluate_γ₁dim(dimdata, element_index_i, j)   # interpolant γ₁Φ at qnode j

    K_input = α*ϕj - γ₀Φj               # Double layer input vector
    T_input = β*cross(nj, ϕj) - γ₁Φj    # Single layer input vector
    K = double_layer_kernel(yi, yj, k, ni, K_input)   # Double layer operator
    T = single_layer_kernel(yi, yj, k, ni, T_input)   # Single layer operator
    return wj*(K + T)
end

function compute_potencial(dimdata::DimData, xlist::AbstractArray{Point3D})
    result = similar(xlist, ComplexPoint3D)
    for i in eachindex(xlist)
        x = xlist[i]
        result[i] = compute_potencial(dimdata, x)
    end
end
function compute_potencial(dimdata::DimData, x)
    result = zero(ComplexPoint3D)
    for j in get_qnode_indices(dimdata.gquad)
        
    end
end
function _compute_potencial_integrand(dimdata::DimData, j::Integer, x)
    k, α, β = getparameters(dimdata)
    yj = dimdata.gquad.nodes[j]     # qnode j
    nj = dimdata.gquad.normals[j]   # qnormal at qnode j
    wj = dimdata.gquad.weigths[j]   # qweigth at qnode j
    ϕj = get_surface_density(dimdata, j) # surf. dens. ϕ at qnode j
    # Double layer potencial
    Kpot = 1
    # Single layer potencial
    Tpot = 1
end



