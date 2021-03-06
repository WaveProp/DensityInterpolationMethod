"""
Methods for solving Maxwell's equations using 
Density Interpolation Method.
"""

"""
    assemble_interpolant_LQ_matrices!(dimdata::DimData)
    assemble_interpolant_LQ_matrices!(dimdata::DimData, element_index::Integer)

Assembles the matrix for computing the Density Interpolant and
stores its LQ decomposition in `dimdata`, for each element in `dimdata`.
"""
function assemble_interpolant_LQ_matrices!(dimdata::DimData)
    # Compute DIM matrices for each element
    for element_index in get_element_indices(dimdata.gquad) 
        assemble_interpolant_LQ_matrices!(dimdata, element_index)
    end
end
function assemble_interpolant_LQ_matrices!(dimdata::DimData, element_index)
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
    size_Lmatrix = size(lqobject.L)
    if size_Lmatrix[1] != size_Lmatrix[2]
        @warn """ The `L` interpolant matrix is not square.
        The number of sources of the interpolant is probably not enough.
        """
    end
    dimdata.Lmatrices[element_index] = LowerTriangular(lqobject.L)
    dimdata.Qmatrices[element_index] = Matrix(lqobject.Q)
end
function _assemble_submatrix!(dimdata::DimData, Mmatrix, qnode, src, n_qnodes, r_index, l_index) 
    k, _, _ = getparameters(dimdata)
    x, _, jac, n = get_qnode_data(qnode)
    # J????????G, size=2??3
    M0submatrix = transpose(jac) *
                  single_layer_kernel(x, src, k, n)  
    # J???(-n x ?????G), size=2??3
    M1submatrix = -transpose(jac) *
                  cross_product_matrix(n) * 
                  double_layer_kernel(x, src, k, n)  
    # Initial indices (i, j)
    initial_i0 = 2*r_index - 1              # for M0
    initial_i1 = initial_i0 + 2*n_qnodes    # for M1
    initial_j = 3*l_index - 2               # for both M0 and M1
    # copy M0submatrix and M1submatrix to Mmatrix
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
    compute_density_interpolant!(dimdata::DimData)
    compute_density_interpolant!(dimdata::DimData, element_index)

Computes the Density Interpolant coefficients, for each element
in `dimdata`. This assumes that [`assemble_interpolant_LQ_matrices!`](@ref) has already
been called.
"""
function compute_density_interpolant!(dimdata::DimData)
    # Compute DIM matrices for each element
    for element_index in get_element_indices(dimdata.gquad) 
        compute_density_interpolant!(dimdata, element_index)
    end
end
function compute_density_interpolant!(dimdata::DimData, element_index)
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
    # RHS = [????? ?????]?????, size=2??1,
    # where ?? is the surface density
    _, _, jacobian, _ = get_qnode_data(qnode)
    ?? = get_surface_density(dimdata, qnode)
    rhs = transpose(jacobian) * ??   
    index = 2*r_index - 1
    for i in 1:2
        Bvector[index] = rhs[i]
        index += 1
    end
end
function _apply_scaling_to_rhs!(dimdata::DimData, Bvector)
    # Transform a vector [b???, ..., b???, x, ..., x]???
    # into [??*b???, ..., ??*b???, ??*b???, ..., ??*b???]???
    @assert iseven(length(Bvector))
    _, ??, ?? = getparameters(dimdata)
    n = length(Bvector) ?? 2
    for i in 1:n
        Bvector[n+i] = ??*Bvector[i]
        Bvector[i] = ??*Bvector[i]
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
    compute_integral_operator!(dimdata::DimData)

Computes the integral operator `C??_{??,??}[??]` at all quadrature points,
using the density interpolation method.
"""
function compute_integral_operator!(dimdata::DimData)
    # Set integral operator value to zero
    reset_integral_operator_value!(dimdata)
    # Compute density interpolant for all elements
    compute_density_interpolant!(dimdata)
    # Loop for computing the integral operator.
    # (i, j) correspond to the indices of the 
    # observation and source qnodes, respectively.
    Threads.@threads for i in get_qnode_indices(dimdata.gquad)
        _compute_integral_operator_innerloop!(dimdata, i)
    end
    # Return integral op. value
    # at all quadrature points
    return dimdata.integral_op
end
function _compute_integral_operator_innerloop!(dimdata::DimData, i)
    qnode_i = get_qnode(dimdata.gquad, i)     # qnode i object
    element_index_i = qnode_i.element_index   # element index of qnode i
    for j in get_outelement_qnode_indices(dimdata.gquad, element_index_i)
        qnode_j = get_qnode(dimdata.gquad, j)     # qnode j object
        # Update integral op. value at qnode i
        dimdata.integral_op[i] +=
            _compute_integral_operator_integrand(dimdata, qnode_i, qnode_j)
    end
    # Interpolant ??????? at qnode i
    ???????i = evaluate_?????interpolant(dimdata, element_index_i, qnode_i)   
    # Update integral op. value at qnode i
    dimdata.integral_op[i] += -0.5*???????i
end
function _compute_integral_operator_integrand(dimdata::DimData, qnode_i, qnode_j)
    k, ??, ?? = getparameters(dimdata)
    # qnode i data
    yi, _, _, ni = get_qnode_data(qnode_i)
    element_index_i = qnode_i.element_index   # element index of qnode i
    # qnode j data
    yj, wj, _, nj = get_qnode_data(qnode_j)
    ??j = get_surface_density(dimdata, qnode_j)           # surf. dens. ?? at qnode j
    ???????j = evaluate_?????interpolant(dimdata, element_index_i, qnode_j)   # interpolant ??????? at qnode j
    ???????j = evaluate_?????interpolant(dimdata, element_index_i, qnode_j)   # interpolant ??????? at qnode j
    # operators
    K_input = ??*??j - ???????j               # Double layer input vector
    T_input = ??*cross(nj, ??j) - ???????j    # Single layer input vector
    K = double_layer_kernel(yi, yj, k, ni, K_input)   # Double layer operator
    T = single_layer_kernel(yi, yj, k, ni, T_input)   # Single layer operator
    return wj*(K + T)
end

"""
    compute_exterior_nystrom_integral_operator!(dimdata::DimData)

Computes the (exterior) Nystrom integral operator `C??_{??,??}[??]` at all quadrature points,
using the density interpolation method. This operator includes the double layer jump 
and the transposed jacobians.
"""
function compute_exterior_nystrom_integral_operator!(dimdata::DimData{F}) where F
    _, ??, _ = getparameters(dimdata)
    # Set integral operator value to zero
    reset_integral_operator_value!(dimdata)
    # Compute density interpolant for all elements
    compute_density_interpolant!(dimdata)
    # Loop for computing the integral operator.
    # (i, j) correspond to the indices of the 
    # observation and source qnodes, respectively.
    nystrom_iop = similar(dimdata.integral_op, ComplexPoint2D)
    Threads.@threads for i in get_qnode_indices(dimdata.gquad)
        # compute dimdata.integral_op[i]
        _compute_integral_operator_innerloop!(dimdata, i)
        # compute nystrom_iop[i]
        qnode_i = get_qnode(dimdata.gquad, i)     # qnode i object
        if F === IndirectDimFormulation
            ?? = get_surface_density(dimdata, qnode_i)
        elseif F === DirectDimFormulation
            ??, _ = get_surface_density(dimdata, qnode_i)
        end
        nystrom_iop[i] = dual_jacobian(qnode_i) * (??*??/2 + dimdata.integral_op[i])
    end
    return nystrom_iop
end

"""
    compute_potencial(dimdata::DimData, xlist::AbstractArray{Point3D})
    compute_potencial(dimdata::DimData, x)

Computes the potential `C_{??,??}[??]` at all points `x` in
`xlist`.
"""
function compute_potencial(dimdata::DimData, xlist::AbstractArray{Point3D})
    result = similar(xlist, ComplexPoint3D)
    Threads.@threads for i in eachindex(xlist)
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
    k, ??, ?? = getparameters(dimdata)
    yj, wj, _, nj = get_qnode_data(qnode)   # qnode j data
    ??j = get_surface_density(dimdata, qnode)    # surf. dens. ?? at qnode j
    # Double layer potencial
    K_input = ?? * ??j
    Kpot = double_layer_potential_kernel(x, yj, k, K_input)  
    # Single layer potencial
    T_input = ?? * cross(nj, ??j)
    Tpot = single_layer_potential_kernel(x, yj, k, T_input) 
    return wj*(Kpot + Tpot)
end



