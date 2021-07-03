
"""
    compute_nystrom_maxwell_matrix(dimdata::IndirectDimData, formtype::NystromFormulationType)

Computes and returns the matrix used in the Nystrom integral equation. Threads are used to speed up
computations.
"""
function compute_nystrom_maxwell_matrix(dimdata::IndirectDimData, formtype::NystromFormulationType)
    V₀ = _NystromMaxwellOperator{formtype}(dimdata)
    V₁ = generate_interpolant_forwardmap_matrix(dimdata)
    n_qnodes = get_number_of_qnodes(dimdata)  
    M = generate_pseudoblockmatrix(ReducedReducedMaxwellKernelType, n_qnodes, n_qnodes)
    # M = blockmatrix_to_matrix(V₀+V₁)
    Threads.@threads for j in 1:n_qnodes
        for i in 1:n_qnodes
            M[Block(i, j)] = V₀[i, j] + V₁[i, j]
        end
    end
    return get_matrix_from_pseudoblockmatrix(M)
end

"""
    compute_nystrom_maxwell_rhs(dimdata::IndirectDimData, field)

Computes the right-hand-side of the Nystrom integral equation using the vector field `field`.
"""
function compute_nystrom_maxwell_rhs(dimdata::DimData, field)
    n_qnodes = get_number_of_qnodes(dimdata)
    @assert length(field) == n_qnodes
    rhs = similar(field, ComplexPoint2D)
    for i in 1:n_qnodes
        qnode = get_qnode(dimdata.gquad, i)
        _, _, _, ni = get_qnode_data(qnode)
        rhs[i] = dual_jacobian(qnode) * cross(ni, field[i])
    end
    return reinterpret(ComplexF64, rhs)
end

function solve_nystrom_LU!(dimdata::IndirectDimData, A, b)
    dimdata.density_coeff_data .= A \ b
    compute_density_interpolant!(dimdata) # for future evaluations
end

function solve_nystrom_GMRES!(dimdata::IndirectDimData, A, b; initialguess=b, kwargs...)
    copyto!(dimdata.density_coeff_data, initialguess)
    out = gmres!(dimdata.density_coeff_data, A, b; kwargs...)
    compute_density_interpolant!(dimdata) # for future evaluations
    return out
end