function compute_nystrom_maxwell_matrix(dimdata::IndirectDimData, formtype::NystromFormulationType)
    V₀ = _NystromMaxwellOperator{formtype}(dimdata)
    V₁ = generate_interpolant_forwardmap_matrix(dimdata)
    M = blockmatrix_to_matrix(V₀+V₁)
    return M
end

function compute_nystrom_maxwell_rhs(dimdata::IndirectDimData, field)
    n_qnodes = get_number_of_qnodes(dimdata)
    @assert length(field) == n_qnodes
    rhs = similar(field, ComplexPoint2D)
    for i in 1:n_qnodes
        qnode = get_qnode(dimdata.gquad, i)
        _, _, jacᵢ, ni = get_qnode_data(qnode)
        rhs[i] = transpose(jacᵢ) * cross(ni, field[i])
    end
    return reinterpret(ComplexF64, rhs)
end

function solve_nystrom_LU!(dimdata::IndirectDimData, A, b)
    dimdata.density_coeff_data .= A \ b
    compute_density_interpolant!(dimdata)
end

function solve_nystrom_GMRES! end