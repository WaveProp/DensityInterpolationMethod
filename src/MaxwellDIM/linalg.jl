function compute_nystrom_maxwell_matrix(dimdata::IndirectDimData, formtype::NystromFormulationType)
    V₀ = _NystromMaxwellOperator{formtype}(dimdata)
    V₁ = generate_interpolant_forwardmap_matrix(dimdata)
    n_qnodes = get_number_of_qnodes(dimdata)  
    block_sizes = [DIMENSION2 for _ in 1:n_qnodes] # number of unknowns = (2 per qnode)
    M = PseudoBlockArray{ComplexF64}(undef, block_sizes, block_sizes)
    # M = blockmatrix_to_matrix(V₀+V₁)
    for j in 1:n_qnodes
        for i in 1:n_qnodes
            M[Block(i, j)] = V₀[i, j] + V₁[i, j]
        end
    end
    return Array(M)
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
    compute_density_interpolant!(dimdata) # for future evaluations
end

function solve_nystrom_GMRES! end