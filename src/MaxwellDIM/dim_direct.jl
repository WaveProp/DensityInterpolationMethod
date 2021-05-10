"""
DIM methods for `DirectDimData`.
"""

"""
    get_surface_densities(dimdata::DirectDimData, qnode_index::Integer)

Returns the surface densities `ϕ(yⱼ)` and `φ(yⱼ)` at qnode `j = qnode_index`,
of operators K and T, respectively.
"""
function get_surface_density(dimdata::DirectDimData, qnode::QNode)
    jac = qnode.jacobian       # jacobian
    qnode_index = qnode.index  # qnode global index
    ϕ_coeff = dimdata.density_coeff[qnode_index]
    φ_coeff = dimdata.density2_coeff[qnode_index]
    ϕ = jac * ϕ_coeff
    φ = jac * φ_coeff
    return ϕ, φ
end

"""
    project_field_onto_surface_density(dimdata::DirectDimData, Kfield, Tfield)

Projects the tangential component of a vector field `field`, 
defined on the quadrature nodes, onto the surface densities `ϕ` and `φ`.
"""
function project_field_onto_surface_density(dimdata::DirectDimData, Kfield, Tfield)
    @assert length(Kfield) == length(Tfield) == get_number_of_qnodes(dimdata)
    for i in get_qnode_indices(dimdata.gquad)
        Kvec = Kfield[i]                    # Kvector field at qnode i
        Tvec = Tfield[i]                    # Tvector field at qnode i
        qnode = get_qnode(dimdata.gquad, i) # qnode i object
        jac = qnode.jacobian                # jacobian at qnode i
        dimdata.density_coeff[i] = jac \ Kvec   # ϕ density
        dimdata.density2_coeff[i] = jac \ Tvec  # φ density
    end
end

function _assemble_submatrix!(dimdata::DirectDimData, Mmatrix, qnode, src, n_qnodes, r_index, l_index) 
    k, _, _ = getparameters(dimdata)
    x, _, jac, n = get_qnode_data(qnode)
    # Jᵗγ₀G, size=2×3
    M0submatrix = transpose(jac) *
                  single_layer_kernel(x, src, k, n)  
    # Jᵗγ₁G, size=2×3
    M1submatrix = transpose(jac) *
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

function _assemble_rhs!(dimdata::DirectDimData, Bvector, qnode, r_index) 
    # RHS_1 = α[τ₁ τ₂]ᵗϕ, size=2×1,
    # RHS_2 = β[τ₁ τ₂]ᵗφ, size=2×1,
    # where J = [τ₁ τ₂] is the jacobian
    _, α, β = getparameters(dimdata)
    _, _, jacobian, _ = get_qnode_data(qnode)
    ϕ, φ = get_surface_density(dimdata, qnode)
    rhs_1 = α * transpose(jacobian) * ϕ
    rhs_2 = β * transpose(jacobian) * φ 

    index_1 = 2*r_index - 1
    index_2 = index_1 + length(Bvector)÷2
    for i in 1:2
        Bvector[index_1] = rhs_1[i]
        index_1 += 1
        Bvector[index_2] = rhs_2[i]
        index_2 += 1
    end
end

function _apply_scaling_to_rhs!(dimdata::DirectDimData, Bvector)
    # Do nothing
    return
end

function _compute_integral_operator_integrand(dimdata::DirectDimData, element_index_i, qnode_i, qnode_j)
    k, α, β = getparameters(dimdata)
    # qnode i data
    yi, _, _, ni = get_qnode_data(qnode_i)
    # qnode j data
    yj, wj, _, nj = get_qnode_data(qnode_j)
    ϕj, φj = get_surface_density(dimdata, qnode_j)    # surf. dens. ϕ and φ at qnode j
    γ₀Φj = evaluate_γ₀interpolant(dimdata, element_index_i, qnode_j)   # interpolant γ₀Φ at qnode j
    γ₁Φj = evaluate_γ₁interpolant(dimdata, element_index_i, qnode_j)   # interpolant γ₁Φ at qnode j

    K_input = α*ϕj - γ₀Φj               # Double layer input vector
    T_input = β*φj - γ₁Φj    # Single layer input vector
    K = double_layer_kernel(yi, yj, k, ni, K_input)   # Double layer operator
    T = single_layer_kernel(yi, yj, k, ni, T_input)   # Single layer operator
    return wj*(K + T)
end

function _compute_potencial_integrand(dimdata::DirectDimData, qnode::QNode, x)
    k, α, β = getparameters(dimdata)
    yj, wj, _, nj = get_qnode_data(qnode)   # qnode j data
    ϕj, φj = get_surface_density(dimdata, qnode)    # surf. dens. ϕ and φ at qnode j
    # Double layer potencial
    K_input = α * ϕj
    Kpot = double_layer_potential_kernel(x, yj, k, K_input)  
    # Single layer potencial
    T_input = β * φj
    Tpot = single_layer_potential_kernel(x, yj, k, T_input) 
    return wj*(Kpot + Tpot)
end