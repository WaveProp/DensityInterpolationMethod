function get_single_and_double_layer_operators(gquad, k)
    n_qnodes = get_number_of_qnodes(gquad)
    S = Matrix{MaxwellKernelType}(undef, n_qnodes, n_qnodes)
    D = Matrix{MaxwellKernelType}(undef, n_qnodes, n_qnodes)
    for j in 1:n_qnodes
        qnode_j = get_qnode(gquad, j)
        yj, wj, _, _ = get_qnode_data(qnode_j)  
        for i in 1:n_qnodes
            i == j && continue
            qnode_i = get_qnode(gquad, i)
            yi, _, _, ni = get_qnode_data(qnode_i)
            S[i, j] = wj*single_layer_kernel(yi, yj, k, ni)  
            D[i, j] = wj*double_layer_kernel(yi, yj, k, ni)  
    end
    return S, D
end

function get_lebedev_sources(gquad, n_src)
    # compute source points
    _, bbox_center, bbox_radius = compute_bounding_box(gquad)
    src_radius = r * bbox_radius
    src_list = get_sphere_sources_lebedev(n_src, src_radius, bbox_center)
    return src_list
end

function get_basis_dim(gquad, k, n_src)
    function γₒ(qnode, src)
        # Jᵗγ₀G, size=2×3
        x, _, jac, n = get_qnode_data(qnode)
        return transpose(jac)*single_layer_kernel(x, src, k, n) 
    end
    function γ₁(qnode, src)
        # Jᵗ(-n x γ₁G), size=2×3
        x, _, jac, n = get_qnode_data(qnode)
        return -transpose(jac)*cross_product_matrix(n)*double_layer_kernel(x, src, k, n)  
    end
    src_list = get_lebedev_sources(gquad, n_src) # list of Lebedev sources
    basis     = [(qnode) -> γₒ(qnode, src) for src in src_list]
    γ₁_basis  = [(qnode) -> γ₁(qnode, src) for src in src_list]
    return basis, γ₁_basis
end

function get_auxiliary_quantities_dim(gquad,Sop,Dop,basis,γ₁_basis)
    n_src = length(basis)
    # compute matrix of basis evaluated on Y
    ynodes = get_qnodes(gquad)
    γ₀B = Matrix{T}(undef, length(ynodes), n_src)
    γ₁B = Matrix{T}(undef, length(ynodes), n_src)
    for k in 1:n_src
        for i in 1:length(ynodes)
            γ₀B[i,k] = basis[k](ynodes[i])
            γ₁B[i,k] = γ₁_basis[k](ynodes[i])
        end
    end
    # integrate the basis over Y
    R = Op1*γ₁B - Op2*γ₀B
    return γ₀B, γ₁B, R
end

function single_doublelayer_dim(gquad::GlobalQuadrature)
    k = 2.0
    n_src = 26
    
    S, D = get_single_and_double_layer_operators(gquad, k)
    # precompute dim quantities
     # list of functions γ₀ and γ₁ for each source
    basis, γ₁_basis = get_basis_dim(gquad, k, n_src) 
    γ₀B, γ₁B, R = get_auxiliary_quantities_dim(gquad,Sop,Dop,basis,γ₁_basis)
    # compute corrections
    δS = _singular_weights_dim(Sop,γ₀B,γ₁B,R,dict_near)
    δD = _singular_weights_dim(Dop,γ₀B,γ₁B,R,dict_near)
    # add corrections to the dense part
    axpy!(true,δS,S)  # S = S + δS
    axpy!(true,δD,D)  # D = D + δD
    return S, D
end