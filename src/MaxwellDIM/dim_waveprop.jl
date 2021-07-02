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
    end
    return S, D
end

function get_lebedev_sources(gquad, n_src; kfactor=5)
    # compute source points
    _, bbox_center, bbox_radius = compute_bounding_box(gquad)
    src_radius = kfactor * bbox_radius
    src_list = get_sphere_sources_lebedev(n_src, src_radius, bbox_center)
    return src_list
end

function get_basis_dim(gquad, k, n_src)
    function γₒ(qnode, src)
        x, _, _, n = get_qnode_data(qnode)
        return single_layer_kernel(x, src, k, n) 
    end
    function γ₁(qnode, src)
        x, _, _, n = get_qnode_data(qnode)
        return double_layer_kernel(x, src, k, n)  
    end
    src_list = get_lebedev_sources(gquad, n_src) # list of Lebedev sources
    basis = [(qnode) -> γₒ(qnode, src) for src in src_list]
    γ₁_basis = [(qnode) -> γ₁(qnode, src) for src in src_list]
    return basis, γ₁_basis
end

function get_auxiliary_quantities_dim(gquad,Sop,Dop,basis,γ₁_basis)
    n_src = length(basis)
    # compute matrix of basis evaluated on Y
    ynodes = get_qnodes(gquad)
    γ₀B = Matrix{MaxwellKernelType}(undef, length(ynodes), n_src)
    γ₁B = Matrix{MaxwellKernelType}(undef, length(ynodes), n_src)
    for k in 1:n_src
        for i in 1:length(ynodes)
            γ₀B[i,k] = basis[k](ynodes[i])
            γ₁B[i,k] = γ₁_basis[k](ynodes[i])
        end
    end
    # integrate the basis over Y
    R = -0.5*γ₀B - Dop*γ₀B - Sop*γ₁B
    return γ₀B, γ₁B, R
end

function get_singular_weights_dim(gquad,γ₀B,γ₁B,R)
    n_src = size(γ₀B,2)
    T = MaxwellKernelType
    Is = Int[]
    Js = Int[]
    Ss = T[]   # for single layer
    Ds = T[]   # for double layer
    num_els = get_number_of_elements(gquad)
    # FIXME: assumes all elements are equal
    n_qnodes_per_element = length(first(gquad.elements))
    M = Matrix{T}(undef,2*n_qnodes_per_element,n_src)
    for n in 1:num_els
        j_glob = get_inelement_qnode_indices(gquad, n)
        M[1:n_qnodes_per_element,:]     = @view γ₀B[j_glob,:]
        M[n_qnodes_per_element+1:end,:] = @view γ₁B[j_glob,:]
        F = pinv(blockmatrix_to_matrix(M))  # FIXME: Change pseudoinverse for LQ
        for i in j_glob
            tmp_scalar  = blockmatrix_to_matrix(R[i:i,:]) * F
            tmp = matrix_to_blockmatrix(tmp_scalar,T)
            Dw = view(tmp,1:n_qnodes_per_element)
            Sw = view(tmp,(n_qnodes_per_element+1):(2*n_qnodes_per_element))
            #w = axpby!(a,view(tmp,1:n_qnodes),b,view(tmp,(n_qnodes+1):(2*n_qnodes)))
            append!(Is,fill(i,n_qnodes_per_element))
            append!(Js,j_glob)
            append!(Ss,Sw)
            append!(Ds,Dw)
        end
    end
    n_qnodes = get_number_of_qnodes(gquad)
    Sp = sparse(Is,Js,Ss,n_qnodes,n_qnodes)
    Dp = sparse(Is,Js,Ds,n_qnodes,n_qnodes)
    return Sp, Dp
end

function single_doublelayer_dim(gquad::GlobalQuadrature; k, n_src)
    S, D = get_single_and_double_layer_operators(gquad, k)
    # precompute dim quantities
     # list of functions γ₀ and γ₁ for each source
    basis, γ₁_basis = get_basis_dim(gquad, k, n_src) 
    γ₀B, γ₁B, R = get_auxiliary_quantities_dim(gquad,S,D,basis,γ₁_basis)
    # compute corrections
    δS, δD = get_singular_weights_dim(gquad,γ₀B,γ₁B,R)
    # add corrections to the dense part
    axpy!(true,δS,S)  # S = S + δS
    axpy!(true,δD,D)  # D = D + δD
    return S, D
end

function diagonal_ncross_jac_matrix(gquad)
    qnodes = get_qnodes(gquad)
    nmatrix = Diagonal([cross_product_matrix(q.normal) for q in qnodes])
    jmatrix = Diagonal([q.jacobian for q in qnodes])
    return nmatrix, jmatrix
end

function assemble_dim_exterior_nystrom_matrix(gquad, α, β, D, S)
    N, J = diagonal_ncross_jac_matrix(gquad)
    Jm = diagonalblockmatrix_to_matrix(J.diag)
    n_qnodes = get_number_of_qnodes(gquad)
    M = Matrix{ComplexF64}(undef, 2*n_qnodes, 2*n_qnodes)
    M .= transpose(Jm)*blockmatrix_to_matrix(0.5*α*I + α*D + β*S*N)*Jm
    return M
end