
abstract type AbstractNystromFormulation end
abstract type NoneNystromFormulation <: AbstractNystromFormulation end
abstract type ExteriorNystromFormulation <: AbstractNystromFormulation end
abstract type InteriorNystromFormulation <: AbstractNystromFormulation end

"""
    const NystromFormulationType = Type{<:AbstractNystromFormulation}

Abstract types that defines the evaluation of the Double Layer operator jump
in some functions and structures.
"""
const NystromFormulationType = Type{<:AbstractNystromFormulation}

"""
    assemble_interpolant_correction_matrices!(dimdata::IndirectDimData)

Assembles the matrices `Θᵢ` for each element `i` in `dimdata`. These matrices are used
for efficient evaluations of the operator `V₁`. ('It accounts for the
local kernel-regularization performed around the diagonal (singular) entries by gathering all the
contributions arising from the collocation density interpolants.')
"""
function assemble_interpolant_correction_matrices!(dimdata::IndirectDimData)
    # Assemble auxiliary matrices
    # [B]_{i, l} = γ₀G(yᵢ, zₗ) 
    # [C]_{i, l} = γ₁G(yᵢ, zₗ) 
    Bmatrix, Cmatrix = _compute_correction_matrices_auxiliary_matrices(dimdata)
    # Compute correction matrix
    # TODO: implement fast evaluation of single and double layer operators
    t1 = @elapsed A = convert_operator_to_matrix(DoubleLayerOperator(dimdata))
    t2 = @elapsed Θmatrix = -0.5*Bmatrix - A*Bmatrix
    t3 = @elapsed convert_operator_to_matrix!(A, SingleLayerOperator(dimdata))
    t4 = @elapsed Θmatrix .-= A*Cmatrix
    t5 = @elapsed _compute_correction_matrices_store_matrix!(dimdata, Θmatrix)
    @info "Correction matrix elapsed times (seg)" t1 t2 t3 t4 t5
end
function _compute_correction_matrices_auxiliary_matrices(dimdata::IndirectDimData)
    n_qnodes = get_number_of_qnodes(dimdata)
    n_sources = get_number_of_srcs(dimdata)
    k, _, _ = getparameters(dimdata)   # wavenumber
    Bmatrix = generate_pseudoblockmatrix(MaxwellKernelType, n_qnodes, n_sources)
    Cmatrix = generate_pseudoblockmatrix(MaxwellKernelType, n_qnodes, n_sources)
    for l in 1:n_sources
        zₗ = get_src_node(dimdata, l)
        for i in 1:n_qnodes
            qnodeᵢ = get_qnode(dimdata.gquad, i)
            yᵢ, _, _, nᵢ = get_qnode_data(qnodeᵢ)          
            Bmatrix[Block(i, l)] = single_layer_kernel(yᵢ, zₗ, k, nᵢ)
            Cmatrix[Block(i, l)] = double_layer_kernel(yᵢ, zₗ, k, nᵢ)
        end
    end
    return get_matrix_from_pseudoblockmatrix(Bmatrix), get_matrix_from_pseudoblockmatrix(Cmatrix)
end  
function _compute_correction_matrices_store_matrix!(dimdata::IndirectDimData, Θmatrix)
    # Store rows of Θmatrix in dimdata
    n_qnodes = get_number_of_qnodes(dimdata)
    n_sources = get_number_of_srcs(dimdata)
    @assert size(Θmatrix) == (DIMENSION3*n_qnodes, DIMENSION3*n_sources)
    for i in 1:n_qnodes
        # interpolant correction matrices
        # of qnode i
        i1 = DIMENSION3*(i-1) + 1
        i2 = DIMENSION3*i
        Θᵢ = Θmatrix[i1:i2, :]
        store_interpolant_correction_matrix!(dimdata, Θᵢ, i)
    end
end

"""
    evaluate_single_and_double_layer_operators(dimdata::DimData, formtype::NystromFormulationType, 
                                               qnode_index_i, qnode_index_j)::Tuple{MaxwellKernelType,MaxwellKernelType}

Evaluates and returns the entry of the (discretized) Maxwell's single and double layer operators `(T_ij, K_ij)`
for qnodes indices `i` and `j`. These entries do not include the self-element (singular) contributions.
"""
function evaluate_single_and_double_layer_operators(dimdata::DimData, formtype::NystromFormulationType, 
                                                    qnode_index_i, qnode_index_j)::Tuple{MaxwellKernelType,MaxwellKernelType}
    qnode_i = get_qnode(dimdata.gquad, qnode_index_i)   # qnode i object
    element_index_i = qnode_i.element_index             # element index of qnode i
    if qnode_index_i == qnode_index_j
        # the single layer operator (Toperator) is zero
        # and the quadrature weigth (wj) is not used (set to one)
        Toperator = zero(MaxwellKernelType)
        wj = 1
        # the double layer operator (Koperator) depends on
        # the formulation type (None, Exterior, Interior)
        if formtype == NoneNystromFormulation
            Koperator = zero(MaxwellKernelType)
        elseif formtype == ExteriorNystromFormulation
            Koperator = MaxwellKernelType(0.5*I)
        elseif formtype == InteriorNystromFormulation
            Koperator = MaxwellKernelType(-0.5*I)
        end
    elseif is_qnode_in_element(dimdata.gquad, qnode_index_j, element_index_i)
        # both the single layer operator (Toperator) and the
        # double layer operator (Koperator) are zero
        # and the quadrature weigth (wj) is not relevant (set to one)
        Toperator = zero(MaxwellKernelType)
        Koperator = zero(MaxwellKernelType)
        wj = 1
    else
        # wavenumber
        k, _, _ = getparameters(dimdata)    
        # qnode i data
        yi, _, _, ni = get_qnode_data(qnode_i)
        # qnode j data
        qnode_j = get_qnode(dimdata.gquad, qnode_index_j)  
        yj, wj, _, _ = get_qnode_data(qnode_j)  
        # single layer operator
        Toperator = single_layer_kernel(yi, yj, k, ni)   
        # double layer operator
        Koperator = double_layer_kernel(yi, yj, k, ni)   
    end
    # return operators weighted by quadrature weight
    return wj*Toperator, wj*Koperator  
end

"""
    evaluate_single_layer_operator(dimdata::DimData, qnode_index_i, qnode_index_j)::MaxwellKernelType 

Evaluates and returns the entry of the (discretized) Maxwell's single operator `T_ij`
for qnodes indices `i` and `j`.
"""
function evaluate_single_layer_operator(dimdata::DimData, qnode_index_i, qnode_index_j)::MaxwellKernelType 
    SL, _ = evaluate_single_and_double_layer_operators(dimdata, NoneNystromFormulation, 
                                                       qnode_index_i, qnode_index_j) 
    return SL
end

"""
    evaluate_double_layer_operator(dimdata::DimData, formtype::NystromFormulationType, 
                                   qnode_index_i, qnode_index_j)::MaxwellKernelType  

Evaluates and returns the entry of the (discretized) Maxwell's double operator `K_ij`
for qnodes indices `i` and `j`.
"""
function evaluate_double_layer_operator(dimdata::DimData, formtype::NystromFormulationType, 
                                        qnode_index_i, qnode_index_j)::MaxwellKernelType  
    _, DL = evaluate_single_and_double_layer_operators(dimdata, formtype, 
                                                       qnode_index_i, qnode_index_j) 
    return DL
end

"""
    evaluate_combined_layer_operator(dimdata::IndirectDimData, formtype::NystromFormulationType,
                                     qnode_index_i, qnode_index_j)::ReducedMaxwellKernelType

Evaluates and returns the entry of the (discretized) Maxwell's combined layer operator `C_ij`
for qnodes indices `i` and `j`. 
"""
function evaluate_combined_layer_operator(dimdata::IndirectDimData, formtype::NystromFormulationType, 
                                          qnode_index_i, qnode_index_j)::ReducedMaxwellKernelType
    _, α, β = getparameters(dimdata)
    SL, DL = evaluate_single_and_double_layer_operators(dimdata, formtype, qnode_index_i, qnode_index_j) 
    # qnode j data
    qnode_j = get_qnode(dimdata.gquad, qnode_index_j)  
    _, _, jacⱼ, nⱼ = get_qnode_data(qnode_j)
    nⱼcross = cross_product_matrix(nⱼ)
    CL = (α*DL + β*SL*nⱼcross)*jacⱼ
    return CL
end

"""
    evaluate_nystrom_maxwell_operator(dimdata::IndirectDimData, formtype::NystromFormulationType, 
                                      qnode_index_i, qnode_index_j)::ReducedReducedMaxwellKernelType

Evaluates and returns the entry of the (discretized) Maxwell's combined layer operator `N_ij`
used in the Nystrom integral equation, for qnodes indices `i` and `j`. 
"""
function evaluate_nystrom_maxwell_operator(dimdata::IndirectDimData, formtype::NystromFormulationType, 
                                           qnode_index_i, qnode_index_j)::ReducedReducedMaxwellKernelType
    # qnode i data
    qnode_i = get_qnode(dimdata.gquad, qnode_index_i)  
    _, _, jacᵢ, _ = get_qnode_data(qnode_i) 
    # combined layer operator
    CL = evaluate_combined_layer_operator(dimdata, formtype, qnode_index_i, qnode_index_j) 
    return transpose(jacᵢ)*CL
end

"""
    evaluate_interpolant_forwardmap(dimdata::IndirectDimData, qnode_index_i)::ComplexPoint3D

Evaluates and returns the entry of the map `[V₁ϕ]ᵢ` for qnode index `i`, where `V₁` is 
the interpolant (correction) operator. This function assumes that `assemble_interpolant_correction_matrices!`
and `compute_density_interpolant!` have already been called.
"""
function evaluate_interpolant_forwardmap(dimdata::IndirectDimData, qnode_index_i)::ComplexPoint3D
    qnode_i = get_qnode(dimdata.gquad, qnode_index_i)   # qnode i object
    element_index_i = qnode_i.element_index             # element index of qnode i
    # interpolant coefficients of element i
    interpolant_coeff = get_interpolant_coeff(dimdata, element_index_i) 
    # interpolant correction matrices of qnode i
    Θᵢ = get_interpolant_correction_matrices(dimdata, qnode_index_i)
    @assert length(Θᵢ) == length(interpolant_coeff)     # sanity check
    return sum(eachindex(interpolant_coeff)) do l
        Θᵢ[l] * interpolant_coeff[l]
    end
end

"""
    evaluate_nystrom_interpolant_forwardmap(dimdata::IndirectDimData, qnode_index_i)::ComplexPoint2D

Evaluates and returns the entry of the map `[V₁ϕ]ᵢ`, used in the Nystrom integral equation, 
for qnode index `i`, where `V₁` is the interpolant (correction) operator. This function assumes 
that `assemble_interpolant_correction_matrices!` and `compute_density_interpolant!` have already 
been called.
"""
function evaluate_nystrom_interpolant_forwardmap(dimdata::IndirectDimData, qnode_index_i)::ComplexPoint2D
    # qnode i data
    qnode_i = get_qnode(dimdata.gquad, qnode_index_i)  
    _, _, jacᵢ, _ = get_qnode_data(qnode_i) 
    # interpolant forward map
    interpolant_forwardmap_i = evaluate_interpolant_forwardmap(dimdata, qnode_index_i)
    return transpose(jacᵢ)*interpolant_forwardmap_i
end

"""
    generate_interpolant_forwardmap_matrix(dimdata::IndirectDimData)

Assembles the operator `V₁` in (sparse) matrix form, used in the Nystrom integral equation, 
where `V₁` is the interpolant (correction) operator.
"""
function generate_interpolant_forwardmap_matrix(dimdata::IndirectDimData; blockmatrix=true)
    # construct sparse array
    I = Int64[]
    J = Int64[]
    if blockmatrix
        V = ReducedReducedMaxwellKernelType[]
        n_unknowns = get_number_of_qnodes(dimdata) 
        # function for storing matrix entries
        store_entry! = _generate_interpolant_forwardmap_matrix_store_entry_block!
    else
        V = ComplexF64[]
        n_unknowns = DIMENSION2 * get_number_of_qnodes(dimdata) 
        # function for storing scalar entries
        store_entry! = _generate_interpolant_forwardmap_matrix_store_entry!
    end
    for element_index in get_element_indices(dimdata.gquad)
        inelement_qnode_indices = get_inelement_qnode_indices(dimdata.gquad, element_index)
        element_matrix = _generate_interpolant_forwardmap_matrix_element_matrix(dimdata, element_index, inelement_qnode_indices)
        for i in inelement_qnode_indices
            qnode_matrices = _generate_interpolant_forwardmap_matrix_qnode_matrices(dimdata, element_matrix, i)
            for (j, qnode_matrix) in zip(inelement_qnode_indices, qnode_matrices)
                store_entry!(I, J, V, i, j, qnode_matrix)
            end
        end
    end
    return sparse(I, J, V, n_unknowns, n_unknowns)
end
function _generate_interpolant_forwardmap_matrix_element_matrix(dimdata::IndirectDimData, element_index, inelement_qnode_indices)
    # generate element matrix
    n_qnodes = length(inelement_qnode_indices)   # number of qnodes in element
    Lmatrix = dimdata.Lmatrices[element_index]
    Qmatrix = dimdata.Qmatrices[element_index]
    _, α, β = getparameters(dimdata)
    n_unknowns = DIMENSION2*n_qnodes  # number of unknowns = 2 per qnode
    Dαβ = [α*I(n_unknowns); β*I(n_unknowns)]  
    jacobians = map(inelement_qnode_indices) do qnode_index
        qnode = get_qnode(dimdata.gquad, qnode_index)
        _, _, jac, _ = get_qnode_data(qnode)
        transpose(jac)*jac
    end
    Jmatrix = diagonalblockmatrix_to_matrix(jacobians) 
    element_matrix = adjoint(Qmatrix) * (Lmatrix \ (Dαβ * Jmatrix))
    return element_matrix
end
function _generate_interpolant_forwardmap_matrix_qnode_matrices(dimdata::IndirectDimData, element_matrix, qnode_index)
    # generate qnode matrix
    qnodeᵢ = get_qnode(dimdata.gquad, qnode_index)
    _, _, jacᵢ, _ = get_qnode_data(qnodeᵢ) # jacobian
    Θᵢmatrix = get_interpolant_correction_matrix(dimdata, qnode_index)
    qnode_matrices = transpose(jacᵢ) * Θᵢmatrix * element_matrix
    # convert the full matrix into a list of smaller matrices
    return reinterpret(ReducedReducedMaxwellKernelType, @view qnode_matrices[:])
end
function _generate_interpolant_forwardmap_matrix_store_entry_block!(I, J, V, i, j, vmatrix::ReducedReducedMaxwellKernelType)
    # store the full matrix
    push!(I, i)
    push!(J, j)
    push!(V, vmatrix)
end
function _generate_interpolant_forwardmap_matrix_store_entry!(I, J, V, i, j, vmatrix::ReducedReducedMaxwellKernelType)
    # store each entry of the matrix
    for m in 1:2
        jglobal = 2*(j-1) + m
        for n in 1:2
            iglobal = 2*(i-1) + n
            push!(I, iglobal)
            push!(J, jglobal)
            push!(V, vmatrix[n, m])
        end
    end
end