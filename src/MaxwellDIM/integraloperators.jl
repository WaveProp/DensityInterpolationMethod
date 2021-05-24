function evaluate_single_and_double_layer_operators(dimdata::DimData, qnode_index_i, qnode_index_j) 
    qnode_i = get_qnode(dimdata.gquad, qnode_index_i)   # qnode i object
    element_index_i = qnode_i.element_index             # element index of qnode i
    if is_qnode_in_element(dimdata.gquad, qnode_index_j, element_index_i)
        # if qnode_j belongs to element_index_i, return zero
        return zero(MaxwellKernelType), zero(MaxwellKernelType)
    end
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
    return wj*Toperator, wj*Koperator
end

function evaluate_single_layer_operator(dimdata::DimData, qnode_index_i, qnode_index_j) 
    SL, _ = evaluate_single_and_double_layer_operators(dimdata, qnode_index_i, qnode_index_j) 
    return SL
end

function evaluate_double_layer_operator(dimdata::DimData, qnode_index_i, qnode_index_j) 
    _, DL = evaluate_single_and_double_layer_operators(dimdata, qnode_index_i, qnode_index_j) 
    return DL
end

function evaluate_combined_layer_operator(dimdata::DimData, qnode_index_i, qnode_index_j) 
    _, α, β = getparameters(dimdata)
    SL, DL = evaluate_single_and_double_layer_operators(dimdata, qnode_index_i, qnode_index_j) 
    # qnode j data
    qnode_j = get_qnode(dimdata.gquad, qnode_index_j)  
    _, _, jacⱼ, nⱼ = get_qnode_data(qnode_j)
    nⱼcross = cross_product_matrix(nⱼ)
    CL = (α*DL + β*SL*nⱼcross)*jacⱼ
    return CL
end

function evaluate_interpolant_forwardmap(dimdata::DimData, qnode_index_i)
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

abstract type AbstractIntegralOperator{T, N} <: AbstractArray{T, N} end
function Base.size(iop::AbstractIntegralOperator{T, N}) where {T, N}
    n_qnodes = get_number_of_qnodes(iop.dimdata)
    return ntuple(_ -> n_qnodes, N)
end

abstract type AbstractIntegralVectorOperator{T} <: AbstractIntegralOperator{T, 1} end
abstract type AbstractIntegralMatrixOperator{T} <: AbstractIntegralOperator{T, 2} end

struct SingleLayerOperator <: AbstractIntegralMatrixOperator{MaxwellKernelType}
    dimdata::IndirectDimData
end
Base.getindex(iop::SingleLayerOperator, i::Integer, j::Integer) = evaluate_single_layer_operator(iop.dimdata, i, j)

struct DoubleLayerOperator <: AbstractIntegralMatrixOperator{MaxwellKernelType}
    dimdata::IndirectDimData
end
Base.getindex(iop::DoubleLayerOperator, i::Integer, j::Integer) = evaluate_double_layer_operator(iop.dimdata, i, j)

struct CombinedLayerOperator <: AbstractIntegralMatrixOperator{ReducedMaxwellKernelType}
    dimdata::IndirectDimData
end
Base.getindex(iop::CombinedLayerOperator, i::Integer, j::Integer) = evaluate_combined_layer_operator(iop.dimdata, i, j)

struct InterpolantOperator <: AbstractIntegralVectorOperator{ComplexPoint3D}
    dimdata::IndirectDimData
end
Base.getindex(iop::InterpolantOperator, i::Integer) = evaluate_interpolant_forwardmap(iop.dimdata, i)

function compute_correction_matrix!(dimdata::IndirectDimData)
    n_qnodes = get_number_of_qnodes(dimdata)
    n_sources = get_number_of_srcs(dimdata)
    # Assemble auxiliary matrices
    # [B]_{i, l} = γ₀G(yᵢ, zₗ) 
    # [C]_{i, l} = γ₁G(yᵢ, zₗ) 
    Bmatrix = Matrix{MaxwellKernelType}(undef, n_qnodes, n_sources)
    Cmatrix = Matrix{MaxwellKernelType}(undef, n_qnodes, n_sources)
    _compute_correction_matrix_auxiliary_matrices!(dimdata, Bmatrix, Cmatrix)
    # Single and double layer operators
    T = SingleLayerOperator(dimdata)
    K = DoubleLayerOperator(dimdata)
    # Compute correction matrix
    Θmatrix = -0.5*Bmatrix - K*Bmatrix - T*Cmatrix
    _compute_correction_matrix_store_matrix!(dimdata, Θmatrix)
end
function _compute_correction_matrix_auxiliary_matrices!(dimdata::IndirectDimData, Bmatrix, Cmatrix)
    n_qnodes = get_number_of_qnodes(dimdata)
    n_sources = get_number_of_srcs(dimdata)
    k, _, _ = getparameters(dimdata)   # wavenumber
    for l in 1:n_sources
        zₗ = dimdata.src_list[l]
        for i in 1:n_qnodes
            qnodeᵢ = get_qnode(dimdata.gquad, i)
            yᵢ, _, _, nᵢ = get_qnode_data(qnodeᵢ)          
            Bmatrix[i, l] = single_layer_kernel(yᵢ, zₗ, k, nᵢ)
            Cmatrix[i, l] = double_layer_kernel(yᵢ, zₗ, k, nᵢ)
        end
    end
end  
function _compute_correction_matrix_store_matrix!(dimdata::IndirectDimData, Θmatrix)
    # Store rows of Θmatrix in dimdata
    n_qnodes = get_number_of_qnodes(dimdata)
    n_sources = get_number_of_srcs(dimdata)
    for i in 1:n_qnodes
        # interpolant correction matrices
        # of qnode i
        Θᵢ = get_interpolant_correction_matrices(dimdata, i)
        for l in 1:n_sources
            Θᵢ[l] = Θmatrix[i, l]
        end
    end
end