
const MaxwellKernelType = SMatrix{DIMENSION3, DIMENSION3, 
                                  ComplexF64, DIMENSION3*DIMENSION3}

function evaluate_combined_operator(dimdata::DimData, cK, cT, qnode_index_i, qnode_index_j) 
    qnode_i = get_qnode(dimdata.gquad, qnode_index_i)   # qnode i object
    element_index_i = qnode_i.element_index                 # element index of qnode i
    if is_qnode_in_element(dimdata.gquad, qnode_index_j, element_index_i)
        # if qnode_j belongs to element_index_i, return zero
        return zero(MaxwellKernelType)
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
    return wj*(cK*Koperator + cT*Toperator)
end

function evaluate_single_layer_operator(dimdata::DimData, qnode_index_i, qnode_index_j) 
    cK, cT = 0, 1
    return evaluate_combined_operator(dimdata::DimData, cK, cT, qnode_index_i, qnode_index_j) 
end

function evaluate_double_layer_operator(dimdata::DimData, qnode_index_i, qnode_index_j) 
    cK, cT = 1, 0
    return evaluate_combined_operator(dimdata::DimData, cK, cT, qnode_index_i, qnode_index_j) 
end

abstract type IntegralOperator{T} <: AbstractMatrix{T} end
function Base.size(iop::IntegralOperator)
    n_qnodes = get_number_of_qnodes(iop.dimdata)
    return (n_qnodes, n_qnodes)
end
Base.getindex(iop::IntegralOperator, ::Integer, ::Integer) = abstractmethod(iop)

struct SingleLayerOperator <: IntegralOperator{MaxwellKernelType}
    dimdata::IndirectDimData
end
Base.getindex(iop::SingleLayerOperator, i::Integer, j::Integer) = evaluate_single_layer_operator(iop.dimdata, i, j)

struct DoubleLayerOperator <: IntegralOperator{MaxwellKernelType}
    dimdata::IndirectDimData
end
Base.getindex(iop::DoubleLayerOperator, i::Integer, j::Integer) = evaluate_double_layer_operator(iop.dimdata, i, j)

function compute_correction_matrix(dimdata::IndirectDimData)
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
    return Θmatrix
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