abstract type AbstractNystromFormulation end
abstract type NoneNystromFormulation <: AbstractNystromFormulation end
abstract type ExteriorNystromFormulation <: AbstractNystromFormulation end
abstract type InteriorNystromFormulation <: AbstractNystromFormulation end
const NystromFormulationType = Type{<:AbstractNystromFormulation}

function evaluate_single_and_double_layer_operators(dimdata::DimData, formtype::NystromFormulationType, 
                                                     qnode_index_i, qnode_index_j) 
    qnode_i = get_qnode(dimdata.gquad, qnode_index_i)   # qnode i object
    element_index_i = qnode_i.element_index             # element index of qnode i
    if qnode_index_i == qnode_index_j
        # the single layer operator (Toperator) is zero
        # and the quadrature weigth (wj) is not used (set to unity)
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
        # and the quadrature weigth (wj) is not relevant (set to unity)
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

function evaluate_single_layer_operator(dimdata::DimData, qnode_index_i, qnode_index_j) 
    SL, _ = evaluate_single_and_double_layer_operators(dimdata, NoneNystromFormulation, qnode_index_i, qnode_index_j) 
    return SL
end

function evaluate_double_layer_operator(dimdata::DimData, formtype::NystromFormulationType, qnode_index_i, qnode_index_j) 
    _, DL = evaluate_single_and_double_layer_operators(dimdata, formtype, qnode_index_i, qnode_index_j) 
    return DL
end

function evaluate_combined_layer_operator(dimdata::DimData, formtype::NystromFormulationType, qnode_index_i, qnode_index_j) 
    _, α, β = getparameters(dimdata)
    SL, DL = evaluate_single_and_double_layer_operators(dimdata, formtype, qnode_index_i, qnode_index_j) 
    # qnode j data
    qnode_j = get_qnode(dimdata.gquad, qnode_index_j)  
    _, _, jacⱼ, nⱼ = get_qnode_data(qnode_j)
    nⱼcross = cross_product_matrix(nⱼ)
    CL = (α*DL + β*SL*nⱼcross)*jacⱼ
    return CL
end

function evaluate_nystrom_maxwell_operator(dimdata::DimData, formtype::NystromFormulationType, qnode_index_i, qnode_index_j)
    # qnode i data
    qnode_i = get_qnode(dimdata.gquad, qnode_index_i)  
    _, _, jacᵢ, _ = get_qnode_data(qnode_i) 
    # combined layer operator
    CL = evaluate_combined_layer_operator(dimdata, formtype, qnode_index_i, qnode_index_j) 
    return transpose(jacᵢ)*CL
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

function evaluate_nystrom_interpolant_forwardmap(dimdata::DimData, qnode_index_i)
    # qnode i data
    qnode_i = get_qnode(dimdata.gquad, qnode_index_i)  
    _, _, jacᵢ, _ = get_qnode_data(qnode_i) 
    # interpolant forward map
    interpolant_forwardmap_i = evaluate_interpolant_forwardmap(dimdata, qnode_index_i)
    return transpose(jacᵢ)*interpolant_forwardmap_i
end

abstract type AbstractIntegralOperator{T, N} <: AbstractArray{T, N} end
function Base.size(iop::AbstractIntegralOperator{T, N}) where {T, N}
    n_qnodes = get_number_of_qnodes(iop.dimdata)
    return ntuple(_ -> n_qnodes, N)
end
abstract type AbstractIntegralVectorOperator{T} <: AbstractIntegralOperator{T, 1} end
abstract type AbstractIntegralMatrixOperator{T} <: AbstractIntegralOperator{T, 2} end

struct InterpolantOperator <: AbstractIntegralVectorOperator{ComplexPoint3D}
    dimdata::IndirectDimData
end
Base.getindex(iop::InterpolantOperator, i::Integer) = evaluate_interpolant_forwardmap(iop.dimdata, i)

struct NystromInterpolantOperator <: AbstractIntegralVectorOperator{ComplexPoint2D}
    dimdata::IndirectDimData
end
Base.getindex(iop::NystromInterpolantOperator, i::Integer) = evaluate_nystrom_interpolant_forwardmap(iop.dimdata, i)

struct SingleLayerOperator <: AbstractIntegralMatrixOperator{MaxwellKernelType}
    dimdata::IndirectDimData
end
Base.getindex(iop::SingleLayerOperator, i::Integer, j::Integer) = evaluate_single_layer_operator(iop.dimdata, i, j)

struct _DoubleLayerOperator{F<:AbstractNystromFormulation} <: AbstractIntegralMatrixOperator{MaxwellKernelType}
    dimdata::IndirectDimData
end
const DoubleLayerOperator = _DoubleLayerOperator{NoneNystromFormulation}
const ExteriorDoubleLayerOperator = _DoubleLayerOperator{ExteriorNystromFormulation}
const InteriorDoubleLayerOperator = _DoubleLayerOperator{InteriorNystromFormulation}
Base.getindex(iop::_DoubleLayerOperator{F}, i::Integer, j::Integer) where F = evaluate_double_layer_operator(iop.dimdata, F, i, j)

struct _CombinedLayerOperator{F<:AbstractNystromFormulation} <: AbstractIntegralMatrixOperator{ReducedMaxwellKernelType}
    dimdata::IndirectDimData
end
const CombinedLayerOperator = _CombinedLayerOperator{NoneNystromFormulation}
const ExteriorCombinedLayerOperator = _CombinedLayerOperator{ExteriorNystromFormulation}
const InteriorCombinedLayerOperator = _CombinedLayerOperator{InteriorNystromFormulation}
Base.getindex(iop::_CombinedLayerOperator{F}, i::Integer, j::Integer) where F = evaluate_combined_layer_operator(iop.dimdata, F, i, j)

struct _NystromMaxwellOperator{F<:AbstractNystromFormulation} <: AbstractIntegralMatrixOperator{ReducedReducedMaxwellKernelType}
    dimdata::IndirectDimData
end
const NystromMaxwellOperator = _NystromMaxwellOperator{NoneNystromFormulation}
const InteriorNystromMaxwellOperator = _NystromMaxwellOperator{InteriorNystromFormulation}
const ExteriorNystromMaxwellOperator = _NystromMaxwellOperator{ExteriorNystromFormulation}
Base.getindex(iop::_NystromMaxwellOperator{F}, i::Integer, j::Integer) where F = evaluate_nystrom_maxwell_operator(iop.dimdata, F, i, j)

function compute_correction_matrices!(dimdata::IndirectDimData)
    n_qnodes = get_number_of_qnodes(dimdata)
    n_sources = get_number_of_srcs(dimdata)
    # Assemble auxiliary matrices
    # [B]_{i, l} = γ₀G(yᵢ, zₗ) 
    # [C]_{i, l} = γ₁G(yᵢ, zₗ) 
    Bmatrix = Matrix{MaxwellKernelType}(undef, n_qnodes, n_sources)
    Cmatrix = Matrix{MaxwellKernelType}(undef, n_qnodes, n_sources)
    _compute_correction_matrices_auxiliary_matrices!(dimdata, Bmatrix, Cmatrix)
    # Single and double layer operators
    T = SingleLayerOperator(dimdata)
    K = DoubleLayerOperator(dimdata)
    # Compute correction matrix
    Θmatrix = -0.5*Bmatrix - K*Bmatrix - T*Cmatrix
    _compute_correction_matrices_store_matrix!(dimdata, Θmatrix)
end
function _compute_correction_matrices_auxiliary_matrices!(dimdata::IndirectDimData, Bmatrix, Cmatrix)
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
function _compute_correction_matrices_store_matrix!(dimdata::IndirectDimData, Θmatrix)
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