abstract type AbstractInteriorExteriorFormulation end
abstract type NoneFormulation <: AbstractInteriorExteriorFormulation end
abstract type ExteriorFormulation <: AbstractInteriorExteriorFormulation end
abstract type InteriorFormulation <: AbstractInteriorExteriorFormulation end
const FormulationType = Type{<:AbstractInteriorExteriorFormulation}

function evaluate_single_and_double_layer_operators(dimdata::DimData, formtype::FormulationType, 
                                                     qnode_index_i, qnode_index_j) 
    qnode_i = get_qnode(dimdata.gquad, qnode_index_i)   # qnode i object
    element_index_i = qnode_i.element_index             # element index of qnode i
    if is_qnode_in_element(dimdata.gquad, qnode_index_j, element_index_i)
        # if qnode j doesn't belong to the element of qnode i,
        # then the single layer operator (Toperator) is zero
        Toperator = zero(MaxwellKernelType)
        if formtype == NoneFormulation
            # the double layer operator (Koperator) is zero
            # and the quadrature weigth is not relevant
            Koperator = zero(MaxwellKernelType)
            wj = 1
        else
            # retrieve quadrature weigth from qnode j,
            # the double layer operator (Koperator) is ±0.5I,
            # depending on whether the problem is exterior or interior
            qnode_j = get_qnode(dimdata.gquad, qnode_index_j) 
            _, wj, _, _ = get_qnode_data(qnode_j)
            if formtype == ExteriorFormulation
                Koperator = MaxwellKernelType(0.5*I)
            elseif formtype == InteriorFormulation
                Koperator = MaxwellKernelType(-0.5*I)
            end
        end
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
    SL, _ = evaluate_single_and_double_layer_operators(dimdata, NoneFormulation, qnode_index_i, qnode_index_j) 
    return SL
end

function evaluate_double_layer_operator(dimdata::DimData, formtype::FormulationType, qnode_index_i, qnode_index_j) 
    _, DL = evaluate_single_and_double_layer_operators(dimdata, formtype, qnode_index_i, qnode_index_j) 
    return DL
end

function evaluate_combined_layer_operator(dimdata::DimData, formtype::FormulationType, qnode_index_i, qnode_index_j) 
    _, α, β = getparameters(dimdata)
    SL, DL = evaluate_single_and_double_layer_operators(dimdata, formtype, qnode_index_i, qnode_index_j) 
    # qnode j data
    qnode_j = get_qnode(dimdata.gquad, qnode_index_j)  
    _, _, jacⱼ, nⱼ = get_qnode_data(qnode_j)
    nⱼcross = cross_product_matrix(nⱼ)
    CL = (α*DL + β*SL*nⱼcross)*jacⱼ
    return CL
end

function evaluate_nystrom_maxwell_operator(dimdata::DimData, formtype::FormulationType, qnode_index_i, qnode_index_j)
    # combined layer operator
    CL = evaluate_combined_layer_operator(dimdata, formtype, qnode_index_i, qnode_index_j) 
    # qnode i data
    qnode_i = get_qnode(dimdata.gquad, qnode_index_i)  
    _, _, jacᵢ, _ = get_qnode_data(qnode_i) 
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

struct SingleLayerOperator <: AbstractIntegralMatrixOperator{MaxwellKernelType}
    dimdata::IndirectDimData
end
Base.getindex(iop::SingleLayerOperator, i::Integer, j::Integer) = evaluate_single_layer_operator(iop.dimdata, i, j)

struct _DoubleLayerOperator{F<:AbstractInteriorExteriorFormulation} <: AbstractIntegralMatrixOperator{MaxwellKernelType}
    dimdata::IndirectDimData
end
const DoubleLayerOperator = _DoubleLayerOperator{NoneFormulation}
const ExteriorDoubleLayerOperator = _DoubleLayerOperator{ExteriorFormulation}
const InteriorDoubleLayerOperator = _DoubleLayerOperator{InteriorFormulation}
Base.getindex(iop::_DoubleLayerOperator{F}, i::Integer, j::Integer) where F = evaluate_double_layer_operator(iop.dimdata, F, i, j)

struct _CombinedLayerOperator{F<:AbstractInteriorExteriorFormulation} <: AbstractIntegralMatrixOperator{ReducedMaxwellKernelType}
    dimdata::IndirectDimData
end
const CombinedLayerOperator = _CombinedLayerOperator{NoneFormulation}
const ExteriorCombinedLayerOperator = _CombinedLayerOperator{ExteriorFormulation}
const InteriorCombinedLayerOperator = _CombinedLayerOperator{InteriorFormulation}
Base.getindex(iop::_CombinedLayerOperator{F}, i::Integer, j::Integer) where F = evaluate_combined_layer_operator(iop.dimdata, F, i, j)

struct _NystromMaxwellOperator{F<:AbstractInteriorExteriorFormulation} <: AbstractIntegralMatrixOperator{ReducedReducedMaxwellKernelType}
    dimdata::IndirectDimData
end
const NystromMaxwellOperator = _NystromMaxwellOperator{NoneFormulation}
const InteriorNystromMaxwellOperator = _NystromMaxwellOperator{InteriorFormulation}
const ExteriorNystromMaxwellOperator = _NystromMaxwellOperator{ExteriorFormulation}
Base.getindex(iop::_NystromMaxwellOperator{F}, i::Integer, j::Integer) where F = evaluate_nystrom_maxwell_operator(iop.dimdata, F, i, j)

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