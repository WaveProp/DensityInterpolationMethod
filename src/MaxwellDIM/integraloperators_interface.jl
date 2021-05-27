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

struct _NystromLinearMap{F<:AbstractNystromFormulation}
    dimdata::IndirectDimData
end
const NystromLinearMap = _NystromLinearMap{NoneNystromFormulation}
const InteriorNystromLinearMap = _NystromLinearMap{InteriorNystromFormulation}
const ExteriorNystromLinearMap = _NystromLinearMap{ExteriorNystromFormulation}
function Base.size(iop::_NystromLinearMap) 
    n_qnodes = get_number_of_qnodes(iop.dimdata)
    n_unknowns = DIMENSION2 * n_qnodes  # 2 unknowns per qnode
    return (n_unknowns, n_unknowns)
end
Base.size(iop::_NystromLinearMap, i::Integer) = size(iop)[i] 
Base.eltype(::_NystromLinearMap) = ComplexF64
LinearAlgebra.mul!(y, iop::_NystromLinearMap, b) = y .= evaluate_nystrom_linear_map(iop, b)

function evaluate_nystrom_linear_map(nstruct::_NystromLinearMap{F}, ϕvec::AbstractVector{ComplexF64}) where F
    # copy density into dimdata (FIX: this shouldn't be necessary)
    @assert length(nstruct.dimdata.density_coeff_data) == length(ϕvec)
    copyto!(nstruct.dimdata.density_coeff_data, ϕvec)
    # compute linear map
    compute_density_interpolant!(nstruct.dimdata)
    v = NystromInterpolantOperator(nstruct.dimdata)
    A = _NystromMaxwellOperator{F}(nstruct.dimdata)
    ϕ = nstruct.dimdata.density_coeff
    result = A*ϕ + v
    reinterpreted_result = reinterpret(ComplexF64, result)
    return reinterpreted_result
end

function IterativeSolvers.gmres!(nstruct::_NystromLinearMap, rhs; kwargs...)
    result = zeros(ComplexF64, size(nstruct.dimdata.density_coeff_data))
    IterativeSolvers.gmres!(result, nstruct, rhs; kwargs...) # initially_zero=true
    copyto!(nstruct.dimdata.density_coeff_data, result)
    return nothing
end
