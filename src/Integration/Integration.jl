"""
    module Integration

Methods for integrating over instances of [`AbstractReferenceShape`](@ref). 
"""
module Integration

using LinearAlgebra
using StaticArrays

using DensityInterpolationMethod.Utils
using DensityInterpolationMethod.Mesh
import DensityInterpolationMethod.Mesh: getdomain

export 
    # Abstract types
    AbstractQuadratureRule,
    # Types
    # Functions
    integrate,
    integrateflux

include("quadrature.jl")
include("meshintegration.jl")
end