"""
    module Integration

Methods for integrating over instances of [`AbstractReferenceShape`](@ref). 
"""
module Integration

using DensityInterpolationMethod.Utils
using DensityInterpolationMethod.Mesh

export 
    # Abstract types
    AbstractQuadratureRule

include("quadrature.jl")
end