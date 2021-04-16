"""
    module MaxwellDIM

Methods for solving Maxwell's equations using Density interpolation
Method.
"""

module MaxwellDIM

using LinearAlgebra
using StaticArrays

using DensityInterpolationMethod.Mesh
using DensityInterpolationMethod.Integration
using DensityInterpolationMethod.Utils

include("lebedev.jl")
include("maxwell.jl")
include("dim.jl")
end