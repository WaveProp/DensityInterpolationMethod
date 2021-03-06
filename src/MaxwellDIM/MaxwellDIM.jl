"""
    module MaxwellDIM

Methods for solving Maxwell's equations using Density interpolation
Method.
"""

module MaxwellDIM

using LinearAlgebra
using SparseArrays
using StaticArrays
using BlockArrays
using IterativeSolvers
using Lebedev

# for parametric surfaces and NystromMesh
import WavePropBase: Geometry.Domain, boundary
import ParametricSurfaces: meshgen, Sphere
import Nystrom

using DensityInterpolationMethod.Mesh
using DensityInterpolationMethod.Integration
using DensityInterpolationMethod.Utils
import DensityInterpolationMethod.Integration: get_number_of_qnodes

export 
    # Abstract types
    # Types
    # Functions
    generate_dimdata

include("blockmatrices.jl")
include("lebedev.jl")
include("maxwell.jl")
include("dimdata.jl")
include("dim.jl")
include("dim_direct.jl")
include("integraloperators.jl")
include("integraloperators_interface.jl")
include("linalg.jl")
include("dim_waveprop.jl")
include("parametricsurfaces.jl")
end