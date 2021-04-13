"""
    module Integration

Methods for integrating over instances of [`AbstractReferenceShape`](@ref). 
"""
module Integration

using LinearAlgebra
using StaticArrays

using DensityInterpolationMethod.Utils
using DensityInterpolationMethod.Mesh
import DensityInterpolationMethod.Mesh: getdomain, 
                                        get_number_of_lnodes, 
                                        get_number_of_qnodes,
                                        get_number_of_elements
export 
    # Abstract types
    AbstractQuadratureRule,
    # Types
    # Functions
    integrate,
    integrateflux,
    generate_globalquadrature,
    get_number_of_lnodes,
    get_number_of_qnodes,
    get_number_of_elements


include("quadrature.jl")
include("meshintegration.jl")
include("globalquadrature.jl")
include("gquadintegration.jl")
end