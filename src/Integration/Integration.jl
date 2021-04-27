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
                                        get_number_of_elements
export 
    # Abstract types
    AbstractQuadratureRule,
    # Types
    GlobalQuadrature,
    # Functions
    integrate,
    integrateflux,
    generate_globalquadrature,
    get_number_of_lnodes,
    get_number_of_qnodes,
    get_number_of_elements,
    get_qnode_indices,
    get_element_indices,
    get_element_index,
    get_inelement_qnode_indices,
    get_outelement_qnode_indices,
    get_nodedata_from_element,
    compute_bounding_box

include("quadrature.jl")
include("meshintegration.jl")
include("globalquadrature.jl")
include("gquadintegration.jl")
end