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
import DensityInterpolationMethod.IO: get_qrule_from_gmsh

export 
    # Abstract types
    AbstractQuadratureRule,
    # Types
    QNode,
    GlobalQuadrature,
    # Functions
    integrate,
    integrateflux,
    generate_globalquadrature,
    get_number_of_lnodes,
    get_number_of_qnodes,
    get_number_of_elements,
    get_qnode,
    get_qnodes,
    get_qnode_data,
    get_qnode_indices,
    get_element_indices,
    get_element_index,
    get_inelement_qnode_indices,
    get_outelement_qnode_indices,
    is_qnode_in_element,
    compute_bounding_box

include("quadrature.jl")
include("meshintegration.jl")
include("globalquadrature.jl")
include("gquadintegration.jl")
end