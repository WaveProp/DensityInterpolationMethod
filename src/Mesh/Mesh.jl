"""
    module Mesh

Mesh data structures and interface.
"""
module Mesh

using LinearAlgebra
using StaticArrays
using OrderedCollections

# for evaluating and differentiating Lagrange polynomials
using StaticPolynomials       
import DynamicPolynomials: @polyvar  
                            
using DensityInterpolationMethod.Utils

export 
    # Abstract types
    AbstractReferenceShape,
    AbstractElement,
    # Types
    LagrangeElement,
    ReferenceTriangle,
    FlatTriangleElement,
    QuadraticTriangleElement,
    GenericMesh,
    # Functions
    getvertices,
    getcenter,
    get_number_of_nodes,
    getnodes,
    getjacobian,
    getdomain,
    getmeasure,
    getnormal,
    getelementdata,
    getorder

include("lagrangepoly.jl")
include("referenceshapes.jl")
include("element.jl")
include("meshes.jl")
end 