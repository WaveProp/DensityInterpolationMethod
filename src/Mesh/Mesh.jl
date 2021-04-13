"""
    module Mesh

Mesh data structures and interface.
"""
module Mesh

using LinearAlgebra
using StaticArrays
import OrderedCollections: OrderedDict

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
    get_number_of_lnodes,
    get_lnodes,
    getjacobian,
    evaluate_and_getjacobian,
    getdomain,
    getmeasure,
    getnormal,
    getelementdata,
    getorder,
    getelements,
    get_number_of_elements,
    get_etypes_and_elements

include("lagrangepoly.jl")
include("referenceshapes.jl")
include("element.jl")
include("meshes.jl")
end 