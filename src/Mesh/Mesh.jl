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

export getvertices, getcenter, get_number_of_nodes, getnodes
export getjacobian, getdomain, getmeasure, getnormal, getelementdata, getorder,
       AbstractElement, LagrangeElement, FlatTriangleElement, QuadraticTriangleElement
export GenericMesh

include("lagrangepoly.jl")
include("referenceshapes.jl")
include("element.jl")
include("meshes.jl")
end 