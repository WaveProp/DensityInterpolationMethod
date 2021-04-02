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

export vertices, center, number_of_nodes, getnodes
export jacobian, domain, measure, normal, elementdata, order,
       AbstractElement, FlatTriangleElement, QuadraticTriangleElement
export GenericMesh

include("lagrangepoly.jl")
include("referenceshapes.jl")
include("element.jl")
include("meshes.jl")
end 