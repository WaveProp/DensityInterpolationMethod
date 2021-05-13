"""
    Definitions of quadratures rules for various AbstractReferenceShape.
"""

"""
    abstract type AbstractQuadratureRule{D<:AbstractReferenceShape}

A quadrature rule for integrating a function over the domain `D`.

An instance `q` of `AbstractQuadratureRule{D}` is expected to implement the
following methods:

- `get_qnodes_and_qweights(q)` : return the nodes `x` and weights `w` of the quadrature rule on the
  reference domain `D`. For performance reasons, the result shoudl depend only
  on the type of `q`.
- `get_qnodes_and_qweights(q,el)` : return the nodes `x` and weights `w` of the quadrature rule on the
  element `D`. This assumes that `domain(q)==domain(el)`, so that the element
  quadrature can be computed by *pushing forward* a reference quadrature to `el`.
"""
abstract type AbstractQuadratureRule{D<:AbstractReferenceShape} end

getdomain(q::AbstractQuadratureRule{D}) where {D} = D()

"""
    get_qnodes_and_qweights(q::AbstractQuadratureRule)

Return the quadrature nodes `x` and weights `w` on the `domain(q)`.
"""
function get_qnodes_and_qweights(q::AbstractQuadratureRule)
    abstractmethod(typeof(q))
end

"""
    getqnodes(Y)

Return the quadrature nodes associated with `Y`.
"""
getqnodes(q::AbstractQuadratureRule) = get_qnodes_and_qweights(q)[1]

"""
    getqweights(Y)

Return the quadrature weights associated with `Y`.
"""
getqweights(q::AbstractQuadratureRule) = get_qnodes_and_qweights(q)[2]


"""
    struct GaussQuadrature{D, O} <: AbstractQuadratureRule{D}

Tabulated symmetric Gauss quadrature rule of order `O` for integration over `D`.
This is currently implemented on triangles by calling the Gmsh API.
"""
struct GaussQuadrature{D, O} <: AbstractQuadratureRule{D} 
    GaussQuadrature(ref::Type{AbstractReferenceShape}, order) = new{ref, order}()
    GaussQuadrature(ref, order) = new{typeof(ref), order}()
end

"""
    get_number_of_qnodes(q::GaussQuadrature)

Returns the total number of quadrature nodes (for a single element) 
associated with the quadrature rule `q`.
"""
function get_number_of_qnodes(q::GaussQuadrature)
    _, w = get_qnodes_and_qweights(q)
    return length(w)
end

"""
    get_qrule_for_reference_shape(ref, order)

Given a `ref`erence shape and a desired quadrature `order`, return
an appropiate quadrature rule.
"""
function get_qrule_for_reference_shape(ref, order)
    if ref isa ReferenceTriangle
        # This orders contains points outside the
        # reference triangle, so they can't be used
        forbidden_orders = SVector(11, 15, 16, 18, 20)
        if order ∉ forbidden_orders
            return GaussQuadrature(ref, order)
        end
    end
    error("no appropriate quadrature rule found.")
end

"""
    get_qrule_for_element(E, order)

Given an element type `E`, return an appropriate quadrature of order `order`.
"""
function get_qrule_for_element(E, order)
    return get_qrule_for_reference_shape(getdomain(E), order)
end

@generated function 
    get_qnodes_and_qweights(q::GaussQuadrature{<:ReferenceTriangle, O}) where {O}
    element_name = "Triangle"
    qrule_name = "Gauss$O"
    x, w = get_qrule_from_gmsh(element_name, qrule_name)
    @assert (length(x) == length(w)) && (length(x) > 0)
    return :($x,$w)
end
