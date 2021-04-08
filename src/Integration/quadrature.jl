"""
    Definitions of quadratures rules for various AbstractReferenceShape.
"""

"""
    abstract type AbstractQuadratureRule{D<:AbstractReferenceShape}

A quadrature rule for integrating a function over the domain `D`.

An instance `q` of `AbstractQuadratureRule{D}` is expected to implement the
following methods:

- `q()` : return the nodes `x` and weights `w` of the quadrature rule on the
  reference domain `D`. For performance reasons, the result shoudl depend only
  on the type of `q`.
- `q(el)` : return the nodes `x` and weights `w` of the quadrature rule on the
  elemenent `D`. This assumes that `domain(q)==domain(el)`, so that the element
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
    struct GaussQuadrature{D, N} <: AbstractQuadratureRule{D}

Tabulated `N`-point symmetric Gauss quadrature rule for integration over `D`.
This is currently implemented *by hand* for low values of `N` on triangles.
"""
struct GaussQuadrature{D, N} <: AbstractQuadratureRule{D} 
end
GaussQuadrature(ref,n) = GaussQuadrature{typeof(ref),n}()
GaussQuadrature(ref;n) = GaussQuadrature{typeof(ref),n}()

@generated function 
    get_qnodes_and_qweights(q::GaussQuadrature{<:ReferenceTriangle, N}) where {N}
    if N == 1
        x = SVector((Point2D(1/3, 1/3),))
        w = SVector(1/2)
    elseif N == 3
        x = SVector(Point2D(1/6,1/6),
                    Point2D(2/3,1/6),
                    Point2D(1/6,2/3))
        w = SVector(1/6, 1/6, 1/6)
    elseif N == 4
        x = SVector(Point2D(1/3,1/3),
                    Point2D(1/5,1/5),
                    Point2D(1/5,3/5),
                    Point2D(3/5,1/5))
        w = SVector(-9/32, 25/96, 25/96, 25/96)
    elseif N == 6
        x = SVector(Point2D(0.445948490915965, 0.445948490915965),
                    Point2D(0.445948490915965, 0.10810301816807),
                    Point2D(0.10810301816807, 0.445948490915965),
                    Point2D(0.091576213509771, 0.091576213509771),
                    Point2D(0.091576213509771, 0.816847572980459),
                    Point2D(0.816847572980459, 0.091576213509771))
        w = SVector(0.111690794839005, 0.111690794839005, 0.111690794839005, 
                    0.054975871827661, 0.054975871827661, 0.054975871827661)
    else
        notimplemented()
    end
    return :($x,$w)
end

"""
    get_qrule_for_reference_shape(ref,order)

Given a `ref`erence shape and a desired quadrature `order`, return
an appropiate quadrature rule.
"""
function get_qrule_for_reference_shape(ref, order)
    if ref isa ReferenceTriangle
        if order <= 1
            return GaussQuadrature(ref, n=1)
        elseif order <= 2
            return GaussQuadrature(ref, n=3)
        elseif order <= 3
            return GaussQuadrature(ref, n=4)
        elseif order <= 4
            return GaussQuadrature(ref, n=6)
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
