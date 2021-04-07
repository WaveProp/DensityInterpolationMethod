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
    (q::AbstractQuadratureRule)()

Return the quadrature nodes `x` and weights `w` on the `domain(q)`.
"""
function (q::AbstractQuadratureRule)() 
    abstractmethod(typeof(q))
end

"""
    qnodes(Y)

Return the quadrature nodes associated with `Y`.
"""
getqnodes(q::AbstractQuadratureRule) = q()[1]

"""
    qweights(Y)

Return the quadrature weights associated with `Y`.
"""
getqweights(q::AbstractQuadratureRule) = q()[2]

"""
    qnormals(Y)

Return the normal vector at the quadrature nodes of `Y.
"""
getqnormals(q::AbstractQuadratureRule) = abstractmethod(typeof(q))

"""
    integrate(f, q::AbstractQuadrature)

Integrate the function `f` using the quadrature rule `q`. This is simply
`sum(f.(x) .* w)`, where `x` and `w` are the quadrature nodes and weights, respectively.
"""
function integrate(f, q::AbstractQuadratureRule)
    x,w = q()
    result = sum(zip(x,w)) do (x,w)
                f(x)*prod(w)
             end
    return result
end

"""
    struct GaussQuadrature{D, N} <: AbstractQuadratureRule{D}

Tabulated `N`-point symmetric Gauss quadrature rule for integration over `D`.
This is currently implemented *by hand* for low values of `N` on triangles.
"""
struct GaussQuadrature{D, N} <: AbstractQuadratureRule{D} 
end
GaussQuadrature(ref,n) = GaussQuadrature{typeof(ref),n}()
GaussQuadrature(ref;n) = GaussQuadrature{typeof(ref),n}()

@generated function (q::GaussQuadrature{<:ReferenceTriangle, N})() where {N}
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
        end
    end
    error("no appropriate quadrature rule found.")
end

"""
    get_qrule_for_element(E, order)

Given an element type `E`, return an appropriate quadrature of order `order`.
"""
function get_qrule_for_element(E, order)
    get_qrule_for_reference_shape(getdomain(E), order)
end
