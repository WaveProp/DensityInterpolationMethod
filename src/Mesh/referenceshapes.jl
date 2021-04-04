"""
    Definitions of reference domains/shapes in `ℜ²`.
    Used mostly for defining more complex shapes as transformations mapping an
    `AbstractReferenceShape` into some region of `ℜ³`.
"""

"""
    abstract type AbstractReferenceShape{N}
    
A reference domain/shape in `ℜ²`.
"""
abstract type AbstractReferenceShape end
Base.in(x,::AbstractReferenceShape) = abstractmethod(typeof(el))
getvertices(::AbstractReferenceShape) = abstractmethod(typeof(el))
getcenter(::AbstractReferenceShape) = abstractmethod(typeof(el))
get_number_of_nodes(::AbstractReferenceShape) = abstractmethod(typeof(el))  # lagrangian nodes
getgetnodes(::AbstractReferenceShape) = abstractmethod(typeof(el))            # lagrangian nodes

"""
    lagrange_basis(el::AbstractReferenceShape)

Function that returns the Lagrange Polynomial basis associated with 
the reference shape 'el'. Returns a tuple (b_1, ..., b_n), where b_i
is the Lagrange polynomial associated with node n_i of 'el'.
"""
function lagrange_basis(el::AbstractReferenceShape)
    abstractmethod(typeof(el))
end

"""
    abstract type ReferenceTriangle
    
An abstract 2D reference triangle with vertices `(0,0),(0,1),(1,0)`.
"""
abstract type ReferenceTriangle <: AbstractReferenceShape end
Base.in(x,::ReferenceTriangle) = 0 ≤ x[1] ≤ 1 && 0 ≤ x[2] ≤ 1 - x[1]
getvertices(::ReferenceTriangle) = Point2D(0,0), Point2D(1,0), Point2D(0,1)
getcenter(::ReferenceTriangle) = Point2D(1/3, 1/3)

"""
    struct ReferenceTriangle3 <: ReferenceTriangle

Singleton of a reference triangle with 3 lagrangian intepolation nodes.
It is used to map the ReferenceTriangle into flat triangles in 3D.
"""
struct ReferenceTriangle3 <: ReferenceTriangle
end
get_number_of_nodes(::ReferenceTriangle3) = 3
getnodes(::ReferenceTriangle3) = Point2D(0,0), Point2D(1,0), Point2D(0,1)

function lagrange_basis(el::ReferenceTriangle3)
    return _ReferenceTriangle3_basis
end

"""
    struct ReferenceTriangle6 <: ReferenceTriangle

Singleton of a reference triangle with 6 lagrangian intepolation nodes.
It is used to map the ReferenceTriangle into quadratic triangles in 3D.
"""
struct ReferenceTriangle6 <: ReferenceTriangle
end
get_number_of_nodes(::ReferenceTriangle6) = 6
getnodes(::ReferenceTriangle6) = Point2D(0,0), Point2D(1,0), Point2D(0,1),
                              Point2D(1/2,0), Point2D(1/2,1/2), Point2D(0,1/2)

function lagrange_basis(el::ReferenceTriangle6)
    return _ReferenceTriangle6_basis
end

