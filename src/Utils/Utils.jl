"""
    Utils

Module containing various utility functions .
"""
module Utils
using StaticArrays

export DIMENSION2,
       DIMENSION3,
       Point3D, 
       Point2D, 
       ComplexPoint3D,
       notimplemented, 
       abstractmethod,
       assert_extension

# Constants
const DIMENSION2 = 2
const DIMENSION3 = 3
       
"""
    const Point3D
    const Point3D(x1, x2, x3)
    const Point3D(x::NTuple{3, Float64})

A point in 3D space, stored in a StaticArray.
Point3D = SVector{3, Float64}.
"""
const Point3D = SVector{3, Float64}

"""
    const Point2D
    const Point2D(x1, x2)
    const Point2D(x::NTuple{2, Float64})

A point in 2D space, stored in a StaticArray.
Point2D = SVector{2, Float64}.
"""
const Point2D = SVector{2, Float64}

"""
    const ComplexPoint3D
    const ComplexPoint3D(x1, x2, x3)
    const ComplexPoint3D(x::NTuple{3, ComplexF64})

A complex 3D point, stored in a StaticArray.
ComplexPoint3D = SVector{3, ComplexF64}.
"""
const ComplexPoint3D = SVector{3, ComplexF64}

"""
    notimplemented()

Things which should probably be implemented at some point.
"""
function notimplemented()
    error("not (yet) implemented")
end 

"""
    abstractmethod

A method of an `abstract type` for which concrete subtypes are expected
to provide an implementation.
"""
function abstractmethod(T)
    error("this method needs to be implemented by the concrete subtype $T.")
end 

"""
    assert_extension(fname,ext,[msg])

Check that `fname` is of extension `ext`. Print the message `msg` as an assertion error otherwise.
"""
function assert_extension(fname::String,ext::String,msg="file extension must be $(ext)")
    r = Regex("$(ext)\$")    
    @assert occursin(r,fname) msg
end
end # module