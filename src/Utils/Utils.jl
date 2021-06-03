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
       ComplexPoint2D,
       notimplemented,
       abstractmethod,
       assert_extension,
       print_threads_info,
       enable_debug,
       diagonalblockmatrix_to_matrix,
       blockmatrix_to_matrix,
       matrix_to_blockmatrix

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
    const ComplexPoint2D
    const ComplexPoint2D(x1, x2)
    const ComplexPoint2D(x::NTuple{2, ComplexF64})

A complex 2D point, stored in a StaticArray.
ComplexPoint2D = SVector{2, ComplexF64}.
"""
const ComplexPoint2D = SVector{2, ComplexF64}

"""
    notimplemented()

Things which should probably be implemented at some point.
"""
function notimplemented()
    error("not (yet) implemented")
end

"""
    abstractmethod(T)

A method of an `abstract type` for which concrete subtypes are expected
to provide an implementation.
"""
function abstractmethod(T::Type)
    error("this method needs to be implemented by the concrete subtype $T.")
end
abstractmethod(T) = abstractmethod(typeof(T))

"""
    assert_extension(fname,ext,[msg])

Check that `fname` is of extension `ext`. Print the message `msg` as an assertion error otherwise.
"""
function assert_extension(fname::String,ext::String,msg="file extension must be $(ext)")
    r = Regex("$(ext)\$")
    @assert occursin(r,fname) msg
end

"""
    print_threads_info()

Prints in console the total number of threads.
"""
function print_threads_info()
    @info "Number of threads: $(Threads.nthreads())"

end

"""
    enable_debug(flag=true)

Activate debugging messages by setting the environment variable `JULIA_DEBUG` to
`DensityInterpolationMethod`. If `flag=false` deactivate debugging messages.
"""
function enable_debug(flag=true)
    if flag
        ENV["JULIA_DEBUG"] = "DensityInterpolationMethod"
    else
        ENV["JULIA_DEBUG"] = ""
    end
end

"""
    diagonalblockmatrix_to_matrix(A::Matrix{B}) where {B<:SMatrix}

Convert a diagonal block matrix `A::AbstractVector{B}`, where `A` is the list of diagonal blocks
and `B<:SMatrix`, to the equivalent `Matrix{T}`, where `T = eltype(B)`.
"""
function diagonalblockmatrix_to_matrix(A::AbstractVector{B}) where B<:SMatrix
    # FIXME: convert to a sparse matrix instead
    T = eltype(B) 
    sblock = size(B)
    ss = size(A) .* sblock  # matrix size when viewed as matrix over T
    Afull = zeros(T, ss)
    i_full, j_full = (1, 1)
    for subA in A
        i_tmp = i_full
        for j in 1:sblock[2]
            i_full = i_tmp
            for i in 1:sblock[1]
                Afull[i_full, j_full] = subA[i, j]
                i_full += 1
            end
            j_full += 1
        end
    end
    return Afull
end

"""
    blockmatrix_to_matrix(A::AbstractMatrix{B}) where {B<:SMatrix}

Convert an `AbstractMatrix{B}`, where `B<:SMatrix`, to the equivalent `Matrix{T}`, where `T = eltype(B)`.
"""
function blockmatrix_to_matrix(A::AbstractMatrix{B}) where B<:SMatrix
    T = eltype(B) 
    sblock = size(B)
    ss     = size(A).*sblock # matrix size when viewed as matrix over T
    Afull = Matrix{T}(undef,ss)
    for i=1:ss[1], j=1:ss[2]
        bi, ind_i = divrem(i-1,sblock[1]) .+ (1,1)
        bj, ind_j = divrem(j-1,sblock[2]) .+ (1,1)
        Afull[i,j] = A[bi,bj][ind_i,ind_j]
    end
    return Afull
end

"""
    matrix_to_blockmatrix(A::AbstractMatrix,B)

Convert an `AbstractMatrix{T}` to a `Matrix{B}`, where `B<:Type{SMatrix}`. The element
type of `B` must match that of `A`, and the size of `A` must be divisible by the
size of `B` along each dimension. 
"""
function matrix_to_blockmatrix(A::AbstractMatrix, B::Type{<:SMatrix})
    @assert eltype(A) == eltype(B)
    @assert sum(size(A) .% size(B)) == 0 "block size $(size(B)) not compatible with size of A=$(size(A))"
    sblock = size(B)
    nblock = div.(size(A),sblock)
    Ablock = Matrix{B}(undef,nblock)
    for i in 1:nblock[1]
        istart = (i-1)*sblock[1] + 1
        iend = i*sblock[1]
        for j in 1:nblock[2]
            jstart = (j-1)*sblock[2] + 1
            jend   = j*sblock[2]
            Ablock[i,j] = A[istart:iend,jstart:jend]
        end
    end
    return Ablock
end

end # module
