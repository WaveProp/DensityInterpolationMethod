
"""
    struct UniformBlockSize{D, N} <: AbstractVector{Int64}

Structure that represents a vector of length `N` with all
entries equal to `D`. This is useful for generating
BlockArrays.PseudoBlockMatrix with uniform block sizes
without allocating memory for the block sizes. 
See also: [`generate_pseudoblockmatrix`](@ref).
"""
struct UniformBlockSize <: AbstractVector{Int64}
    blocksize::Int64
    len::Int64
end
Base.getindex(b::UniformBlockSize, ::Int64) = b.blocksize
Base.size(b::UniformBlockSize) = (b.len,)

"""
    generate_pseudoblockmatrix(blocktype::StaticMatrix, n, m)

Generates an uninitialized PseudoBlockMatrix of size (`n` blocks) × (`m` blocks) with
blocks of type `blocktype`.
"""
function generate_pseudoblockmatrix(blocktype::Type{<:SMatrix}, n, m)
    T = eltype(blocktype)
    isize, jsize = size(blocktype)
    blocksize_i = UniformBlockSize(isize, n)
    blocksize_j = UniformBlockSize(jsize, m)
    return PseudoBlockMatrix{T}(undef, blocksize_i, blocksize_j)
end

"""
    wrap_into_pseudoblockmatrix(matrix::Matrix, blocktype::SMatrix)

Wraps a `matrix::Matrix` into a PseudoBlockMatrix with blocks of type `blocktype::SMatrix`.
The sizes must be compatible. This operation does not allocate a new matrix.
"""
function wrap_into_pseudoblockmatrix(matrix::Matrix, blocktype::Type{<:SMatrix})
    T = eltype(blocktype)
    @assert eltype(matrix) == T
    (n, nrem), (m, mrem) = divrem.(size(matrix), size(blocktype))
    @assert nrem == 0 && mrem == 0  # size are compatible
    isize, jsize = size(blocktype)
    blocksize_i = UniformBlockSize(isize, n)
    blocksize_j = UniformBlockSize(jsize, m)
    return PseudoBlockMatrix(matrix, blocksize_i, blocksize_j)
end

"""
    get_matrix_from_pseudoblockmatrix(p::PseudoBlockMatrix)

Returns the underlying matrix of `p::PseudoBlockMatrix`. This operation 
does not allocate a new matrix.
"""
function get_matrix_from_pseudoblockmatrix(p::PseudoBlockMatrix)
    return p.blocks
end