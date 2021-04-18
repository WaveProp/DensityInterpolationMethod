"""
Structures that contains the data for the 
Density Interpolation Method.
"""

"""
    struct DimData

Structure that contains all necessary data
for the Density Interpolation Method.
"""
struct DimData
    # Mesh and quadrature data
    mesh::GenericMesh              # Contains the elements parametrization
    gquad::GlobalQuadrature        # Contains quadrature data
    bbox::SVector{2, Point3D}      # Bounding box `[low_corner, high_corner]`

    # Parameters (maybe move this to a mutable struct?)
    k::Float64       # Wavenumber               
    α::Float64       # Density interpolant `α` parameter  
    β::Float64       # Density interpolant `β` parameter 

    # Surface density `ϕ` at quadrature nodes `yⱼ`.
    # Each entry `(ϕ₁, ϕ₂)` correspond to the coefficients
    # in `ϕ(yⱼ) = ϕ₁τ₁ + ϕ₂τ₂`, where `τ₁,τ₂` are the tangential 
    # vectors at `yⱼ`.
    ϕcoeff::Vector{SVector{2, ComplexF64}}

    # Coefficients of the density interpolant `Φ` at element `eⱼ`.
    # Each entry `cⱼ` is the vector of coefficients of `Φ` at `eⱼ`.
    ccoeff::Vector{Vector{ComplexF64}}

    # List of source points for constructing the
    # density interpolant `Φ`.
    src_list::Vector{Point3D}
end

"""
    getparameters(dimdata::DimData)

Returns the wavenumber `k` and the `α`, `β` density 
interpolant parameters of `dimdata`.
"""
function getparameters(dimdata::DimData)
    return dimdata.k, dimdata.α, dimdata.β
end

"""
    save_ccoeff!(dimdata::DimData, ccoeff, element_index)

Saves the density interpolant coefficients `cⱼ` for element `eⱼ`,
with `j = element_index`.
"""
function save_ccoeff!(dimdata::DimData, ccoeff, element_index)
    dimdata.ccoeff[element_index] = ccoeff
end
