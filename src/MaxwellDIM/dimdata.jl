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
    generate_dimdata(mesh::GenericMesh; qorder=2, k=1, α=1, β=1, n_src=14, r=5)

Generates a `DimData` structure from `mesh::GenericMesh`.

# Arguments
- `mesh::GenericMesh`: mesh of elements parametrization.
- `qorder=2`: quadrature order.
- `k=1`: wavenumber.
- `α=1`: density interpolant `α` parameter.
- `β=1`: density interpolant `β` parameter.
- `n_src=14`: number of density interpolant source points.
- `r=5`: radius factor of density interpolant source points sphere.
"""
function generate_dimdata(mesh::GenericMesh; qorder=2, k=1, α=1, β=1, n_src=14, r=5)
    gquad = generate_globalquadrature(mesh, order=qorder)
    n_qnodes = get_number_of_qnodes(gquad)
    n_elements = get_number_of_elements(gquad)
    
    ϕcoeff = zeros(SVector{2, ComplexF64}, n_qnodes)
    n_ccoef = DIMENSION3 * n_qnodes   # density interpolation coeffs: 3 per qnode
    ccoeff = [zeros(ComplexF64, n_ccoef) for _ in 1:n_elements]

    # compute source points
    bbox, bbox_center, bbox_radius = compute_bounding_box(gquad)
    src_radius = r * bbox_radius
    src_list = get_sphere_sources_lebedev(n_src, src_radius, bbox_center)
    return DimData(mesh, gquad, k, α, β, ϕcoeff, ccoeff, src_list)
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
    get_dimcoeff!(dimdata::DimData, element_index)

Returns the density interpolant coefficients `cⱼ` for element `eⱼ`,
with `j = element_index`. The coefficients is reshaped into a list
of complex 3D vectors for each source point.
"""
function get_dimcoeff(dimdata::DimData, element_index)
    ccoeff = dimdata.ccoeff[element_index]
    return reinterpret(SVector{DIMENSION3, ComplexF64}, ccoeff)
end

"""
    save_dimcoeff!(dimdata::DimData, element_index, ccoeff)

Saves the density interpolant coefficients `cⱼ` for element `eⱼ`,
with `j = element_index`.
"""
function save_dimcoeff!(dimdata::DimData, element_index, ccoeff)
    dimdata.ccoeff[element_index] = ccoeff
end

"""
    evaluate_γ₀dim(dimdata::DimData, element_index, qnode_index::Integer)

Evaluates `γ₀Φₘ(yⱼ)`, where `Φₘ` is the density interpolant
of element `eₘ` with `m = element_index` and `yⱼ` is a quadrature 
node with `j = qnode_index`.
"""
function evaluate_γ₀dim(dimdata::DimData, element_index, qnode_index::Integer)
    k = dimdata.k       # wavenumber
    qnode = dimdata.gquad.nodes[qnode_index]
    qnormal = dimdata.gquad.normals[qnode_index]
    # Density interpolant coefficients
    ccoeff = get_dimcoeff(dimdata, element_index)   
    return sum(zip(dimdata.src_list, ccoeff)) do (z,c)
        single_layer_kernel_eval(qnode, z, k, qnormal, c)
    end
end

"""
    evaluate_γ₀dim(dimdata::DimData, element_index, x̂)

Evaluates `γ₀Φₘ(y(x̂))`, where `Φₘ` is the density interpolant
of element `eₘ` with `m = element_index`,`y` is the element parametrization
and `x̂` is a 2D point in parametric coordinates.
"""
function evaluate_γ₀dim(dimdata::DimData, element_index, x̂)
    @assert length(x̂) == 2
    k = dimdata.k       # wavenumber
    element = getelement(dimdata.mesh, element_index)
    node = element(x̂)
    normal = getnormal(element, x̂)
    # Density interpolant coefficients
    ccoeff = get_dimcoeff(dimdata, element_index)   
    return sum(zip(dimdata.src_list, ccoeff)) do (z,c)
        single_layer_kernel_eval(node, z, k, normal, c)
    end
end

"""
    evaluate_γ₁dim(dimdata::DimData, element_index, qnode_index::Integer)

Evaluates `γ₁Φₘ(yⱼ)`, where `Φₘ` is the density interpolant
of element `eₘ` with `m = element_index` and `yⱼ` is a quadrature 
node with `j = qnode_index`.
"""
function evaluate_γ₁dim(dimdata::DimData, element_index, qnode_index::Integer)
    k = dimdata.k       # wavenumber
    qnode = dimdata.gquad.nodes[qnode_index]
    qnormal = dimdata.gquad.normals[qnode_index]
    # Density interpolant coefficients
    ccoeff = get_dimcoeff(dimdata, element_index)   
    return sum(zip(dimdata.src_list, ccoeff)) do (z,c)
        double_layer_kernel_eval(qnode, z, k, qnormal, c)
    end
end

"""
    evaluate_γ₁dim(dimdata::DimData, element_index, x̂)

Evaluates `γ₁Φₘ(y(x̂))`, where `Φₘ` is the density interpolant
of element `eₘ` with `m = element_index`,`y` is the element parametrization
and `x̂` is a 2D point in parametric coordinates.
"""
function evaluate_γ₁dim(dimdata::DimData, element_index, x̂)
    @assert length(x̂) == 2
    k = dimdata.k       # wavenumber
    element = getelement(dimdata.mesh, element_index)
    node = element(x̂)
    normal = getnormal(element, x̂)
    # Density interpolant coefficients
    ccoeff = get_dimcoeff(dimdata, element_index)   
    return sum(zip(dimdata.src_list, ccoeff)) do (z,c)
        double_layer_kernel_eval(node, z, k, normal, c)
    end
end
