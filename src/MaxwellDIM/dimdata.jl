"""
Structures that contains the data for the 
Density Interpolation Method.
"""

"""
    abstract type AbstractDimFormulation

Type of formulation used in the Density Interpolation Method (DIM).
"""
abstract type AbstractDimFormulation end

"""
    abstract type IndirectDimFormulation <: AbstractDimFormulation

Indirect formulation. The integral operator `C̃[ϕ]:= αK[ϕ] + βT[n×ϕ]` is
regularized with the Density Interpolation Method.
"""
abstract type IndirectDimFormulation <: AbstractDimFormulation end

"""
    abstract type IndirectDimFormulation <: AbstractDimFormulation

Direct formulation. The integral operator `C̃[ϕ, φ]:= αK[ϕ] + βT[φ]` is
regularized with the Density Interpolation Method.
"""
abstract type DirectDimFormulation <: AbstractDimFormulation end

"""
    struct DimData

Structure that contains all necessary data
for the Density Interpolation Method.
"""
struct DimData{F<:AbstractDimFormulation}
    # Mesh and quadrature data
    hmax::Float64                  # Maximum element size
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
    density_coeff::Vector{ComplexPoint2D}
    # Surface density `φ` at quadrature nodes `yⱼ`.
    # This is only used in a direct formulation 
    # (`DimData{DirectDimFormulation}`).
    density2_coeff::Vector{ComplexPoint2D} 
    # Coefficients of the density interpolant `Φ` at element `eⱼ`.
    # Each entry `cⱼ` is the vector of coefficients of `Φ` at `eⱼ`.
    interpolant_coeff::Vector{Vector{ComplexF64}}
    # Integral operator `C̃_{α,β}[ϕ]` value at quadrature nodes.
    integral_op::Vector{ComplexPoint3D}
    # LQ matrices for constructing the density interpolant `Φ`,
    # for each element.
    Lmatrices::Vector{LowerTriangular{ComplexF64, Matrix{ComplexF64}}}
    Qmatrices::Vector{Matrix{ComplexF64}}
    # List of source points for constructing the
    # density interpolant `Φ`.
    src_list::Vector{Point3D}
    # Constructor
    function DimData{F}(args...) where F
        msg = """Only strict subtypes of `AbstractDimFormulation`
        can be used to instantiate DimData."""
        @assert (F!=AbstractDimFormulation) msg
        return new{F}(args...)
    end
end

# Aliases
"""
    const IndirectDimData = DimData{IndirectDimFormulation}
"""
const IndirectDimData = DimData{IndirectDimFormulation}

"""
    const DirectDimData = DimData{DirectDimFormulation}
"""
const DirectDimData = DimData{DirectDimFormulation}

"""
    generate_dimdata(mesh::GenericMesh; qorder=2, k=1, α=1, β=1, n_src=14, r=5, indirect=true)

Generates a `DimData` structure from `mesh::GenericMesh`.

# Arguments
- `mesh::GenericMesh`: mesh of elements parametrization.
- `qorder=2`: quadrature order.
- `k=1`: wavenumber.
- `α=1`: density interpolant `α` parameter.
- `β=1`: density interpolant `β` parameter.
- `n_src=14`: number of density interpolant source points.
- `r=5`: radius factor of density interpolant source points sphere.
- `indirect=true`: select `true` for an indirect formulation (`IndirectDimData`),
otherwise `false` for a direct formulation (`DirectDimData`). 
"""
function generate_dimdata(mesh::GenericMesh; qorder=2, k=1, α=1, β=1, n_src=14, r=5, indirect=true)
    hmax = compute_hmax(mesh)
    gquad = generate_globalquadrature(mesh, order=qorder)
    n_qnodes = get_number_of_qnodes(gquad)
    n_elements = get_number_of_elements(gquad)
    
    density_coeff = Vector{ComplexPoint2D}(undef, n_qnodes)
    n_interpolant_coeff = DIMENSION3 * n_src   # density interpolation coeffs: 3 per src point
    interpolant_coeff = [Vector{ComplexF64}(undef, n_interpolant_coeff) for _ in 1:n_elements]
    integral_op = Vector{ComplexPoint3D}(undef, n_qnodes)
    Lmatrices = [LowerTriangular(Matrix{ComplexF64}(undef, 0, 0)) for _ in 1:n_elements]
    Qmatrices = [Matrix{ComplexF64}(undef, 0, 0) for _ in 1:n_elements]

    # compute source points
    bbox, bbox_center, bbox_radius = compute_bounding_box(gquad)
    src_radius = r * bbox_radius
    src_list = get_sphere_sources_lebedev(n_src, src_radius, bbox_center)

    # formulation
    if indirect
        density2_coeff = Vector{ComplexPoint2D}[]
        Formulation = IndirectDimData
    else
        density2_coeff = Vector{ComplexPoint2D}(undef, n_qnodes)
        Formulation = DirectDimData
    end
    return Formulation(hmax, mesh, gquad, k, α, β, density_coeff, density2_coeff, interpolant_coeff, integral_op,
                   Lmatrices, Qmatrices, src_list)
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
    get_interpolant_coeff(dimdata::DimData, element_index)

Returns the density interpolant coefficients `cⱼ` for element `eⱼ`,
with `j = element_index`. The coefficients is reshaped into a list
of complex 3D vectors for each source point.
"""
function get_interpolant_coeff(dimdata::DimData, element_index)
    interpolant_coeff = dimdata.interpolant_coeff[element_index]
    return reinterpret(SVector{DIMENSION3, ComplexF64}, interpolant_coeff)
end

"""
    get_number_of_qnodes(dimdata::DimData)

Returns the total number of quadrature nodes in `dimdata`.
"""
function get_number_of_qnodes(dimdata::DimData)
    return get_number_of_qnodes(dimdata.gquad)
end

"""
    get_number_of_srcs(dimdata::DimData)

Returns the total number of source points in `dimdata`.
"""
function get_number_of_srcs(dimdata::DimData)
    return length(dimdata.src_list)
end

"""
    reset_integral_operator_value(dimdata::DimData)
    
Sets to zero the value of the integral operator `dimdata.integral_op`.
"""
function reset_integral_operator_value(dimdata::DimData)
    fill!(dimdata.integral_op, zero(eltype(dimdata.integral_op)))
end

"""
    get_surface_density(dimdata::DimData, qnode::QNode)
    get_surface_density(dimdata::DimData, qnode_index::Integer)

Returns the surface density `ϕ(yⱼ) = ϕ₁τ₁ + ϕ₂τ₂` at qnode `j = qnode_index`,
where `τ₁,τ₂` are the tangential vectors and `(ϕ₁, ϕ₂)` are coefficients
at `yⱼ`.
"""
function get_surface_density(dimdata::DimData, qnode::QNode)
    jac = qnode.jacobian       # jacobian
    qnode_index = qnode.index  # qnode global index
    density_coeff = dimdata.density_coeff[qnode_index]
    ϕ = jac * density_coeff
    return ϕ
end
function get_surface_density(dimdata::DimData, qnode_index::Integer)
    qnode = get_qnode(dimdata.gquad, qnode_index)    # qnode object
    return get_surface_density(dimdata, qnode)
end

"""
    project_field_onto_surface_density(dimdata::DimData, field)

Projects the tangential component of a vector field `field`, 
defined on the quadrature nodes, onto the surface density `ϕ`.
The new surface density components are stored in `dimdata.density_coeff`.
"""
function project_field_onto_surface_density(dimdata::DimData, field)
    @assert length(field) == get_number_of_qnodes(dimdata)
    for i in get_qnode_indices(dimdata.gquad)
        vec = field[i]                      # vector field at qnode i
        qnode = get_qnode(dimdata.gquad, i) # qnode i
        jac = qnode.jacobian                # jacobian at qnode i
        dimdata.density_coeff[i] = jac \ vec
    end
end

"""
    evaluate_γ₀interpolant(dimdata::DimData, element_index::Integer, qnode::QNode)
    evaluate_γ₀interpolant(dimdata::DimData, element_index::Integer, qnode_index::Integer)

Evaluates `γ₀Φₘ(yⱼ)`, where `Φₘ` is the density interpolant
of element `eₘ` with `m = element_index` and `yⱼ` is a quadrature 
node with `j = qnode_index`.
"""
function evaluate_γ₀interpolant(dimdata::DimData, element_index::Integer, qnode::QNode)
    k = dimdata.k              # wavenumber
    qnode_index = qnode.index  # qnode global index
    x, _, _, n = get_qnode_data(qnode)
    # Density interpolant coefficients
    interpolant_coeff = get_interpolant_coeff(dimdata, element_index)   
    return sum(zip(dimdata.src_list, interpolant_coeff)) do (z,c)
        single_layer_kernel(x, z, k, n, c)
    end
end
function evaluate_γ₀interpolant(dimdata::DimData, element_index::Integer, qnode_index::Integer)
    qnode = get_qnode(dimdata.gquad, qnode_index)
    return evaluate_γ₀interpolant(dimdata, element_index, qnode)
end


"""
    evaluate_γ₀interpolant(dimdata::DimData, element_index, x̂)

Evaluates `γ₀Φₘ(y(x̂))`, where `Φₘ` is the density interpolant
of element `eₘ` with `m = element_index`, `y` is the element parametrization
and `x̂` is a 2D point in parametric coordinates.
"""
function evaluate_γ₀interpolant(dimdata::DimData, element_index, x̂)
    @assert length(x̂) == 2
    k = dimdata.k       # wavenumber
    element = getelement(dimdata.mesh, element_index)
    node = element(x̂)
    normal = getnormal(element, x̂)
    # Density interpolant coefficients
    ccoeff = get_interpolant_coeff(dimdata, element_index)   
    return sum(zip(dimdata.src_list, ccoeff)) do (z,c)
        single_layer_kernel(node, z, k, normal, c)
    end
end

"""
    evaluate_γ₁interpolant(dimdata::DimData, element_index::Integer, qnode::QNode)
    evaluate_γ₁interpolant(dimdata::DimData, element_index::Integer, qnode_index::Integer)

Evaluates `γ₁Φₘ(yⱼ)`, where `Φₘ` is the density interpolant
of element `eₘ` with `m = element_index` and `yⱼ` is a quadrature 
node with `j = qnode_index`.
"""
function evaluate_γ₁interpolant(dimdata::DimData, element_index::Integer, qnode::QNode)
    k = dimdata.k              # wavenumber
    qnode_index = qnode.index  # qnode global index
    x, _, _, n = get_qnode_data(qnode)
    # Density interpolant coefficients
    interpolant_coeff = get_interpolant_coeff(dimdata, element_index)   
    return sum(zip(dimdata.src_list, interpolant_coeff)) do (z,c)
        double_layer_kernel(x, z, k, n, c)
    end
end
function evaluate_γ₁interpolant(dimdata::DimData, element_index::Integer, qnode_index::Integer)
    qnode = get_qnode(dimdata.gquad, qnode_index)
    return evaluate_γ₁interpolant(dimdata, element_index, qnode)
end

"""
    evaluate_γ₁interpolant(dimdata::DimData, element_index, x̂)

Evaluates `γ₁Φₘ(y(x̂))`, where `Φₘ` is the density interpolant
of element `eₘ` with `m = element_index`,`y` is the element parametrization
and `x̂` is a 2D point in parametric coordinates.
"""
function evaluate_γ₁interpolant(dimdata::DimData, element_index, x̂)
    @assert length(x̂) == 2
    k = dimdata.k       # wavenumber
    element = getelement(dimdata.mesh, element_index)
    node = element(x̂)
    normal = getnormal(element, x̂)
    # Density interpolant coefficients
    ccoeff = get_interpolant_coeff(dimdata, element_index)   
    return sum(zip(dimdata.src_list, ccoeff)) do (z,c)
        double_layer_kernel(node, z, k, normal, c)
    end
end
