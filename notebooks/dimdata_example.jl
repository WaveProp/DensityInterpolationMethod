using StaticArrays
using LinearAlgebra
using DensityInterpolationMethod
using DensityInterpolationMethod.IO
using DensityInterpolationMethod.Integration
using DensityInterpolationMethod.MaxwellDIM

# Load a mesh with quadratic elements
ELEM_ORDER = 2.0
HMAX = 0.5
mesh_filename = "test/meshes/sphere1.geo"
mesh = read_gmsh_geo(mesh_filename, h=HMAX, order=ELEM_ORDER);

# Generates a DimData
# with a quadrature of order QUADRATURE_ORDER
QUADRATURE_ORDER = 2
k = 3      # Wavenumber
n_src = 50  # number of Lebedev sources
α = 2       # DIM α parameter
β = 3       # DIM β parameter
r_factor = 5  # radius factor for Lebedev sources
dimdata = generate_dimdata(mesh, qorder=QUADRATURE_ORDER, k=k,
                           n_src=n_src, α=α, β=β, r=r_factor)

# Set the surface density `ϕ` equal to `τ₁`, 
# where `τ₁` is the (first) tangent vector
ϕcoeff = SVector(1, 0)   # [τ₁coeff, τ₂coeff]
for i in eachindex(dimdata.ϕcoeff)
    dimdata.ϕcoeff[i] = ϕcoeff
end

# Compute density interpolant coefficients for element 1
element_index = 1
DensityInterpolationMethod.MaxwellDIM.
    construct_density_interpolant(dimdata, element_index)
ccoeff = DensityInterpolationMethod.MaxwellDIM.get_dimcoeff(dimdata, element_index)

# Compare density with density interpolant
# at qnodes in element
qnode_indices_in_element = dimdata.gquad.el2indices[element_index]
ϕlist = []     # α * density
nϕlist = []    # β * n × density
γ₀Φlist = []   # γ₀(density interpolant)
γ₁Φlist = []   # γ₁(density interpolant)
for qnode_index in qnode_indices_in_element
    jac = dimdata.gquad.jacobians[qnode_index]
    normal = dimdata.gquad.normals[qnode_index]
    ϕ =  jac * ϕcoeff
    push!(ϕlist, dimdata.α * ϕ)
    push!(nϕlist, dimdata.β * cross(normal, ϕ))

    γ₀Φ = DensityInterpolationMethod.MaxwellDIM. 
            evaluate_γ₀dim(dimdata, element_index, qnode_index)
    γ₁Φ = DensityInterpolationMethod.MaxwellDIM. 
            evaluate_γ₁dim(dimdata, element_index, qnode_index)
    push!(γ₀Φlist, γ₀Φ)
    push!(γ₁Φlist, γ₁Φ)
end
error0 = maximum(norm.(ϕlist - γ₀Φlist)) / maximum(norm.(ϕlist))
error1 = maximum(norm.(nϕlist - γ₁Φlist)) / maximum(norm.(ϕlist))




