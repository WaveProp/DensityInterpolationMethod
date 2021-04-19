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
k = 1       # Wavenumber
n_src = 14  # number of Lebedev sources
α = 2       # DIM α parameter
β = 3       # DIM β parameter
dimdata = generate_dimdata(mesh, qorder=QUADRATURE_ORDER, k=k,
                           n_src=n_src, α=α, β=β)

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
ϕlist = []  # density
Φlist = []  # density interpolant
α = dimdata.α
for qnode_index in qnode_indices_in_element
    jac = dimdata.gquad.jacobians[qnode_index]
    ϕ = α * jac * ϕcoeff
    push!(ϕlist, ϕ)

    Φ = DensityInterpolationMethod.MaxwellDIM.
            evaluate_γ₀dim(dimdata, element_index, qnode_index)
    push!(Φlist, Φ)
end
error = maximum(norm.(ϕlist - Φlist)) / maximum(norm.(ϕlist))




