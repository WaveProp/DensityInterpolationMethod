using StaticArrays
using LinearAlgebra
using DensityInterpolationMethod
using DensityInterpolationMethod.IO
using DensityInterpolationMethod.Mesh
using DensityInterpolationMethod.Integration
using DensityInterpolationMethod.MaxwellDIM

# Load a mesh with quadratic elements
ELEM_ORDER = 2
HMAX = 0.5
mesh_filename = "test/meshes/sphere1.geo"
mesh = read_gmsh_geo(mesh_filename, h=HMAX, order=ELEM_ORDER);

# Generates a DimData
# with a quadrature of order QUADRATURE_ORDER
QUADRATURE_ORDER = 4
k = 3      # Wavenumber
n_src = 14  # number of Lebedev sources
α = 2       # DIM α parameter
β = 3       # DIM β parameter
r_factor = 5  # radius factor for Lebedev sources
dimdata = generate_dimdata(mesh, qorder=QUADRATURE_ORDER, k=k,
                           n_src=n_src, α=α, β=β, r=r_factor);

# Set the surface density `ϕ` equal to `τ₁`, 
# where `τ₁` is the (first) tangent vector
ϕcoeff = SVector(1, 0)   # [τ₁coeff, τ₂coeff]
for i in eachindex(dimdata.ϕcoeff)
    dimdata.ϕcoeff[i] = ϕcoeff
end

# Compute density interpolant coefficients for element 1
element_index = 1
DensityInterpolationMethod.MaxwellDIM.assemble_dim_matrices(dimdata, element_index)
DensityInterpolationMethod.MaxwellDIM.compute_density_interpolant(dimdata, element_index)

# Reference triangle sampling
"""
    sample_reference_triangle(n_samples)

Returns `n_samples` uniformly sampled points in the reference triangle.
https://stackoverflow.com/questions/4778147/sample-random-point-in-triangle
"""
function sample_reference_triangle(n_samples)
    A = SVector(0, 0)
    B = SVector(1, 0)
    C = SVector(0, 1)
    r1 = rand(n_samples)
    r2 = rand(n_samples)
    samples = zeros(SVector{2, Float64}, n_samples)
    samples_matrix = reshape(reinterpret(Float64, samples), 2, n_samples)
    for j in 1:n_samples
        for i in 1:2
            samples_matrix[i, j] = (1 - sqrt(r1[j])) * A[i] + 
                                   (sqrt(r1[j]) * (1 - r2[j])) * B[i] + 
                                   (sqrt(r1[j]) * r2[j]) * C[i]
        end
    end
    return samples
end

# TEST: compare density with density interpolant
# at element nodes
n_nodes = 5000
nodelist = sample_reference_triangle(n_nodes)
ϕlist = []     # α * density
nϕlist = []    # β * n × density
γ₀Φlist = []   # γ₀(density interpolant)
γ₁Φlist = []   # γ₁(density interpolant)
element = getelement(dimdata.mesh, element_index)
for node in nodelist
    jac = getjacobian(element, node)
    normal = getnormal(element, node)
    ϕ =  jac * ϕcoeff
    push!(ϕlist, dimdata.α * ϕ)
    push!(nϕlist, dimdata.β * cross(normal, ϕ))

    γ₀Φ = DensityInterpolationMethod.MaxwellDIM. 
            evaluate_γ₀dim(dimdata, element_index, node)
    γ₁Φ = DensityInterpolationMethod.MaxwellDIM. 
            evaluate_γ₁dim(dimdata, element_index, node)
    push!(γ₀Φlist, γ₀Φ)
    push!(γ₁Φlist, γ₁Φ)
end
error0list = norm.(ϕlist - γ₀Φlist) ./ maximum(norm.(ϕlist))
error0 = maximum(error0list) 
error1list = norm.(nϕlist - γ₁Φlist) ./ maximum(norm.(nϕlist))
error1 = maximum(error1list)
println("error_γ₀ $error0, error_γ₁ $error1")


