using StaticArrays
using LinearAlgebra
using BenchmarkTools
using DensityInterpolationMethod
using DensityInterpolationMethod.IO
using DensityInterpolationMethod.Mesh
using DensityInterpolationMethod.Integration
using DensityInterpolationMethod.MaxwellDIM
const DM = DensityInterpolationMethod.MaxwellDIM

# Load a mesh with quadratic elements
ELEM_ORDER = 2
HMAX = 0.5
mesh_filename = "test/meshes/sphere1.geo"
mesh = read_gmsh_geo(mesh_filename, h=HMAX, order=ELEM_ORDER);
##
# Generates a DimData
# with a quadrature of order QUADRATURE_ORDER
QUADRATURE_ORDER = 2
k = 1     # Wavenumber
n_src = 14  # number of Lebedev sources
α = 2       # DIM α parameter
β = 3       # DIM β parameter
r_factor = 5  # radius factor for Lebedev sources
dimdata = generate_dimdata(mesh, qorder=QUADRATURE_ORDER, k=k,
                           n_src=n_src, α=α, β=β, r=r_factor);

# Set the surface density `ϕ` equal to `τ₁`, 
# where `τ₁` is the (first) tangent vector
ϕcoeff = SVector(1, 0)   # [τ₁coeff, τ₂coeff]
for i in eachindex(dimdata.density_coeff)
    dimdata.density_coeff[i] = ϕcoeff
end

# Compute density interpolant coefficients for all elements
println("Computing DIM matrices...")
DM.assemble_dim_matrices(dimdata)
println("Done")
println("Computing density interpolant...")
DM.compute_density_interpolant(dimdata)
println("Done")

# Reference triangle sampling
function sample_reference_triangle(n_samples)
    sample_per_axis = ceil(Int64, sqrt(2*n_samples))+2
    iter = range(0, 1, length=sample_per_axis)
    iter = iter[2:end]  # remove endpoints
    samples = SVector{2, Float64}[]
    for x in iter
        for y in iter
            if y ≥ 1 - x 
                continue
            end
            push!(samples, SVector(x, y))
        end
    end
    return samples
end

# Sample nodes
n_nodes = 5000   # nodes in mesh
n_nodes_per_element = n_nodes÷get_number_of_elements(mesh)+1
nodelist = sample_reference_triangle(n_nodes_per_element)

# TEST: compare density with density interpolant
# at mesh nodes
ϕlist = []     # α * density
nϕlist = []    # β * n × density
γ₀Φlist = []   # γ₀(density interpolant)
γ₁Φlist = []   # γ₁(density interpolant)
for element_index in eachindex(dimdata.gquad.elements)
    element = getelement(dimdata.mesh, element_index)
    for node in nodelist
        jac = getjacobian(element, node)
        normal = getnormal(element, node)
        ϕ =  jac * ϕcoeff
        push!(ϕlist, dimdata.α * ϕ)
        push!(nϕlist, dimdata.β * cross(normal, ϕ))

        γ₀Φ = DM.evaluate_γ₀interpolant(dimdata, element_index, node)
        γ₁Φ = DM.evaluate_γ₁interpolant(dimdata, element_index, node)
        push!(γ₀Φlist, γ₀Φ)
        push!(γ₁Φlist, γ₁Φ)
    end
end
error0list = norm.(ϕlist - γ₀Φlist) ./ maximum(norm.(ϕlist))
error0 = maximum(error0list) 
error1list = norm.(nϕlist - γ₁Φlist) ./ maximum(norm.(nϕlist))
error1 = maximum(error1list)
println("hmax $(dimdata.hmax)")
println("error_γ₀ $error0")
println("error_γ₁ $error1")


