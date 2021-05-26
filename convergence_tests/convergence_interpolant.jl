using StaticArrays
using LinearAlgebra
using BenchmarkTools
using DensityInterpolationMethod
using DensityInterpolationMethod.Utils
using DensityInterpolationMethod.IO
using DensityInterpolationMethod.Mesh
using DensityInterpolationMethod.Integration
using DensityInterpolationMethod.MaxwellDIM
using Random
print_threads_info()
const DM = DensityInterpolationMethod.MaxwellDIM

function sample_reference_triangle(n_samples)
    isnotintriangle(x) = x[1] + x[2] ≥ 1
    reflect(x) = SVector(1 - x[1], 1 - x[2])
    samples = rand(SVector{2, Float64}, n_samples)
    outsamples = @view samples[isnotintriangle.(samples)]
    outsamples .= reflect.(outsamples)
    return samples
end

# Compute error
function compute_element_error(dimdata, interpolant_element_index, target_element_index, k, src, pol, n_nodes_per_element)
    element = getelement(dimdata.mesh, target_element_index)
    error0 = 0.0
    error1 = 0.0
    norm_αϕ = 0.0
    norm_βnϕ = 0.0
    nodelist = sample_reference_triangle(n_nodes_per_element)
    for node in nodelist
        normal = getnormal(element, node)
        yi = element(node)
        ϕ =  DM.single_layer_kernel(yi, src, k, normal, pol)
        αϕ = dimdata.α * ϕ
        βnϕ = dimdata.β * cross(normal, ϕ)
        γ₀Φ = DM.evaluate_γ₀interpolant(dimdata, interpolant_element_index, target_element_index, node)
        γ₁Φ = DM.evaluate_γ₁interpolant(dimdata, interpolant_element_index, target_element_index, node)
        error0 = max(error0, norm(αϕ - γ₀Φ))
        error1 = max(error1, norm(βnϕ - γ₁Φ))
        norm_αϕ = max(norm_αϕ, norm(αϕ))
        norm_βnϕ = max(norm_βnϕ, norm(βnϕ))
    end
    return error0, error1, norm_αϕ, norm_βnϕ
end

function convergence_interpolant(ELEM_ORDER, HMAX, QUADRATURE_ORDER, k, n_src, r_factor)
    # Load mesh 
    mesh_filename = "test/meshes/sphere1.geo"
    mesh = read_gmsh_geo(mesh_filename, h=HMAX, order=ELEM_ORDER, verbosity=false);
    neighbors = find_neighboring_elements(mesh)

    # Generates a DimData
    α = 2       # DIM α parameter
    β = 3       # DIM β parameter
    dimdata = generate_dimdata(mesh, qorder=QUADRATURE_ORDER, k=k,
                           n_src=n_src, α=α, β=β, r=r_factor);
    DM.assemble_dim_matrices(dimdata)

    # Set the surface density `ϕ` equal to an electric dipole field
    # Field produced by electric dipole
    src = Point3D(3, 3, 3)    # dipole location
    pol = Point3D(1, 1, 1)    # dipole polarization
    γ₀Efields = similar(dimdata.gquad.qnodes, ComplexPoint3D)
    for i in eachindex(dimdata.gquad.qnodes)
        qnode = get_qnode(dimdata.gquad, i)
        yi, _, _, ni = get_qnode_data(qnode)
        γ₀Efields[i] = DM.single_layer_kernel(yi, src, k, ni, pol)
    end
    # Project fields in each DimData
    DM.project_field_onto_surface_density!(dimdata, γ₀Efields)
    DM.compute_density_interpolant!(dimdata)

    # Sample nodes
    n_nodes = 5000   # nodes in mesh
    n_nodes_per_element = n_nodes÷get_number_of_elements(mesh)+1

    ##
    # TEST: compare density with density interpolant
    # at mesh nodes
    Random.seed!(1)
    n_elements = get_number_of_elements(dimdata.gquad)
    error0list = zeros(Float64, n_elements)
    error1list = zeros(Float64, n_elements)
    max_αϕ = Threads.Atomic{Float64}(0)
    max_βnϕ = Threads.Atomic{Float64}(0)
    Threads.@threads for element_index in eachindex(dimdata.gquad.elements)
        # self error
        error0, error1, norm_αϕ, norm_βnϕ = compute_element_error(dimdata, element_index, element_index, k, src, pol, n_nodes_per_element)
        Threads.atomic_max!(max_αϕ, norm_αϕ)
        Threads.atomic_max!(max_βnϕ, norm_βnϕ)
        error0list[element_index] = error0
        error1list[element_index] = error1
    end
    error0list = error0list ./ max_αϕ[]
    error1list = error1list ./ max_βnϕ[]
    error0, element_max_index = findmax(error0list) 
    error1 = maximum(error1list)

    # neighbor error
    error0_neighbor = 0.0
    error1_neighbor = 0.0
    for neighbor_index in neighbors[element_max_index]
        error0_n, error1_n, _, _ = compute_element_error(dimdata, element_max_index, neighbor_index, k, src, pol, n_nodes_per_element)
        error0_neighbor = max(error0_neighbor, error0_n)
        error1_neighbor = max(error1_neighbor, error1_n)
    end
    error0_neighbor /= max_αϕ[]
    error1_neighbor /= max_βnϕ[]
    @info """h: $(dimdata.hmax), nElem: $(get_number_of_elements(dimdata.gquad)), k: $k, EO: $ELEM_ORDER, QO: $QUADRATURE_ORDER, 
    error0: $error0, error1: $error1, error0_neighbor: $error0_neighbor, error1_neighbor: $error1_neighbor"""
end

##
ELEM_ORDER = 2
HMAX = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
QUADRATURE_ORDER = 6
k = 1     # Wavenumber
n_src = 26  # number of Lebedev sources
r_factor = 5  # radius factor for Lebedev sources
for h in HMAX
    convergence_interpolant(ELEM_ORDER, h, QUADRATURE_ORDER, k, n_src, r_factor)
end


