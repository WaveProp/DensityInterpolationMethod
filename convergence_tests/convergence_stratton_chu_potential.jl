using StaticArrays
using LinearAlgebra
using BenchmarkTools
using DensityInterpolationMethod
using DensityInterpolationMethod.Utils
using DensityInterpolationMethod.IO
using DensityInterpolationMethod.Mesh
using DensityInterpolationMethod.Integration
using DensityInterpolationMethod.MaxwellDIM
const DM = DensityInterpolationMethod.MaxwellDIM

function convergence_potential(eval_pts, ELEM_ORDER, HMAX, QUADRATURE_ORDER, k, n_src, r_factor)
    # Load mesh 
    mesh_filename = "test/meshes/sphere1.geo"
    mesh = read_gmsh_geo(mesh_filename, h=HMAX, order=ELEM_ORDER, verbosity=false);  

    # Generates a DimDataDirect
    α = 1       # DIM α parameter
    β = 1       # DIM β parameter
    indirect = false  # direct formulation
    dimdata = DM.generate_dimdata(mesh, qorder=QUADRATURE_ORDER, k=k,
                                n_src=n_src, α=α, β=β, r=r_factor, indirect=indirect);
    # Compute DIM matrices   
    DM.assemble_dim_matrices(dimdata)

    # Field produced by electric dipole
    src = Point3D(-0.1, 0.3, -0.2)    # dipole location
    pol = Point3D(1, -1, 1)    # dipole polarization  
    γ₀Efields = similar(dimdata.gquad.qnodes, ComplexPoint3D)
    γ₁Efields = similar(dimdata.gquad.qnodes, ComplexPoint3D)
    for i in eachindex(dimdata.gquad.qnodes)
        qnode = get_qnode(dimdata.gquad, i)
        yi, _, _, ni = get_qnode_data(qnode)
        γ₀Efields[i] = DM.single_layer_kernel(yi, src, k, ni, pol)
        γ₁Efields[i] = DM.double_layer_kernel(yi, src, k, ni, pol)
    end

    # Get γ₀
    println("gamma 0")
    DM.project_field_onto_surface_density(dimdata, γ₀Efields, γ₁Efields)
    γ₀approx = 2 .* DM.compute_integral_operator(dimdata)

    # Get γ₁
    println("gamma 1")
    DM.project_field_onto_surface_density(dimdata, γ₁Efields, k^2 .* γ₀Efields)
    γ₁approx = 2 .* DM.compute_integral_operator(dimdata)
    
    # Error
    println("potential")
    DM.project_field_onto_surface_density(dimdata, γ₀approx, γ₁approx)
    E_exact = [DM.electric_dipole_electric_field(x, src, k, pol) for x in eval_pts]
    E_approx = DM.compute_potencial(dimdata, eval_pts)
    error = maximum(norm.(E_exact-E_approx))/maximum(norm.(E_exact))
    println("h: $(dimdata.hmax), nElem: $(get_number_of_elements(dimdata.gquad)), k: $k, EO: $ELEM_ORDER, QO: $QUADRATURE_ORDER, error: $error")
end

# Eval points
npoints = 50
r = 3
θrange = range(0, 2π, length=npoints)[1:end-1]
ϕrange = range(0, π, length=npoints)[1:end-1]
eval_pts = [Point3D(r*sin(ϕ)*cos(θ), 
                 r*sin(ϕ)*sin(θ), 
                 r*cos(ϕ)) for ϕ in ϕrange for θ in θrange]

##
ELEM_ORDER = 2
HMAX = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
QUADRATURE_ORDER = 6
k = 1     # Wavenumber
n_src = 26  # number of Lebedev sources
r_factor = 5  # radius factor for Lebedev sources
for h in HMAX
    convergence_potential(eval_pts, ELEM_ORDER, h, QUADRATURE_ORDER, k, n_src, r_factor)
end


