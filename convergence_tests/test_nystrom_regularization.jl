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
BLAS.set_num_threads(Threads.nthreads())
print_threads_info()

# Load a mesh with quadratic elements
ELEM_ORDER = 2
HMAX = 1.7
mesh_filename = "test/meshes/sphere2.geo"
mesh = read_gmsh_geo(mesh_filename, h=HMAX, order=ELEM_ORDER);

## Generates a DimData
QUADRATURE_ORDER = 4
k = 2π             # Wavenumber
h = 1/2         # mean curvature
k2 = k + im*0.4*h^(2/3)*k^(1/3)
n_src = 50  # number of Lebedev sources
r_factor = 5  # radius factor for Lebedev sources
α = 1       # DIM α parameter
β = 2*k*k2     # DIM β parameter
indirect = false  # direct formulation
dimdata = generate_dimdata(mesh, qorder=QUADRATURE_ORDER, k=k,
                             n_src=n_src, α=α, β=β, r=r_factor, indirect=indirect);
DM.assemble_interpolant_LQ_matrices!(dimdata)

## Dimdata child
α = 0     # DIM α parameter
β = 1     # DIM β parameter
indirect = false  # direct formulation
dimdata_child = generate_dimdata(mesh, qorder=QUADRATURE_ORDER, k=k2,
                             n_src=n_src, α=α, β=β, r=r_factor, indirect=indirect);
DM.assemble_interpolant_LQ_matrices!(dimdata_child)
fill!(dimdata_child.density_coeff_data, 0); # density of double layer not used

## Evaluate forwardmap
struct NFM
end
Base.size(::NFM) = 2 .* (get_number_of_qnodes(dimdata), get_number_of_qnodes(dimdata))
Base.size(x::NFM, i::Integer) = size(x)[i] 
Base.eltype(::NFM) = ComplexF64
LinearAlgebra.mul!(y, ::NFM, b) = y .= eval_map(b, dimdata, dimdata_child)

function eval_map(ϕvec::AbstractVector{ComplexF64}, dimdata, dimdata_child)
    # eval child (indirect)
    @assert length(dimdata_child.density2_coeff_data) == length(ϕvec)
    copyto!(dimdata_child.density2_coeff_data, ϕvec) # for single layer operator
    iop_child = DM.compute_integral_operator!(dimdata_child)

    # eval parent 
    DM.project_field_onto_surface_density!(dimdata, iop_child, iop_child) # for single layer operator
    copyto!(dimdata.density_coeff_data, ϕvec)  # for double layer operator
    nystrom_iop_parent = DM.compute_exterior_nystrom_integral_operator!(dimdata)
    return reinterpret(ComplexF64, nystrom_iop_parent)
end

## Field produced by electric dipole
src = Point3D(0.1, -0.1, 0.2)    # dipole location
pol = Point3D(1, 1, 1)    # dipole polarization  
Efield = similar(dimdata.gquad.qnodes, ComplexPoint3D)
for i in eachindex(dimdata.gquad.qnodes)
    qnode = get_qnode(dimdata.gquad, i)
    yi, _, _, _ = get_qnode_data(qnode)
    Efield[i] = DM.electric_dipole_electric_field(yi, src, k, pol)
end

##
rhs = DM.compute_nystrom_maxwell_rhs(dimdata_child, Efield)
@info "Solving system..."
M = NFM()  # object for gmres
result = copy(rhs)
DM.gmres!(result, M, rhs; log=true, verbose=true, maxiter=100, restart=250, abstol=1e-4)
# force computation of interpolant
# for future evaluations
nres = eval_map(result, dimdata, dimdata_child);
@info "Done."

##
# Evaluate in points on a sphere
npoints = 50
r = 5
θrange = range(0, 2π, length=npoints)[1:end-1]
ϕrange = range(0, π, length=npoints)[1:end-1]
xlist = [Point3D(r*sin(ϕ)*cos(θ), 
                 r*sin(ϕ)*sin(θ), 
                 r*cos(ϕ)) for ϕ in ϕrange for θ in θrange]
@info "Computing potential..."
E_exact = [DM.electric_dipole_electric_field(x, src, k, pol) for x in xlist]
E_approx = DM.compute_potencial(dimdata, xlist)
@info "Done."
errors = norm.(E_exact-E_approx)/maximum(norm.(E_exact))
error = maximum(errors)
@show error;
