using StaticArrays
using LinearAlgebra
using DensityInterpolationMethod
using DensityInterpolationMethod.Utils
using DensityInterpolationMethod.IO
using DensityInterpolationMethod.Mesh
using DensityInterpolationMethod.Integration
using DensityInterpolationMethod.MaxwellDIM
const DM = DensityInterpolationMethod.MaxwellDIM
BLAS.set_num_threads(Threads.nthreads())
print_threads_info()

function convergence_nystrom(eval_points, HMAX, QUADRATURE_ORDER)
    # Load a mesh with quadratic elements
    ELEM_ORDER = 2
    mesh_filename = "test/meshes/sphere3.geo"
    mesh = read_gmsh_geo(mesh_filename, h=HMAX, order=ELEM_ORDER, verbosity=false);

    # Generate DimData
    k = 1     # Wavenumber
    n_src = 26  # number of Lebedev sources
    r_factor = 5  # radius factor for Lebedev sources
    α = 1       # DIM α parameter
    β = 0      # DIM β parameter
    dimdata = generate_dimdata(mesh, qorder=QUADRATURE_ORDER, k=k,
                                n_src=n_src, α=α, β=β, r=r_factor);
    DM.initialize!(dimdata)     
    
    # Field produced by electric dipole
    src = Point3D(0.1, -0.1, 0.2)    # dipole location
    pol = Point3D(1, 1, 1)    # dipole polarization  
    Efield = similar(dimdata.gquad.qnodes, ComplexPoint3D)
    for i in eachindex(dimdata.gquad.qnodes)
        qnode = get_qnode(dimdata.gquad, i)
        yi, _, _, _ = get_qnode_data(qnode)
        Efield[i] = DM.electric_dipole_electric_field(yi, src, k, pol)
    end

    # Solve using LU
    rhs = DM.compute_nystrom_maxwell_rhs(dimdata, Efield)
    @info "Computing matrix..."
    @info "Done."
    M = DM.compute_nystrom_maxwell_matrix(dimdata, DM.ExteriorNystromFormulation)
    nUnk = get_number_of_qnodes(dimdata) * DIMENSION2
    @info "Solving..." nUnk
    DM.solve_nystrom_GMRES!(dimdata, M, rhs, log=true, verbose=true, maxiter=100, restart=150, abstol=1e-6)
    @info "Done."

    # Evaluate errors on a sphere
    @info "Computing potential..."
    E_exact = [DM.electric_dipole_electric_field(x, src, k, pol) for x in eval_points]
    E_approx = DM.compute_potencial(dimdata, eval_points)
    @info "Done."
    error = maximum(norm.(E_exact-E_approx))/maximum(norm.(E_exact))

    # print results
    nElem = get_number_of_elements(dimdata.gquad)
    h = dimdata.hmax
    QO = QUADRATURE_ORDER
    @info "results: h= $h, nElem= $nElem, nUnk= $nUnk, QO= $QO, error= $error\n"
    println()
end

## Eval points
npoints = 50
r = 5  # sphere radius
θrange = range(0, 2π, length=npoints)[1:end-1]
ϕrange = range(0, π, length=npoints)[1:end-1]
eval_points = [Point3D(r*sin(ϕ)*cos(θ), 
                       r*sin(ϕ)*sin(θ), 
                       r*cos(ϕ)) for ϕ in ϕrange for θ in θrange];

##
HMAX = [1, 0.8, 0.6, 0.5, 0.45, 0.3, 0.25]
HMAX = [0.24, 0.22]
QUADRATURE_ORDER = 2
for h in HMAX
    convergence_nystrom(eval_points, h, QUADRATURE_ORDER)
end