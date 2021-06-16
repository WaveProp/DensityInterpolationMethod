using StaticArrays: Iterators
using StaticArrays
using LinearAlgebra
using BenchmarkTools
using DensityInterpolationMethod
using DensityInterpolationMethod.Utils
using DensityInterpolationMethod.IO
using DensityInterpolationMethod.Mesh
using DensityInterpolationMethod.Integration
using DensityInterpolationMethod.MaxwellDIM
using Preconditioners
const DM = DensityInterpolationMethod.MaxwellDIM
BLAS.set_num_threads(Threads.nthreads())
print_threads_info()

function convergence_gmres(HMAX, QUADRATURE_ORDER, k, α)
    println("\n***********************************************************")
    # Load a mesh with quadratic elements
    ELEM_ORDER = 2
    mesh_filename = "test/meshes/sphere2.geo"
    mesh = read_gmsh_geo(mesh_filename, h=HMAX, order=ELEM_ORDER, verbosity=false);

    ## Generates a DimData
    n_src = 50  # number of Lebedev sources
    r_factor = 5  # radius factor for Lebedev sources
    β = im*k     # DIM β parameter
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

    # points on sphere for evaluation
    npoints = 50
    r = 5
    θrange = range(0, 2π, length=npoints)[1:end-1]
    ϕrange = range(0, π, length=npoints)[1:end-1]
    xlist = [Point3D(r*sin(ϕ)*cos(θ), 
                    r*sin(ϕ)*sin(θ), 
                    r*cos(ϕ)) for ϕ in ϕrange for θ in θrange]
    # exact solution
    E_exact = [DM.electric_dipole_electric_field(x, src, k, pol) for x in xlist]

    # function for computing error
    function compute_error()
        E_approx = DM.compute_potencial(dimdata, xlist)
        error = maximum(norm.(E_exact-E_approx)/maximum(norm.(E_exact)))
        return error
    end

    # solve using LU 
    rhs = DM.compute_nystrom_maxwell_rhs(dimdata, Efield)
    M = DM.compute_nystrom_maxwell_matrix(dimdata, DM.ExteriorNystromFormulation)
    DM.solve_nystrom_LU!(dimdata, M, rhs)
    error_lu = compute_error()

    # solve using GMRES
    _, iter_gmres = DM.solve_nystrom_GMRES!(dimdata, M, rhs, log=true, verbose=false, maxiter=100, restart=150, abstol=1e-4)
    error_gmres = compute_error()
    

    # solve using GMRES with block diagonal preconditioner
    # TODO: compute preconditioner when computing system matrix
    V₁ = DM.generate_interpolant_forwardmap_matrix(dimdata, blockmatrix=false)
    mask = V₁ .!= 0
    P = lu(mask .* M)
    _, iter_gmres_block = DM.solve_nystrom_GMRES!(dimdata, M, rhs, Pl=P, log=true, verbose=false, maxiter=100, restart=150, abstol=1e-4)
    error_gmres_block = compute_error()

    # print results
    @info "Results" dimdata.hmax QUADRATURE_ORDER k α error_lu iter_gmres error_gmres iter_gmres_block error_gmres_block
end             

## 
HMAX = 1.7
QUADRATURE_ORDER = [2, 4, 6]
klist = [1, 2π]
αlist = [0]
for (q, k, α) in Iterators.product(QUADRATURE_ORDER, klist, αlist)
    convergence_gmres(HMAX, q, k, α)
end
