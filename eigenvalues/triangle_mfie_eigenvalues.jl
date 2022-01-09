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
using Plots
BLAS.set_num_threads(Threads.nthreads())
print_threads_info()

##
k = 3π
sph_radius = 1
n_src = 50         # number of interpolant sources
α = 1              # MFIE constant
β = 0              # EFIE constant
qorder = 5         # quadrature order 
h = 0.4              # mesh size

##
mesh_filename = "test/meshes/sphere1.geo"
mesh = read_gmsh_geo(mesh_filename, h=h, order=2, verbosity=false);
dimdata = generate_dimdata(mesh, qorder=qorder, k=k,n_src=n_src, α=α, β=β);
DM.initialize!(dimdata) 
L = DM.compute_nystrom_maxwell_matrix(dimdata, DM.ExteriorNystromFormulation)

##
eigenvalues = eigvals!(L)
fig = scatter(eigenvalues,label="eigenvalues",framestyle=:box,xtickfontsize=10,ytickfontsize=10)
title!("k= $k, qorder=$qorder, h=$h, n_src=$n_src")
scatter!(fig, [0+0im], markersize=7, label="origin")
xlabel!(fig, "Re λ")
ylabel!(fig, "Im λ")