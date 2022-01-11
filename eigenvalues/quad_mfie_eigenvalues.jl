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
k = π
sph_radius = 1
n_src = 26         # number of interpolant sources
α = 1              # MFIE constant
β = 0              # EFIE constant
qorder = 6         # quadrature order 
n = 4    # number of patches

##
Γ = DM.parametric_sphere(;radius=sph_radius)
gquad = DM.nystrom_gquad(Γ; n, order=qorder)
_,K = DM.single_doublelayer_dim(gquad; k, n_src)
L = 0.5I + DM.get_matrix_from_pseudoblockmatrix(K)

##
eigenvalues = eigvals!(L)
fig = scatter(eigenvalues,label="eigenvalues",framestyle=:box,xtickfontsize=10,ytickfontsize=10)
title!("k= $k, qorder=$qorder, n=$n, n_src=$n_src, plane")
scatter!(fig, [0+0im], markersize=7, label="origin")
xlabel!(fig, "Re λ")
ylabel!(fig, "Im λ")