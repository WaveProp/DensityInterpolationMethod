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

# Convergence test using Carlos'
# operators and definitions

##
k = 1.3
sph_radius = 2
Γ = DM.parametric_sphere(;radius=sph_radius)

# exact solution
xs   = SVector(0.1,0.2,0.3) 
c    = SVector(1+im,-2.,3.)
γₒG  = (qnode) -> DM.single_layer_kernel(qnode.qnode, xs, k, qnode.normal, c);
γ₁G  = (qnode) -> DM.double_layer_kernel(qnode.qnode, xs, k, qnode.normal, c);

## Indirect formulation
n_src = 50         # number of interpolant sources
qorder = 7         # quadrature order 
ndofs = Float64[]
γ₀errs = Float64[]
γ₁errs = Float64[];
##
# Load a mesh with quadratic elements
for n in [4,8,12,16]
    gquad = DM.nystrom_gquad(Γ; n, order=qorder)
    γ₀E   = γₒG.(qnode for qnode in gquad.qnodes)
    γ₁E   = γ₁G.(qnode for qnode in gquad.qnodes)
    @info "Computing matrices..."
    T,K   = DM.single_doublelayer_dim(gquad; k, n_src)
    T = T.blocks
    K = K.blocks
    @info "Computing forward map..."
    γ₀E = reinterpret(ComplexF64, γ₀E)
    γ₁E = reinterpret(ComplexF64, γ₁E)
    γ₀Eₐ = 2*(T*γ₁E + K*γ₀E)
    γ₁Eₐ = 2*(K*γ₁E + k^2*T*γ₀E)
    
    γ₀err = norm(γ₀Eₐ - γ₀E, Inf) / norm(γ₀E, Inf)
    γ₁err = norm(γ₁Eₐ - γ₁E, Inf) / norm(γ₁E, Inf)
    ndof = get_number_of_qnodes(gquad)
    push!(ndofs, ndof)  
    push!(γ₀errs, γ₀err)    
    push!(γ₁errs, γ₁err)     
    @info "" ndof γ₀err γ₁err
end

## Plot
sqrt_ndofs = sqrt.(ndofs)
fig = plot(sqrt_ndofs,γ₀errs,xscale=:log10,yscale=:log10,m=:o,label="γ₀error",lc=:black)
plot!(sqrt_ndofs,γ₁errs,xscale=:log10,yscale=:log10,m=:o,label="γ₁error",lc=:red)
title = "k=$k, qorder=$qorder"
plot!(xlabel="√ndofs",ylabel="error",title=title)
for p in 1:5
    cc = γ₀errs[end]*sqrt_ndofs[end]^p
    plot!(fig,sqrt_ndofs,cc./sqrt_ndofs.^p,xscale=:log10,yscale=:log10,label="h^$p",ls=:dash)
end
display(fig)