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
k = 3.3
sph_radius = 2
Γ = DM.parametric_sphere(;radius=sph_radius)

# evaluation mesh
eval_npoints = 50
eval_r = 5
eval_θrange = range(0, 2π, length=eval_npoints)[1:end-1]
eval_ϕrange = range(0, π, length=eval_npoints)[1:end-1]
eval_mesh = [Point3D(eval_r*sin(ϕ)*cos(θ), 
                 eval_r*sin(ϕ)*sin(θ), 
                 eval_r*cos(ϕ)) for ϕ in eval_ϕrange for θ in eval_θrange]


# exact solution
xs   = SVector(0.1,0.2,0.3) 
c    = SVector(1+im,-2.,3.)
G    = (x,y) -> DM._green_tensor(x, y, k)
γₒG  = (qnode) -> DM.single_layer_kernel(qnode.qnode, xs, k, qnode.normal, c);
#G    = (x,y) -> DM._curl_green_tensor(x, y, k)
#γₒG  = (qnode) -> DM.double_layer_kernel(qnode.qnode, xs, k, qnode.normal, c);
E    = (dof) -> G(dof,xs)*c
exa  = E.(eval_mesh);

## Indirect formulation
n_src = 38         # number of interpolant sources
α = 1              # MFIE constant
β = im*k/sph_radius              # EFIE constant
qorder = 5         # quadrature order 
ndofs = Float64[]
errs = Float64[]
iterative = false;
##
for n in [2,4,8,12,16]
    global gquad
    gquad = DM.nystrom_gquad(Γ; n, order=qorder)
    γ₀E   = γₒG.(qnode for qnode in gquad.qnodes)
    S,D   = DM.single_doublelayer_dim(gquad; k, n_src)
    N,J,dualJ = DM.diagonal_ncross_jac_matrix(gquad)

    @info "Assembling matrix..."
    global L
    L = DM.assemble_dim_exterior_nystrom_matrix(gquad, α, β, D, S)
    @info "Solving..."
    if iterative
        rhs   = dualJ*γ₀E
        Pl = DM.get_blockdiag_precond(gquad, L) 
        ϕ_coeff = DM.solve_GMRES(L, rhs; Pl, verbose=true,maxiter=600,restart=600,abstol=1e-8)
    else
        rhs   = reinterpret(ComplexF64, dualJ*γ₀E)
        ϕ_coeff = reinterpret(ComplexPoint2D, L\rhs)
    end
    ϕ     = J*ϕ_coeff
    Spot  = DM.maxwellCFIE_SingleLayerPotencial(k, gquad)
    Dpot  = DM.maxwellCFIE_DoubleLayerPotencial(k, gquad)
    Eₐ    = (x) -> α*Dpot(ϕ,x) + β*Spot(N*ϕ,x)
    er    = (Eₐ.(eval_mesh) - exa)/norm(exa,Inf)
    ndof = get_number_of_qnodes(gquad)
    err = norm(er,Inf)

    push!(ndofs, ndof)   
    push!(errs, err)     
    @info "" ndof err
end

## Plot
sqrt_ndofs = sqrt.(ndofs)
fig = plot(sqrt_ndofs,errs,xscale=:log10,yscale=:log10,m=:o,label="error",lc=:black)
title = "k=$k, α=$α, β=$β, qorder=$qorder, R=$sph_radius"
plot!(xlabel="√ndofs",ylabel="error",title=title)
for p in 1:5
    cc = errs[end]*sqrt_ndofs[end]^p
    plot!(fig,sqrt_ndofs,cc./sqrt_ndofs.^p,xscale=:log10,yscale=:log10,label="h^$p",ls=:dash)
end
display(fig)