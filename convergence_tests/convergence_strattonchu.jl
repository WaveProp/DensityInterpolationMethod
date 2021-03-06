using StaticArrays
using LinearAlgebra
using Statistics
using BenchmarkTools
using Plots
using DensityInterpolationMethod
using DensityInterpolationMethod.Utils
using DensityInterpolationMethod.IO
using DensityInterpolationMethod.Mesh
using DensityInterpolationMethod.Integration
using DensityInterpolationMethod.MaxwellDIM
const DM = DensityInterpolationMethod.MaxwellDIM
print_threads_info()

function convergence_strattonchu(ELEM_ORDER, HMAX, QUADRATURE_ORDER, k, n_src, r_factor)
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
    DM.assemble_interpolant_LQ_matrices!(dimdata)

    # Field produced by electric dipole
    src = Point3D(3, 3, 3)    # dipole location
    pol = Point3D(1, 1, 1)    # dipole polarization  
    γ₀Efields = similar(dimdata.gquad.qnodes, ComplexPoint3D)
    γ₁Efields = similar(dimdata.gquad.qnodes, ComplexPoint3D)
    for i in eachindex(dimdata.gquad.qnodes)
        qnode = get_qnode(dimdata.gquad, i)
        yi, _, _, ni = get_qnode_data(qnode)
        γ₀Efields[i] = DM.single_layer_kernel(yi, src, k, ni, pol)
        γ₁Efields[i] = DM.double_layer_kernel(yi, src, k, ni, pol)
    end
    # Project fields in each DimData
    DM.project_field_onto_surface_density!(dimdata, γ₀Efields, γ₁Efields)

    # Error
    result = -2 .* DM.compute_integral_operator!(dimdata)
    error_list = norm.(result.-γ₀Efields)/maximum(norm.(γ₀Efields))
    error_max = maximum(error_list)
    error_95 = quantile(error_list, 0.95)
    nqnodes = get_number_of_qnodes(dimdata)
    return dimdata.hmax, nqnodes, error_max

    #println("""h: $(dimdata.hmax), nElem: $(get_number_of_elements(dimdata.gquad)), k: $k, EO: $ELEM_ORDER, QO: $QUADRATURE_ORDER, 
    #error_max: $error_max, error_95: $error_95""")
end

##
ELEM_ORDER = 2
HMAX = [0.5, 0.4, 0.3, 0.2]
QUADRATURE_ORDER = 4
k = 3.3     # Wavenumber
n_src = 26  # number of Lebedev sources
r_factor = 5  # radius factor for Lebedev sources
h_list = zeros(length(HMAX))
nqnodes_list = zeros(length(HMAX))
err_list = zeros(length(HMAX))
for i in eachindex(HMAX)
    H = HMAX[i]
    h, nqnodes, err = convergence_strattonchu(ELEM_ORDER, H, QUADRATURE_ORDER, k, n_src, r_factor)
    h_list[i] = h
    nqnodes_list[i] = nqnodes
    err_list[i] = err
    @info "\n" h nqnodes err
end

## Plot 1
fig = plot(xlabel="sqrt{N}",ylabel="error",title="Stratton-Chu identity k=$k",
           xscale=:log10,yscale=:log10,framestyle=:box,xtickfontsize=10,ytickfontsize=10)
# retrieve data
ndofs = nqnodes_list
errs  = err_list
sqrt_ndofs = sqrt.(ndofs)
# plot
p = 3 #(sqrt(8*qnumber+1)-1)/2
isinteger(p) || @warn "p is not an integer" 
p = Int(p)
plot!(fig,sqrt_ndofs,errs,m=:o,label="p=$p")
# reference slopes
s = p-1 #p-1 # theoretical slope
cc = errs[end]*sqrt_ndofs[end]^s
plot!(fig,sqrt_ndofs,cc./sqrt_ndofs.^s,ls=:dash,lw=2,label="")
display(fig)

## Plot 2
fig = plot(xlabel="h",ylabel="error",title="Stratton-Chu identity k=$k",
           xscale=:log10,yscale=:log10,framestyle=:box,xtickfontsize=10,ytickfontsize=10)
# retrieve data
errs  = err_list
sqrt_ndofs = h_list
# plot
p = 3 #(sqrt(8*qnumber+1)-1)/2
isinteger(p) || @warn "p is not an integer" 
p = Int(p)
plot!(fig,sqrt_ndofs,errs,m=:o,label="p=$p")
# reference slopes
s = p-1 #p-1 # theoretical slope
cc = errs[end]*sqrt_ndofs[end]^(-s)
plot!(fig,sqrt_ndofs,cc./sqrt_ndofs.^(-s),ls=:dash,lw=2,label="")
display(fig)

