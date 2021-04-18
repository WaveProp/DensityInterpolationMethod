using StaticArrays
using DensityInterpolationMethod
using DensityInterpolationMethod.IO
using DensityInterpolationMethod.Integration
using DensityInterpolationMethod.MaxwellDIM

# Load a mesh with quadratic elements
ELEM_ORDER = 2.0
HMAX = 0.5
mesh_filename = "test/meshes/sphere1.geo"
mesh = read_gmsh_geo(mesh_filename, h=HMAX, order=ELEM_ORDER);

# Generates a DimData
# with a quadrature of order QUADRATURE_ORDER
QUADRATURE_ORDER = 2
K = 2       # Wavenumber
dimdata = generate_dimdata(mesh; qorder=QUADRATURE_ORDER, k=K);

# Set the surface density `ϕ` equal to `τ₁`, 
# where `τ₁` is the (first) tangent vector
for i in eachindex(dimdata.ϕcoeff)
    dimdata.ϕcoeff[i] = SVector(1, 0)   # [τ₁coeff, τ₂coeff]
end

# Compute density interpolant coefficients for element 1
element_index = 1
DensityInterpolationMethod.MaxwellDIM.
    construct_density_interpolant(dimdata, element_index)
ccoeff = dimdata.ccoeff[element_index]




