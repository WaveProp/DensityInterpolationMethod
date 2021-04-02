"""
File for testing the source code
"""

using Plots
using DensityInterpolationMethod
using DensityInterpolationMethod.Mesh
using DensityInterpolationMethod.Utils 

const ELEM_ORDER = 2
const HMAX = 1
mesh_filename = "examples/meshes/sphere1.geo"

plotlyjs()   # PlotlyJS backend
scatter([1, 2], [3, 4], [5, 6])

struct hola2
    v
end

h = hola2([4, 4, 4, 4])
h2 = hola(h)

@recipe f(object::hola) = object.v
@recipe f(object::hola2) = object.v

plot(h2)



mesh = read_gmsh_geo(mesh_filename, h=HMAX, order=ELEM_ORDER)
