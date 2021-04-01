"""
File for testing the source code
"""

using DensityInterpolationMethod
using DensityInterpolationMethod.Mesh
using DensityInterpolationMethod.Utils 

const ELEM_ORDER = 2
const HMAX = 1
mesh_filename = "examples/meshes/sphere1.geo"

mesh = read_gmsh_geo(mesh_filename, h=HMAX, order=ELEM_ORDER)
