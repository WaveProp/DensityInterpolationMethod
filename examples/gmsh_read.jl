using GmshSDK

const GEO_DIM = 2
const ELEM_ORDER = 2
const HMAX = 1

gmsh.initialize()
gmsh.open("examples/meshes/sphere1.geo")
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", HMAX)
gmsh.option.setNumber("Mesh.ElementOrder", ELEM_ORDER)
gmsh.model.mesh.generate(GEO_DIM) 

nodeTags, coord, _ = gmsh.model.mesh.getNodes()
nodes_ = reinterpret(Point3D, coord) |> collect
elementTypes, elementTags, elnodeTags = gmsh.model.mesh.getElements(GEO_DIM)

n_nodes = length(nodeTags)
nodetagsdict = Dict(nodeTags[i] => i for i in 1:n_nodes)


#gmsh.model.mesh.getElementType("Triangle", 1) 2
#gmsh.model.mesh.getElementType("Triangle", 2) 9
#gmsh.model.mesh.getElementType("Point", 3) 15
#gmsh.model.mesh.getElementType("Line", 1) 1


gmsh.finalize()