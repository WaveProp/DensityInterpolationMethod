using GmshSDK

const GEOMET = 2

gmsh.initialize()


gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
gmsh.model.add("model_name")

#occ:  Open Cascade backend
tag = gmsh.model.occ.addSphere(0, 0, 0, 1)  
gmsh.model.occ.synchronize()

gmsh.model.addPhysicalGroup(2, [tag])  # dimension, [tags]
gmsh.model.mesh.generate(2)  # dimension
gmsh.write("file.msh")


gmsh.finalize()