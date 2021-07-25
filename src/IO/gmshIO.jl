"""
GMSH IO methods.
"""

"""
    read_geo(fname::String; h=nothing, order=nothing)

Read a `.geo` file and generate a [`GenericMesh`](@ref).
Assumes that the mesh correspond to a surface in 3D.
"""
function read_gmsh_geo(fname; h=nothing, order=nothing, verbosity=true)
    assert_extension(fname, ".geo")    
    gmsh.initialize()
    if !verbosity _gmsh_set_verbosity(verbosity) end
    gmsh.open(fname)    

    if !(h === nothing) _gmsh_set_meshsize(h) end
    if !(order === nothing) _gmsh_set_meshorder(order) end

    gmsh.model.mesh.generate(DIMENSION2)  # mesh surfaces
    mesh = _initialize_mesh()
    gmsh.finalize()
    return mesh
end

function _gmsh_set_meshsize(hmax, hmin=hmax)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hmin)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hmax)
end

function _gmsh_set_meshorder(order)
    gmsh.option.setNumber("Mesh.ElementOrder", order)
end

function _gmsh_set_verbosity(i) 
    gmsh.option.setNumber("General.Verbosity",i)
end

"""
    _initialize_mesh()

Performs all the GMSH API calls to extract the information necessary to
construct the mesh. This function assumes `gmsh` has been initialized. Also,
assumes that the elements correspond to surface elements only.
"""
function _initialize_mesh()
    nodeTags, nodes, _ = gmsh.model.mesh.getNodes()
    n_nodes = length(nodeTags)
    @assert n_nodes>0 "Mesh is empty!"

    # Process node tags and nodes
    nodes = reinterpret(Point3D, nodes) |> collect
    tag2node = Dict(nodeTags[i] => nodes[i] for i = 1:n_nodes)  # (NodeTag => node)
    
    # Get element data (surface elements only)
    elementTypes, _, elNodeTagsList = gmsh.model.mesh.getElements(DIMENSION2)

    # Generate mesh
    mesh = GenericMesh()
    for (gmsh_etype, elNodeTags) in zip(elementTypes, elNodeTagsList)
        # Map gmsh type tags to actual internal types
        etype, nodes_per_element = _get_gmsh_element_data(gmsh_etype)
        elements = etype[]

        # Process element node tags
        elNodeTags = reshape(elNodeTags, nodes_per_element, :)

        # Generate elements
        for tags in eachcol(elNodeTags)
            el = _generate_element(etype, tag2node, tags) 
            push!(elements, el)     # Push into list
        end
        mesh.etype2elements[etype] = elements   # Push into mesh dict
    end
    return mesh
end

"""
    _get_gmsh_element_data(etag)

Function that given a gmsh element type `etag` (encoded as an integer), 
returns the internal element type (encoded as a Type{<:AbstractElement}) and 
the number of nodes of the element. This function assumes `gmsh` has been 
initialized.
"""
function _get_gmsh_element_data(etag)
    name,_,_,numNodes,_,_  = gmsh.model.mesh.getElementProperties(etag)
    numNodes = Int64(numNodes)   # convert to Int64 (reshape doesn't work on Int32)
    if occursin("Triangle 3", name)
        etype = FlatTriangleElement
    elseif occursin("Triangle 6", name)
        etype = QuadraticTriangleElement
    else
        notimplemented()   
    end    
    return etype, numNodes
end    

"""
    _generate_element(etype, nodes, tag2index, tags)

Returns an element of type `etype`, given the list of nodes `nodes`,
the dictionary `tag2index`, that maps node tags into node indices,
and the element node tags `tags`.
"""
function _generate_element(etype, tag2node, tags)
    element_nodes = (tag2node[t] for t in tags)
    element = etype(element_nodes, tags)
    return element
end

"""
    get_qrule_from_gmsh(element_name, qrule_name)

Returns the qnodes `x` and qweights `w` for a given `element_name` (e.g. "Triangle") 
and `qrule_name` (e.g. "Gauss4"). This function calls the Gmsh API. 
"""
function get_qrule_from_gmsh(element_name, qrule_name)
    gmsh.initialize()
    # Order of the element, higher order elements
    # should have the same qnodes and qweights.
    order = 1    
    etype = gmsh.model.mesh.getElementType(element_name, order)
    x, w = gmsh.model.mesh.getIntegrationPoints(etype, qrule_name)
    gmsh.finalize()
    xdata, wdata = _reshape_qrule_data(x, w)
    return xdata, wdata
end
function _reshape_qrule_data(x, w)
    n_qnodes = length(w)
    xmatrix = reshape(x, DIMENSION3, :)
    xdata = SVector{n_qnodes}([Point2D(i[1], i[2]) for i in eachcol(xmatrix)])
    wdata = SVector{n_qnodes}(w)
    return xdata, wdata
end

"""
    gmsh_sphere(;radius=0.5,center=(0,0,0),dim=3,h=radius/10,order=1)

Use `gmsh` API to generate a sphere and return `mesh::GenericMesh`.
"""
function gmsh_sphere(;radius=0.5,center=(0., 0., 0.),h=radius/10,order=1,verbosity=true)
    gmsh.initialize()
    if !verbosity _gmsh_set_verbosity(verbosity) end
    _gmsh_set_meshsize(h)
    _gmsh_set_meshorder(order)
    gmsh.model.occ.addSphere(center..., radius)
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(DIMENSION2) # mesh surfaces
    mesh = _initialize_mesh()
    gmsh.finalize()
    return mesh
end