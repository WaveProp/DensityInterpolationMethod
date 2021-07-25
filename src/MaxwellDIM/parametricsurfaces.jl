function generate_gquad(mesh)
    @assert length(mesh.elt2dof) == 1  # single type of element
    old_elements = first(values(mesh.elt2dof))
    qnodes = [nystromdof2qnode(dof) for dof in Nystrom.dofs(mesh)]
    elements = [element2element(col) for col in eachcol(old_elements)]
    return GlobalQuadrature(qnodes, elements)
end
function nystromdof2qnode(dof)
    index = 0
    element_index = 0
    qnode = Nystrom.coords(dof)
    weigth = Nystrom.weight(dof)
    jacobian = Nystrom.jacobian(dof)
    normal = Nystrom.normal(dof)
    return QNode(index, element_index, qnode, weigth, jacobian, normal)
end
function element2element(element_old)
    return Vector{Int64}(element_old)
end

function parametric_sphere(;radius=0.5)
    geo = Sphere(;radius)
    Ω = Domain(geo)
    Γ = boundary(Ω)
    return Γ
end

function nystrom_gquad(Γ; n, order)
    M = meshgen(Γ,(n,n))
    mesh = Nystrom.NystromMesh(view(M,Γ);order)
    gquad = generate_gquad(mesh)
    return gquad
end