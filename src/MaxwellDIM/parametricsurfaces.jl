function generate_gquad(mesh)
    @assert length(mesh.elt2dof) == 1  # single type of element
    old_elements = first(values(mesh.elt2dof))
    elements = [element2element(col) for col in eachcol(old_elements)]
    qnodes = generate_qnodes(mesh, elements)
    return GlobalQuadrature(qnodes, elements)
end
function generate_qnodes(mesh, elements)
    element_index = 0
    dofs = Nystrom.dofs(mesh)
    qnodes = QNode[]
    for el in elements
        element_index += 1
        for index in el 
            qnode = nystromdof2qnode(dofs[index], index, element_index)
            push!(qnodes, qnode)
        end
    end
    return qnodes
end
function nystromdof2qnode(dof, index, element_index)
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