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

function nystrom_dimdata(Γ; n, order, k=1, α=1, β=1, n_src=14, r=5, indirect=true)
    gquad = nystrom_gquad(Γ; n, order)
    dimdata = _nystrom_dimdata(gquad; k, α, β, n_src, r, indirect)
    return dimdata
end
function _nystrom_dimdata(gquad::GlobalQuadrature; k, α, β, n_src, r, indirect)
    mesh = GenericMesh()
    hmax = 0
    n_qnodes = get_number_of_qnodes(gquad)
    n_elements = get_number_of_elements(gquad)
    # compute source points
    _, bbox_center, bbox_radius = compute_bounding_box(gquad)
    src_radius = r * bbox_radius
    src_list = get_sphere_sources_lebedev(n_src, src_radius, bbox_center)
    n_src = length(src_list)  # update number of source points
    # initialize data
    density_coeff = Vector{ComplexPoint2D}(undef, n_qnodes)
    interpolant_coeff = [Vector{ComplexPoint3D}(undef, n_src) for _ in 1:n_elements]
    integral_op = Vector{ComplexPoint3D}(undef, n_qnodes)
    Lmatrices = [LowerTriangular(Matrix{ComplexF64}(undef, 0, 0)) for _ in 1:n_elements]
    Qmatrices = [Matrix{ComplexF64}(undef, 0, 0) for _ in 1:n_elements]
    Θmatrices = [Matrix{ComplexF64}(undef, 0, 0) for _ in 1:n_qnodes]
    # reinterpreted data
    density_coeff_data = reinterpret(ComplexF64, density_coeff)
    interpolant_coeff_data = [reinterpret(ComplexF64, interpolant_coeff[i]) for i in eachindex(interpolant_coeff)]
    # formulation
    if indirect
        # density2_coeff is not used
        density2_coeff = Vector{ComplexPoint2D}[]
        density2_coeff_data = density_coeff_data
        Formulation = IndirectDimData
    else
        density2_coeff = Vector{ComplexPoint2D}(undef, n_qnodes)
        density2_coeff_data = reinterpret(ComplexF64, density2_coeff)
        Formulation = DirectDimData
    end
    return Formulation(hmax, mesh, gquad, k, α, β, density_coeff, density2_coeff, interpolant_coeff, integral_op,
                   Lmatrices, Qmatrices, Θmatrices, src_list, density_coeff_data, density2_coeff_data, interpolant_coeff_data)
end