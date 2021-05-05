"""
Structures that contains the data for the 
Density Interpolation Method, direct formulation (Stratton-Chu).
"""

"""
    struct DimDataDirect

Structure that contains all necessary data
for the Density Interpolation Method.
"""
struct DimDataDirect <: AbstractDimData
    # Mesh and quadrature data
    hmax::Float64                  # Maximum element size
    mesh::GenericMesh              # Contains the elements parametrization
    gquad::GlobalQuadrature        # Contains quadrature data

    # Parameters (maybe move this to a mutable struct?)
    k::Float64       # Wavenumber               
    α::Float64       # Density interpolant `α` parameter  
    β::Float64       # Density interpolant `β` parameter 

    # Surface density `ϕ` at quadrature nodes `yⱼ`,
    # for operators K and T.
    # Each entry `(ϕ₁, ϕ₂)` correspond to the coefficients
    # in `ϕ(yⱼ) = ϕ₁τ₁ + ϕ₂τ₂`, where `τ₁,τ₂` are the tangential 
    # vectors at `yⱼ`.
    Kϕcoeff::Vector{SVector{2, ComplexF64}}
    Tϕcoeff::Vector{SVector{2, ComplexF64}}

    # Coefficients of the density interpolant `Φ` at element `eⱼ`.
    # Each entry `cⱼ` is the vector of coefficients of `Φ` at `eⱼ`.
    ccoeff::Vector{Vector{ComplexF64}}

    # Integral operator `C̃_{α,β}[ϕ]` value at quadrature nodes.
    integral_op::Vector{ComplexPoint3D}

    # LQ matrices for constructing the density interpolant `Φ`,
    # for each element.
    Lmatrices::Vector{LowerTriangular{ComplexF64, Matrix{ComplexF64}}}
    Qmatrices::Vector{Matrix{ComplexF64}}

    # List of source points for constructing the
    # density interpolant `Φ`.
    src_list::Vector{Point3D}
end

"""
    generate_dimdata_direct(mesh::GenericMesh; qorder=2, k=1, α=1, β=1, n_src=14, r=5)

Generates a `DimDataDirect` structure from `mesh::GenericMesh`.

# Arguments
- `mesh::GenericMesh`: mesh of elements parametrization.
- `qorder=2`: quadrature order.
- `k=1`: wavenumber.
- `α=1`: density interpolant `α` parameter.
- `β=1`: density interpolant `β` parameter.
- `n_src=14`: number of density interpolant source points.
- `r=5`: radius factor of density interpolant source points sphere.
"""
function generate_dimdata_direct(mesh::GenericMesh; qorder=2, k=1, α=1, β=1, n_src=14, r=5)
    hmax = compute_hmax(mesh)
    gquad = generate_globalquadrature(mesh, order=qorder)
    n_qnodes = get_number_of_qnodes(gquad)
    n_elements = get_number_of_elements(gquad)
    
    Kϕcoeff = Vector{SVector{2, ComplexF64}}(undef, n_qnodes)
    Tϕcoeff = Vector{SVector{2, ComplexF64}}(undef, n_qnodes)
    n_ccoef = DIMENSION3 * n_src   # density interpolation coeffs: 3 per src point
    ccoeff = [Vector{ComplexF64}(undef, n_ccoef) for _ in 1:n_elements]
    integral_op = Vector{ComplexPoint3D}(undef, n_qnodes)
    Lmatrices = [LowerTriangular(Matrix{ComplexF64}(undef, 0, 0)) for _ in 1:n_elements]
    Qmatrices = [Matrix{ComplexF64}(undef, 0, 0) for _ in 1:n_elements]

    # compute source points
    bbox, bbox_center, bbox_radius = compute_bounding_box(gquad)
    src_radius = r * bbox_radius
    src_list = get_sphere_sources_lebedev(n_src, src_radius, bbox_center)
    return DimDataDirect(hmax, mesh, gquad, k, α, β, Kϕcoeff, Tϕcoeff, ccoeff, integral_op,
                   Lmatrices, Qmatrices, src_list)
end

"""
    get_surface_densities(dimdata::DimDataDirect, qnode_index::Integer)

Returns the surface densities `ϕ(yⱼ) = ϕ₁τ₁ + ϕ₂τ₂` at qnode `j = qnode_index`,
for operators K and T,
where `τ₁,τ₂` are the tangential vectors and `(ϕ₁, ϕ₂)` are coefficients
at `yⱼ`.
"""
function get_surface_density(dimdata::DimDataDirect, qnode_index::Integer)
    jac = dimdata.gquad.jacobians[qnode_index]
    Kϕ = jac * dimdata.Kϕcoeff[qnode_index]
    Tϕ = jac * dimdata.Tϕcoeff[qnode_index]
    return Kϕ, Tϕ
end

"""
    project_field_onto_surface_density(dimdata::DimDataDirect, field)

Projects the tangential component of a vector field `field`, 
defined on the quadrature nodes, onto the surface density `ϕ`.
The new surface density components are stored in `dimdata.ϕcoeff`.
"""
function project_field_onto_surface_density(dimdata::DimDataDirect, Kfield, Tfield)
    @assert length(Kfield) == length(Tfield) == get_number_of_qnodes(dimdata)
    for i in get_qnode_indices(dimdata.gquad)
        Kvec = Kfield[i]                    # Kvector field at qnode i
        Tvec = Tfield[i]                    # Tvector field at qnode i
        jac = dimdata.gquad.jacobians[i]    # jacobian at qnode i
        dimdata.Kϕcoeff[i] = jac \ Kvec
        dimdata.Tϕcoeff[i] = jac \ Tvec
    end
end

function _assemble_submatrix!(dimdata::DimDataDirect, Mmatrix, qnode, normal, 
                              jacobian, src, k, n_qnodes, r_index, l_index) 
    # Jᵗγ₀G, size=2×3
    M0submatrix = transpose(jacobian) *
                  single_layer_kernel(qnode, src, k, normal)  
    # Jᵗγ₁G, size=2×3
    M1submatrix = transpose(jacobian) *
                  double_layer_kernel(qnode, src, k, normal)  

    # Initial indices (i, j)
    initial_i0 = 2*r_index - 1              # for M0
    initial_i1 = initial_i0 + 2*n_qnodes    # for M1
    initial_j = 3*l_index - 2               # for both M0 and M1

    index_j = initial_j
    for j in 1:3
        index_i0 = initial_i0
        index_i1 = initial_i1
        for i in 1:2
            Mmatrix[index_i0, index_j] = M0submatrix[i, j]
            Mmatrix[index_i1, index_j] = M1submatrix[i, j]
            index_i0 += 1
            index_i1 += 1
        end 
        index_j += 1
    end
end

function compute_density_interpolant(dimdata::DimDataDirect, element_index)
    # Bvector: pre-allocated RHS vector
    # Get data
    _, α, β = getparameters(dimdata)
    qnode_list, _, jac_list = get_nodedata_from_element(dimdata.gquad, 
                                                        element_index)
    qnode_indices = get_inelement_qnode_indices(dimdata.gquad, element_index)                                                        
    n_qnodes = length(qnode_list)  # number of qnodes in element

    # Initialize RHS vector
    # 4 equations per qnode
    Bvector = Vector{ComplexF64}(undef, 4*n_qnodes)

    # Assemble RHS
    for (r_index, qnode_index) in zip(eachindex(qnode_list), qnode_indices)
        jacobian = jac_list[r_index]        # jacobian at qnode
        Kϕcoeff = dimdata.Kϕcoeff[qnode_index]    # K density coefficients at qnode
        Tϕcoeff = dimdata.Tϕcoeff[qnode_index]    # T density coefficients at qnode
        _assemble_rhs!(Bvector, jacobian, Kϕcoeff, Tϕcoeff, r_index, α, β)
    end

    # Solve system using LQ decomposition
    # and save solution
    _solve_dim_lq!(dimdata, Bvector, element_index)
end
function _assemble_rhs!(Bvector, jacobian, Kϕcoeff, Tϕcoeff, r_index, α, β) 
    # RHS_1 = α[τ₁ τ₂]ᵗKϕ, size=2×1,
    # RHS_2 = β[τ₁ τ₂]ᵗTϕ, size=2×1,
    # where J = [τ₁ τ₂] is the jacobian
    rhs_1 = α * transpose(jacobian) * jacobian * Kϕcoeff   
    rhs_2 = β * transpose(jacobian) * jacobian * Tϕcoeff   
    index_1 = 2*r_index - 1
    index_2 = index_1 + length(Bvector)÷2
    for i in 1:2
        Bvector[index_1] = rhs_1[i]
        index_1 += 1
        Bvector[index_2] = rhs_2[i]
        index_2 += 1
    end
end

function _compute_integral_operator_integrand(dimdata::DimDataDirect, element_index_i, 
                                              yi, ni, j)
    k, α, β = getparameters(dimdata)
    yj = dimdata.gquad.nodes[j]                          # qnode j
    nj = dimdata.gquad.normals[j]                        # qnormal at qnode j
    wj = dimdata.gquad.weigths[j]                        # qweigth at qnode j
    Kϕj, Tϕj = get_surface_density(dimdata, j)                # surf. dens. ϕ at qnode j
    γ₀Φj = evaluate_γ₀dim(dimdata, element_index_i, j)   # interpolant γ₀Φ at qnode j
    γ₁Φj = evaluate_γ₁dim(dimdata, element_index_i, j)   # interpolant γ₁Φ at qnode j

    K_input = α*Kϕj - γ₀Φj               # Double layer input vector
    T_input = β*Tϕj - γ₁Φj    # Single layer input vector
    K = double_layer_kernel(yi, yj, k, ni, K_input)   # Double layer operator
    T = single_layer_kernel(yi, yj, k, ni, T_input)   # Single layer operator
    return wj*(K + T)
end

function _compute_potencial_integrand(dimdata::DimDataDirect, j::Integer, x)
    k, α, β = getparameters(dimdata)
    yj = dimdata.gquad.nodes[j]     # qnode j
    nj = dimdata.gquad.normals[j]   # qnormal at qnode j
    wj = dimdata.gquad.weigths[j]   # qweigth at qnode j
    Kϕj, Tϕj = get_surface_density(dimdata, j)                # surf. dens. ϕ at qnode j
    # Double layer potencial
    K_input = α * Kϕj
    Kpot = double_layer_potential_kernel(x, yj, k, K_input)  
    # Single layer potencial
    T_input = β * Tϕj
    Tpot = single_layer_potential_kernel(x, yj, k, T_input) 
    return wj*(Kpot + Tpot)
end