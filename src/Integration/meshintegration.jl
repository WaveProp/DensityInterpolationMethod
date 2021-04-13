"""
    Integration methods for meshes and elements.
"""

"""
    get_quadrature_data(q::AbstractQuadratureRule, el)

Returns the n-point quadrature data `(d₁, d₂, ..., dₙ)` for the element `el`,
where `dᵢ = (xᵢ, wᵢ, jacᵢ, nᵢ)`, `xᵢ` are the *lifted* quadrature nodes, 
`wᵢ` are the *lifted* quadrature weights, `jacᵢ` is the jacobian at `xᵢ` 
and `nᵢ` is the normal at xi, for `i=1:N`. This requires `el` to support 
the method `getelementdata(el, x̂)`.
"""
function get_quadrature_data(q::AbstractQuadratureRule, el)
    @assert getdomain(el) === getdomain(q) 
    x̂, ŵ = get_qnodes_and_qweights(q)
    qdata = _push_forward_quadrature(el, x̂, ŵ)
    return qdata
end
function _push_forward_quadrature(el, x̂, ŵ)
    qdata = map(x̂, ŵ) do x̂ᵢ,ŵᵢ
        x, jac, μ, n = getelementdata(el, x̂ᵢ)
        w = μ * ŵᵢ
        x, w, jac, n
    end
    return qdata
end    

"""
    get_qnodes_and_qweights(q::AbstractQuadratureRule, el)

Returns the quadrature nodes `x` and *lifted* weights `w` for integrating over `el`.
"""
function get_qnodes_and_qweights(q::AbstractQuadratureRule, el)
    qdata = get_quadrature_data(q, el)
    x, w, _, _ = zip(qdata...)
    return x, w
end

"""
    get_qnodes_qweights_qnormals(q::AbstractQuadratureRule, el)

Returns the quadrature nodes `x`, *lifted* weights `w` and normal vectors `n` 
for integrating over `el`.
"""
function get_qnodes_qweights_qnormals(q::AbstractQuadratureRule, el)
    qdata = get_quadrature_data(q, el)
    x, w, _, n = zip(qdata...)
    return x, w, n
end

"""
    integrate(f, q::AbstractQuadratureRule, el)
    integrate(f, q::AbstractQuadrature)
    integrate(f, x, w)

Integrates the function `f` using the quadrature rule `q` on the element `el`. 
This is simply `sum(f.(x) .* w)`, where `x` and `w` are the quadrature nodes 
and weights, respectively.
"""
function integrate(f, q::AbstractQuadratureRule, el)
    x, w = get_qnodes_and_qweights(q, el)
    return integrate(f, x, w)
end 
function integrate(f, q::AbstractQuadratureRule)
    x̂, ŵ = get_qnodes_and_qweights(q)
    return integrate(f, x̂, ŵ)
end
function integrate(f,x,w)
    return sum(zip(x, w)) do (xᵢ, wᵢ)
        f(xᵢ) * wᵢ
    end
end
function integrate(f,x,w,n)
    return sum(zip(x, w, n)) do (xᵢ, wᵢ, nᵢ)
        dot(f(xᵢ), nᵢ) * wᵢ
    end
end

"""
    integrateflux(f, q::AbstractQuadratureRule, el)

Integrates the flux of the function `f` (ℜ³ ⟶ ℜ³) using the quadrature rule `q` 
on the element `el`. This is simply `sum(dot(f(xᵢ), n(xᵢ)) .* wᵢ)`, where `x` and 
`w` are the quadrature nodes and weights, respectively, and `n` is the unit normal.
"""
function integrateflux(f, q::AbstractQuadratureRule, el)
    x, w, n = get_qnodes_qweights_qnormals(q, el)
    return integrate(f, x, w, n)
end 

"""
    integrate(mesh::GenericMesh, f; order=2)

Integrates a function `f` on all elements of the mesh `mesh`
using a Gaussian quadrature of order `order`.
"""
function integrate(mesh::GenericMesh, f; order=2)
    return sum(get_etypes_and_elements(mesh)) do (etype, elements)
        qrule = get_qrule_for_element(etype, order)
        _integrate_element_list(f, qrule, elements)
    end
end
function _integrate_element_list(f, q::AbstractQuadratureRule, elements)
    return sum(integrate(f, q, el) for el in elements)
end

"""
    integrateflux(mesh::GenericMesh, f; order=2)

Integrates the flux of the function `f` on all elements of the mesh `mesh`
using a Gaussian quadrature of order `order`.
"""
function integrateflux(mesh::GenericMesh, f; order=2)
    return sum(get_etypes_and_elements(mesh)) do (etype, elements)
        qrule = get_qrule_for_element(etype, order)
        _integrateflux_element_list(f, qrule, elements)
    end
end
function _integrateflux_element_list(f, q::AbstractQuadratureRule, elements)
    return sum(integrateflux(f, q, el) for el in elements)
end

"""
    get_number_of_qnodes(mesh::GenericMesh)

Returns the total number of quadrature nodes of a GenericMesh, 
for a quadrature rule of order `order`.
"""
function get_number_of_qnodes(mesh::GenericMesh, order)
    return sum(get_etypes_and_elements(mesh)) do (etype,elements)
        qrule = get_qrule_for_element(etype, order)
        length(elements) * get_number_of_qnodes(qrule)
    end
end