"""
    Integration methods for meshes and elements.
"""

"""
    get_qnodes_and_qweights(q::AbstractQuadratureRule, el)

Returns the quadrature nodes `x` and weights `w` for integrating over `el`. 
The *lifted* quadrature is computed by mapping the reference quadrature through
`el`. This requires `el` to support the methods `el(x̂)` and `getjacobian(el,x̂)`.
"""
function get_qnodes_and_qweights(q::AbstractQuadratureRule, el)
    @assert getdomain(el) === getdomain(q) 
    x̂, ŵ = get_qnodes_and_qweights(q)
    return _push_forward_quadrature(el, x̂, ŵ)
end
function _push_forward_quadrature(el, x̂, ŵ)
    x = map(x̂ᵢ->el(x̂ᵢ),x̂)
    w = map(x̂,ŵ) do x̂ᵢ,ŵᵢ
        μ = getmeasure(el, x̂ᵢ)
        μ * ŵᵢ
    end 
    return x, w
end    

"""
    get_qnodes_qweights_and_qnormals(q::AbstractQuadratureRule, el)

Returns the quadrature nodes `x`, weights `w` and unit normals `n` for integrating
over `el`. The *lifted* quadrature is computed by mapping the reference quadrature
through `el`. This requires `el` to support the methods `el(x̂)` and `getjacobian(el,x̂)`.
TODO: CHECK PERFORMANCE
"""
function get_qnodes_qweights_and_qnormals(q::AbstractQuadratureRule, el)
    @assert getdomain(el) === getdomain(q) 
    x̂, ŵ = get_qnodes_and_qweights(q)
    n = (getnormal(el, xᵢ) for xᵢ in x̂)
    x, w = _push_forward_quadrature(el, x̂, ŵ)
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
TODO: CHECK PERFORMANCE
"""
function integrateflux(f, q::AbstractQuadratureRule, el)
    x, w, n = get_qnodes_qweights_and_qnormals(q, el)
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
TODO: CHECK PERFORMANCE
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