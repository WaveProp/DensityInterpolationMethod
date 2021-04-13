"""
    Integration methods for Global Quadrature.
"""

"""
    integrate(gquad::GlobalQuadrature, f)

Integrates a function `f` using the quadrature data of GlobalQuadrature.
"""
function integrate(gquad::GlobalQuadrature, f)
    return sum(gquad.nodes, gquad.weigths) do xᵢ,wᵢ
        f(xᵢ) * wᵢ
    end
end

"""
    integrateflux(gquad::GlobalQuadrature, f)

Integrates the flux of a function `f` using the quadrature data of GlobalQuadrature.
"""
function integrateflux(gquad::GlobalQuadrature, f)
    return sum(gquad.nodes, gquad.weigths, gquad.normals) do xᵢ,wᵢ,nᵢ
        dot(f(xᵢ), nᵢ) * wᵢ
    end
end

