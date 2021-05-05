"""
    Integration methods for Global Quadrature.
"""

"""
    integrate(gquad::GlobalQuadrature, f)

Integrates a function `f` using the quadrature data of GlobalQuadrature.
"""
function integrate(gquad::GlobalQuadrature, f)
    return sum(get_qnodes(gquad)) do qnode
        x, w, _, _ = get_qnode_data(qnode)
        f(x) * w
    end
end

"""
    integrateflux(gquad::GlobalQuadrature, f)

Integrates the flux of a function `f` using the quadrature data of GlobalQuadrature.
"""
function integrateflux(gquad::GlobalQuadrature, f)
    return sum(get_qnodes(gquad)) do qnode
        x, w, _, n = get_qnode_data(qnode)
        dot(n, f(x)) * w
    end
end

