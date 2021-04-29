"""
Definition of Green tensor for Maxwell's equations.
"""

"""
    helmholtz_green_function(r, k)

Returns the Green function for Helmholtz equation in 3D,
i.e. `g = exp(im*k*r)/(4π*r)`.
"""
function helmholtz_green_function(r, k)
    g = exp(im*k*r)/(4π*r)
    return g
end

"""
    cross_product_matrix(v) 

Returns the matrix `Aᵥ` associated with the 
cross product `v × ϕ` so that `v × ϕ = Aᵥϕ`.
"""
function cross_product_matrix(v)
    return transpose(SMatrix{3,3,Float64,9}(      0, -v[3],  v[2],
                                             v[3],       0, -v[1],
                                            -v[2],    v[1],     0))
end

# Single Layer Kernel
# n × G = γ₀ G
function single_layer_kernel(x, y, k, nx)  
    x==y && return zero(ComplexPoint3D)
    rvec = x - y
    r = norm(rvec)
    g = helmholtz_green_function(r, k)
    gp  = im*k*g - g/r
    gpp = im*k*gp - gp/r + g/r^2
    RRT = rvec*transpose(rvec) # rvec ⊗ rvecᵗ
    G = g*I + 1/k^2*(gp/r*I + (gpp/r^2 - gp/r^3)*RRT)
    ncross = cross_product_matrix(nx)
    return  ncross * G
end

# Double Layer Kernel
# n × ∇ × G = γ₁ G
function double_layer_kernel(x, y, k, nx) 
    x==y && return zero(ComplexPoint3D)
    rvec = x - y
    r = norm(rvec)
    g   = helmholtz_green_function(r, k)
    gp  = im*k*g - g/r
    ncross = cross_product_matrix(nx)
    rcross = cross_product_matrix(rvec)
    return gp/r*ncross*rcross
end

# Single Layer Kernel
# n × G = γ₀ G
function single_layer_kernel_eval(x, y, k, nx, ϕy)  
    x==y && return zero(ComplexPoint3D)
    rvec = x - y
    r = norm(rvec)
    rhat = rvec/r
    kr = k*r
    kr2 = kr^2
    g = helmholtz_green_function(r, k)

    result = (1 + im/kr - 1/kr2)*ϕy
    result += (-1 - 3im/kr + 3/kr2)*dot(rhat, ϕy)*rhat
    result *= g
    return cross(nx, result)
end

# Double Layer Kernel
# n × ∇ × G = γ₁ G
function double_layer_kernel_eval(x, y, k, nx, ϕy) 
    x==y && return zero(ComplexPoint3D)
    rvec = x - y
    r = norm(rvec)
    g = helmholtz_green_function(r, k)
    gp  = (im*k - 1/r)*g
    result = gp/r*cross(nx, cross(rvec, ϕy))
    return result
end

"""
    electric_dipole_electric_field(x, y, k, ϕy)  

Returns the electric field at `x` produced by an electric dipole
located at `y` with direction `ϕy` and wavenumber `k`. 
The electric field is given by `Gₖ(x, y) * ϕy`, where `Gₖ` is the
Green tensor.
"""
function electric_dipole_electric_field(x, y, k, ϕy)
    x==y && return zero(ComplexPoint3D)
    rvec = x - y
    r = norm(rvec)
    rhat = rvec/r
    kr = k*r
    kr2 = kr^2
    g = helmholtz_green_function(r, k)
    result = (1 + im/kr - 1/kr2)*ϕy
    result += (-1 - 3im/kr + 3/kr2)*dot(rhat, ϕy)*rhat
    result *= g
    return result
end

"""
    electric_dipole_magnetic_field(x, y, k, ϕy)  

Returns the magnetic field at `x` produced by an electric dipole
located at `y` with direction `ϕy` and wavenumber `k`. 
The magnetic field is given by `1/(im*k) * ∇ × Gₖ(x, y) * ϕy`, where `Gₖ` is the
Green tensor.
"""
function electric_dipole_magnetic_field(x, y, k, ϕy)
    x==y && return zero(ComplexPoint3D)
    rvec = x - y
    r = norm(rvec)
    g = helmholtz_green_function(r, k)
    gp  = (im*k - 1/r)*g
    result = 1/(im*k)*gp/r*cross(rvec, ϕy)
    return result
end

