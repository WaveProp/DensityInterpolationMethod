"""
Definition of Green tensor and utility functions
for Maxwell's equations.
"""

"""
    const MaxwellKernelType = SMatrix{DIMENSION3, DIMENSION3, 
                                      ComplexF64, DIMENSION3*DIMENSION3}

Default kernel type used in Maxwell's integral operators.
"""
const MaxwellKernelType = SMatrix{DIMENSION3, DIMENSION3, 
                                  ComplexF64, DIMENSION3*DIMENSION3}

"""
    const ReducedMaxwellKernelType = SMatrix{DIMENSION3, DIMENSION2, 
                                             ComplexF64, DIMENSION3*DIMENSION2}

Kernel type used in Maxwell's integral operators. This kernel type is typically
obtained when `MaxwellKernelType` (size = 3x3) is multiplied by a jacobian
matrix (size = 3x2) from the right.
"""
const ReducedMaxwellKernelType = SMatrix{DIMENSION3, DIMENSION2, 
                                         ComplexF64, DIMENSION3*DIMENSION2}

"""
    const ReducedReducedMaxwellKernelType = SMatrix{DIMENSION2, DIMENSION2, 
                                                    ComplexF64, DIMENSION2*DIMENSION2}

Kernel type used in Maxwell's integral operators. This kernel type is typically
obtained when `MaxwellKernelType` (size = 3x3) is multiplied by a jacobian
matrix (size = 3x2) from the right and a transposed jacobian matrix (size = 2x3)
from the left.
"""
const ReducedReducedMaxwellKernelType = SMatrix{DIMENSION2, DIMENSION2, 
                                                ComplexF64, DIMENSION2*DIMENSION2}

"""
    _helmholtz_green_function(r, k)

Returns the Green function for Helmholtz equation in 3D,
i.e. `g = exp(im*k*r)/(4π*r)`.
"""
function _helmholtz_green_function(r, k)
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

"""
    _green_tensor(x, y, k)
    _green_tensor(x, y, k, ϕy)

Returns `G(x, y)` or `G(x, y)*ϕy`, where `G` is the 
Green tensor for Maxwell's equations with wavenumber `k`.
"""
function _green_tensor(x, y, k)
    rvec = x - y
    r = norm(rvec)
    g = _helmholtz_green_function(r, k)
    gp  = im*k*g - g/r
    gpp = im*k*gp - gp/r + g/r^2
    RRT = rvec*transpose(rvec) # rvec ⊗ rvecᵗ
    G = g*I + 1/k^2*(gp/r*I + (gpp/r^2 - gp/r^3)*RRT)
    return G
end
function _green_tensor(x, y, k, ϕy)
    rvec = x - y
    r = norm(rvec)
    rhat = rvec/r
    kr = k*r
    kr2 = kr^2
    g = _helmholtz_green_function(r, k)
    Gϕy = (1 + im/kr - 1/kr2)*ϕy
    Gϕy += (-1 - 3im/kr + 3/kr2)*dot(rhat, ϕy)*rhat
    Gϕy *= g
    return Gϕy
end

"""
    _curl_green_tensor(x, y, k)
    _curl_green_tensor(x, y, k, ϕy)

Returns `∇ₓ × G(x, y)` or `∇ₓ × (G(x, y) * ϕy)`, where `G` 
is the Green tensor for Maxwell's equations with wavenumber `k`.
"""
function _curl_green_tensor(x, y, k)
    rvec = x - y
    r = norm(rvec)
    g   = _helmholtz_green_function(r, k)
    gp  = im*k*g - g/r
    rcross = cross_product_matrix(rvec)
    curl_G = gp/r*rcross
    return curl_G
end
function _curl_green_tensor(x, y, k, ϕy)
    rvec = x - y
    r = norm(rvec)
    g = _helmholtz_green_function(r, k)
    gp = (im*k - 1/r)*g
    curl_Gϕy = gp/r*cross(rvec, ϕy)
    return curl_Gϕy
end

"""
    single_layer_kernel(x, y, k, nx) 
    single_layer_kernel(x, y, k, nx, ϕy)  

Returns the single layer integral operator 
kernel `γ₀G = nₓ × G` or `γ₀(G*ϕy) = nₓ × G*ϕy`.
"""
function single_layer_kernel(x, y, k, nx)  
    G = _green_tensor(x, y, k)
    ncross = cross_product_matrix(nx)
    SL_kernel = ncross * G
    return SL_kernel
end
function single_layer_kernel(x, y, k, nx, ϕy)  
    Gϕy = _green_tensor(x, y, k, ϕy)
    SL_kernel = cross(nx, Gϕy)
    return SL_kernel
end

"""
    double_layer_kernel(x, y, k, nx) 
    double_layer_kernel(x, y, k, nx, ϕy)  

Returns the double layer integral operator 
kernel `γ₁G = nₓ × ∇ₓ × G` or `γ₁(G*ϕy) = nₓ × ∇ₓ × (G*ϕy)`.
"""
function double_layer_kernel(x, y, k, nx) 
    curl_G = _curl_green_tensor(x, y, k)
    ncross = cross_product_matrix(nx)
    DL_kernel = ncross * curl_G
    return DL_kernel
end
function double_layer_kernel(x, y, k, nx, ϕy) 
    curl_Gϕy = _curl_green_tensor(x, y, k, ϕy)
    DL_kernel = cross(nx, curl_Gϕy)
    return DL_kernel
end

"""
    single_layer_potential_kernel(x, y, k) 
    single_layer_potential_kernel(x, y, k, ϕy)  

Returns the single layer potential 
kernel `G` or `G*ϕy`.
"""
function single_layer_potential_kernel(x, y, k)  
    return _green_tensor(x, y, k)
end
function single_layer_potential_kernel(x, y, k, ϕy)  
    return _green_tensor(x, y, k, ϕy)
end

"""
    double_layer_potential_kernel(x, y, k) 
    double_layer_potential_kernel(x, y, k, ϕy)  

Returns the double layer potential 
kernel `∇ₓ × G` or `∇ₓ × (G*ϕy)`.
"""
function double_layer_potential_kernel(x, y, k)
    return _curl_green_tensor(x, y, k)
end
function double_layer_potential_kernel(x, y, k, ϕy) 
    return _curl_green_tensor(x, y, k, ϕy)
end

"""
    electric_dipole_electric_field(x, y, k, ϕy)  

Returns the electric field at `x` produced by an electric dipole
located at `y` with polarizaton `ϕy` and wavenumber `k`. 
The electric field is given by `Gₖ(x, y) * ϕy`, where `Gₖ` is the
Green tensor.
"""
function electric_dipole_electric_field(x, y, k, ϕy)
    return _green_tensor(x, y, k, ϕy)
end

"""
    electric_dipole_magnetic_field(x, y, k, ϕy)  

Returns the magnetic field at `x` produced by an electric dipole
located at `y` with polarizaton `ϕy` and wavenumber `k`. 
The magnetic field is given by `1/(im*k) * ∇ × Gₖ(x, y) * ϕy`, where `Gₖ` is the
Green tensor.
"""
function electric_dipole_magnetic_field(x, y, k, ϕy)
    1/(im*k)*_curl_green_tensor(x, y, k, ϕy)
    return result
end

