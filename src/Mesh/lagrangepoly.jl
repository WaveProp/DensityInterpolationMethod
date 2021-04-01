"""
    Polynomials and Lagrange polynomials definitions
    in `ℜ²`.
    Lagrange polynomials are defined for AbstractReferenceShape and
    its subtypes. These are used to map <:AbstractReferenceShape ⊂ ℜ² into ℜ³.
    The polynomial implementation is based on the DynamicPolynomials module.
    Polynomials are later converted to StaticPolynomials for fast evaluation.
"""

# Definition of symbolic variable u = (u1, u2) ∈ `ℜ²`
let (u,) = @polyvar(u[1:2])
    const global _u_symbolic_variable = u
end

# ReferenceTriangle3 Lagrange polynomial basis
let u = _u_symbolic_variable
    b1 = 1 - u[1] - u[2]
    b2 = u[1]
    b3 = u[2]
    const global _ReferenceTriangle3_basis = (b1, b2, b3)
end

# ReferenceTriangle6 Lagrange polynomial basis
let u = _u_symbolic_variable
    b1 = (1 - u[1] - u[2]) * (1 - 2u[1] - 2u[2]) 
    b2 = u[1] * (2u[1] - 1)
    b3 = u[2] * (2u[2] - 1)
    b4 = 4u[1] * (1 - u[1] - u[2])
    b5 = 4 * u[1] * u[2] 
    b6 = 4u[2] * (1 - u[1] - u[2])
    const global _ReferenceTriangle6_basis = (b1, b2, b3, b4, b5, b6) 
end
