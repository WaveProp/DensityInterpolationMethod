"""
    Polynomials and Lagrange polynomials definitions
    in `ℜ²`.
    Lagrange polynomials are defined for AbstractReferenceShape and
    its subtypes. These are used to map <:AbstractReferenceShape ⊂ ℜ² into ℜ³.
    The polynomial implementation is based on the DynamicPolynomials module.
    Polynomials are then converted to StaticPolynomials for fast evaluation.
"""

"""
    _generate_forwardmap(basis)

Constructs the Forward Map (StaticPolynomial) associated with 
the Lagrange basis `basis`. This function should be called only 
once for each AbstractReferenceShape subtype.
"""
function _generate_forwardmap(basis)
    n_basis = length(basis)
    n_param = DIMENSION3 * n_basis
    # Symbolic variables that represents the 3D Lagrange nodes.
    # n[1:3] represents node 1, n[4:6] represents node 2, etc.
    @polyvar n[1:n_param]  
    
    # Construct interpolation polynomial (forward map)
    poly = sum(1:n_basis) do i
        single_node = n[-2+3i:3i]  # [n_i, n_{i+1}, n_{i+2}]
        basis[i] * single_node
    end

    # Convert to StaticPolynomial
    static_poly = PolynomialSystem(poly, parameters=n)
    return static_poly
end

# Definition of symbolic variable u = (u1, u2) ∈ `ℜ²`
let (u,) = @polyvar(u[1:2])
    const global _u_symbolic_variable = u
end

# ReferenceTriangle3 Lagrange polynomial basis
let u = _u_symbolic_variable
    b1 = 1 - u[1] - u[2]
    b2 = u[1]
    b3 = u[2]
    basis = (b1, b2, b3)
    const global _ReferenceTriangle3_forwardmap = _generate_forwardmap(basis)
end

# ReferenceTriangle6 Lagrange polynomial basis
let u = _u_symbolic_variable
    b1 = (1 - u[1] - u[2]) * (1 - 2u[1] - 2u[2]) 
    b2 = u[1] * (2u[1] - 1)
    b3 = u[2] * (2u[2] - 1)
    b4 = 4u[1] * (1 - u[1] - u[2])
    b5 = 4 * u[1] * u[2] 
    b6 = 4u[2] * (1 - u[1] - u[2])
    basis = (b1, b2, b3, b4, b5, b6) 
    const global _ReferenceTriangle6_forwardmap = _generate_forwardmap(basis) 
end
