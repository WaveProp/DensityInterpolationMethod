"""
    abstract type AbstractElement{D}

Abstract shape ⊂ ℜ³ given by the image of a parametrization with domain
`D<:AbstractReferenceShape` ⊂ ℜ².

Instances `el` of `AbstractElement` are expected to implement:
- `el(x̂)`: evaluate the parametrization defining the element at the parametric
    coordinates `x̂ ∈ D`.
- `jacobian(el,x̂)` : evaluate the jacobian matrix of the parametrization at the
    parametric coordinate `x ∈ D`. For performance reasons, it is important to
    use an `SMatrix` of size `M × N`, where `M` is the [`ambient_dimension`](@ref)
    of `el` and `N` is the [`geometric_dimension`](@ref) of `el`, respectively.
"""
abstract type AbstractElement{D<:AbstractReferenceShape} end

"""
    (el::AbstractElement)(x)

Evaluate the underlying parametrization of the element `el` at point `x` ∈ ℜ². 
This is the push-forward map for the element. 
"""
function (el::AbstractElement)(x) 
    abstractmethod(typeof(el))
end

"""
    jacobian(el::AbstractElement,x)

Evaluate the jacobian of the underlying parametrization of the element `el` at
point `x`. 
"""
function jacobian(el::AbstractElement, x) 
    abstractmethod(typeof(el))
end

"""
    domain(el::AbstractElement)

Returns an instance of the singleton type `D` ⊂ ℜ²; i.e. the reference element.
"""
domain(::Type{<:AbstractElement{D}}) where {D<:AbstractReferenceShape} = D()
domain(el::AbstractElement) = domain(typeof(el))

"""
    center(el::AbstractElement)

Returns the center of the element, in parametric coordinates.
"""
function center(el::AbstractElement)
    dom = domain(el)
    return center(dom)
end

"""
    measure(τ::AbstractElement, u::AbstractVector)
    measure(jacobian::AbstractMatrix)

The integration measure `μ` of the transformation `τ` so that 
```math
\\int_\\tau f(y) ds_y = \\int_{\\hat{\\tau}} f(\\tau(\\hat{y})) \\mu(\\hat{y}) d\\hat{y}
```
where `` \\hat{\\tau} `` is the reference element. The measure μ is evaluated at 
the parametric coordinate u ∈ `` \\hat{\\tau} ``.

In general, this is given by `√det(g)`, where `g = JᵗJ` is the metric tensor and
`J` the jacobian of the element. 
If the jacobian evaluated at u ∈ `` \\hat{\\tau} `` is not provided, it is computed.
"""
function measure(jacobian::AbstractMatrix)
    g = det(transpose(jacobian)*jacobian)   # general case of a surface measure
    μ = sqrt(g)
    return μ
end
function measure(el::AbstractElement, u::AbstractVector)
    jac = jacobian(el, u)
    return measure(jac)
end

"""
    normal(el::AbstractElement,x̂)

The outer normal vector for the `el` ⊂ ℜ³ at the parametric coordinate `x̂ ∈
domain(el)` ⊂ ℜ². It is computed with the jacobian evaluated at `x̂`. If the
jacobian is not provided, it is computed.
"""
function normal(jacobian::AbstractMatrix)
    t₁ = jacobian[:, 1]
    t₂ = jacobian[:, 2]
    n = cross(t₁, t₂)
    return n / norm(n)   
end    
function normal(el::AbstractElement, u::AbstractVector)
    jac = jacobian(el, u)    
    return normal(jacobian::AbstractMatrix)  
end    

"""
    elementdata(el::AbstractElement, u::AbstractVector)

Returns a tuple '(el_eval, jac, mu, n)', where 'el_eval' is the element parametrization, 
'jac' is the jacobian, 'mu' is the measure and 'n' is the normal, all evaluated at 
parametric coordinates 'u'.
"""
function elementdata(el::AbstractElement, u::AbstractVector)
    el_eval = el(u)
    jac = jacobian(el, u)
    μ = measure(jac)
    n = normal(jac)
    return el_eval, jac, μ, n
end

"""
    struct LagrangeElement{D} <: AbstractElement{D}
    
A lagrange element is represented as a polynomial mapping the `Np` reference
lagrangian nodes of the reference element `D` into `nodes`.

The element's parametrization is fully determined by the image of the `Np`
reference points through polynomial interpolation.
"""
struct LagrangeElement{D} <: AbstractElement{D}
    # Polynomial that represents the mapping from the reference element `D`
    # to the element ⊂ ℜ³.
    # StaticPolynomials.PolynomialSystem is used for fast evaluations.
    forwardmap::PolynomialSystem

    # Constructors
    function LagrangeElement{D}(nodes) where D<:AbstractReferenceShape
        domain = D()
        n_nodes = number_of_nodes(domain)
        @assert n_nodes == length(nodes)
        basis = lagrange_basis(domain)

        # Construct Forward Map using Lagrange basis and nodes
        forwardmap = sum(nodes[i] * basis[i] for i in 1:n_nodes)
        # Convert to StaticPolynomials.PolynomialSystem
        forwardmap = PolynomialSystem(forwardmap)   
        return new{D}(forwardmap)
    end
    function LagrangeElement{D}(nodes::AbstractVector{Point3D}) where D<:AbstractReferenceShape
        # Convert static arrays to regular arrays.
        # For the moment, StaticPolynomials doesn't accept StaticArrays
        # for constructing a PolynomialSystem.
        # (although accepts StaticArrays for evaluating PolynomialSystem, 
        # and the result will be an StaticArray.)
        nodes = [[x, y, z] for (x, y, z) in nodes]
        return LagrangeElement{D}(nodes)
    end
end

# Some aliases
"""
    const FlatTriangleElement = LagrangeElement{ReferenceTriangle3}
"""
const FlatTriangleElement = LagrangeElement{ReferenceTriangle3}

"""
    const QuadraticTriangleElement = LagrangeElement{ReferenceTriangle6}
"""
const QuadraticTriangleElement = LagrangeElement{ReferenceTriangle6}

"""
    order(el::LagrangeElement)

The order of the underlying polynomial used to represent this type of element.
"""
order(::LagrangeElement) = abstractmethod(typeof(el))
function order(el::LagrangeElement{<:ReferenceTriangle})
    dom = domain(el)
    Np = number_of_nodes(dom)
    p = (-3 + sqrt(1+8*Np))/2
    msg = "unable to determine order for LagrangeTriangle containing Np=$(Np) interpolation points.
           Need `Np = (p+1)*(p+2)/2` for some integer `p`."
    @assert isinteger(p) msg
    return Int(p)
 end

function (el::LagrangeElement)(u)
    @assert length(u) == DIMENSION2
    @assert u ∈ domain(el) 
    return StaticPolynomials.evaluate(el.forwardmap, u)
end

function jacobian(el::LagrangeElement, u) 
    @assert length(u) == DIMENSION2    
    @assert u ∈ domain(el)
    return StaticPolynomials.jacobian(el.forwardmap, u)
end 