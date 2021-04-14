"""
    abstract type AbstractElement{D}

Abstract shape ⊂ ℜ³ given by the image of a parametrization with domain
`D<:AbstractReferenceShape` ⊂ ℜ².

Instances `el` of `AbstractElement` are expected to implement:
- `el(x̂)`: evaluate the parametrization defining the element at the parametric
    coordinates `x̂ ∈ D`.
- `getjacobian(el,x̂)` : evaluate the jacobian matrix of the parametrization at the
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
    getjacobian(el::AbstractElement,x)

Evaluate the jacobian of the underlying parametrization of the element `el` at
point `x`. 
"""
function getjacobian(el::AbstractElement, x) 
    abstractmethod(typeof(el))
end

"""
    evaluate_and_getjacobian(el::AbstractElement, u)

Returns the element parametrization and jacobian evaluated at `u`.
"""
function evaluate_and_getjacobian(el::AbstractElement, u)
    abstractmethod(typeof(el))
end

"""
    getdomain(el::AbstractElement)

Returns an instance of the singleton type `D` ⊂ ℜ²; i.e. the reference element.
"""
getdomain(::Type{<:AbstractElement{D}}) where {D<:AbstractReferenceShape} = D()
getdomain(el::AbstractElement) = getdomain(typeof(el))

"""
    getmeasure(τ::AbstractElement, u::AbstractVector)
    getmeasure(jacobian::AbstractMatrix)

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
function getmeasure(jacobian::AbstractMatrix)
    g = det(transpose(jacobian)*jacobian)   # general case of a surface measure
    μ = sqrt(g)
    return μ
end
function getmeasure(el::AbstractElement, u::AbstractVector)
    jac = getjacobian(el, u)
    return getmeasure(jac)
end

"""
    getnormal(el::AbstractElement,x̂)

The outer normal vector for the `el` ⊂ ℜ³ at the parametric coordinate `x̂ ∈
domain(el)` ⊂ ℜ². It is computed with the jacobian evaluated at `x̂`. If the
jacobian is not provided, it is computed.
"""
function getnormal(jacobian::AbstractMatrix)
    t₁ = jacobian[:, 1]
    t₂ = jacobian[:, 2]
    n = cross(t₁, t₂)
    return n / norm(n)   
end    
function getnormal(el::AbstractElement, u::AbstractVector)
    jac = getjacobian(el, u)    
    return getnormal(jac)  
end    

"""
    getelementdata(el::AbstractElement, u::AbstractVector)

Returns a tuple `(el_eval, jac, μ, n)`, where `el_eval` is the element parametrization, 
`jac` is the jacobian, `μ` is the measure and `n` is the normal, all evaluated at 
parametric coordinates `u`.
"""
function getelementdata(el::AbstractElement, u::AbstractVector)
    el_eval, jac = evaluate_and_getjacobian(el, u)
    μ = getmeasure(jac)
    n = getnormal(jac)
    return el_eval, jac, μ, n
end

"""
    struct LagrangeElement{D, N} <: AbstractElement{D}
    
A lagrange element is represented as a polynomial mapping the `N` reference
lagrangian nodes of the reference element `D` into `nodes`. The `M` parameter
is always equal to `DIMENSION3 * N`
The element's parametrization is fully determined by the image of the `N`
reference points through the Forward Map (StaticPolynomial) defined in `D`.
"""
struct LagrangeElement{D, N, M} <: AbstractElement{D}
    # 3D Lagrange nodes of the element.
    # Each column represents a node.
    # The nodes are saved in an SMatrix so it can be passed
    # efficiently to the Forward Map (StaticPolynomial)
    nodes::SMatrix{DIMENSION3, N, Float64, M}

    # Constructor
    function LagrangeElement{D, N, M}(nodes_list) where {D<:AbstractReferenceShape,N,M}
        @assert M == N * DIMENSION3
        @assert N == length(nodes_list)
        # Arrange nodes into SMatrix
        nodes = SMatrix{DIMENSION3, N}(vcat(nodes_list...))
        return new{D, N, M}(nodes)
    end
end

"""
    get_lagrange_elemtype(D::Type{<:AbstractReferenceShape})

Returns the corresponding LagrangeElement type associated with
`D::Type{<:AbstractReferenceShape}`.
"""
function get_lagrange_elemtype(D::Type{<:AbstractReferenceShape}) 
    N = get_number_of_lnodes(D())
    M = DIMENSION3 * N
    return LagrangeElement{D, N, M}
end
                                                          
"""
    getcenter(el::LagrangeElement)

Returns the center of the element, in parametric coordinates.
"""
function getcenter(el::LagrangeElement)
    dom = getdomain(el)
    return getcenter(dom)
end

"""
    get_lnodes(el::LagrangeElement)

Returns the lagrangian nodes of the element, in parametric coordinates.
"""
function get_lnodes(el::LagrangeElement)
    dom = getdomain(el)
    return get_lnodes(dom)
end

"""
    get_number_of_lnodes(el::LagrangeElement)

Returns the number of lagrangian nodes of the element.
"""
function get_number_of_lnodes(el::LagrangeElement{D, N}) where {D,N}
    return N
end

# Some aliases
"""
    const FlatTriangleElement = LagrangeElement{ReferenceTriangle3}
"""
const FlatTriangleElement = get_lagrange_elemtype(ReferenceTriangle3)

"""
    const QuadraticTriangleElement = LagrangeElement{ReferenceTriangle6}
"""
const QuadraticTriangleElement = get_lagrange_elemtype(ReferenceTriangle6)

"""
    getorder(el::LagrangeElement)

The order of the underlying polynomial used to represent this type of element.
"""
getorder(::LagrangeElement) = abstractmethod(typeof(el))
function getorder(el::LagrangeElement{<:ReferenceTriangle})
    Np = get_number_of_lnodes(el)
    p = (-3 + sqrt(1+8*Np))/2
    msg = "unable to determine order for LagrangeTriangle containing Np=$(Np) interpolation points.
           Need `Np = (p+1)*(p+2)/2` for some integer `p`."
    @assert isinteger(p) msg
    return Int(p)
 end

function (el::LagrangeElement)(u)
    @assert length(u) == DIMENSION2
    domain = getdomain(el) 
    @assert u ∈ domain
    forwardmap = get_forwardmap(domain)
    return StaticPolynomials.evaluate(forwardmap, u, el.nodes)
end

function getjacobian(el::LagrangeElement, u) 
    @assert length(u) == DIMENSION2
    domain = getdomain(el) 
    @assert u ∈ domain
    forwardmap = get_forwardmap(domain)
    return StaticPolynomials.jacobian(forwardmap, u, el.nodes)
end 

function evaluate_and_getjacobian(el::LagrangeElement, u) 
    @assert length(u) == DIMENSION2
    domain = getdomain(el) 
    @assert u ∈ domain
    forwardmap = get_forwardmap(domain)
    return StaticPolynomials.evaluate_and_jacobian(forwardmap, u, el.nodes)
end