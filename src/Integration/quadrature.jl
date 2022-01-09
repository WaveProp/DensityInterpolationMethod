"""
    Definitions of quadratures rules for various AbstractReferenceShape.
"""

"""
    abstract type AbstractQuadratureRule{D<:AbstractReferenceShape}

A quadrature rule for integrating a function over the domain `D`.

An instance `q` of `AbstractQuadratureRule{D}` is expected to implement the
following methods:

- `get_qnodes_and_qweights(q)` : return the nodes `x` and weights `w` of the quadrature rule on the
  reference domain `D`. For performance reasons, the result shoudl depend only
  on the type of `q`.
- `get_qnodes_and_qweights(q,el)` : return the nodes `x` and weights `w` of the quadrature rule on the
  element `D`. This assumes that `domain(q)==domain(el)`, so that the element
  quadrature can be computed by *pushing forward* a reference quadrature to `el`.
"""
abstract type AbstractQuadratureRule{D<:AbstractReferenceShape} end

getdomain(q::AbstractQuadratureRule{D}) where {D} = D()

"""
    get_qnodes_and_qweights(q::AbstractQuadratureRule)

Return the quadrature nodes `x` and weights `w` on the `domain(q)`.
"""
function get_qnodes_and_qweights(q::AbstractQuadratureRule)
    abstractmethod(typeof(q))
end

"""
    getqnodes(Y)

Return the quadrature nodes associated with `Y`.
"""
getqnodes(q::AbstractQuadratureRule) = get_qnodes_and_qweights(q)[1]

"""
    getqweights(Y)

Return the quadrature weights associated with `Y`.
"""
getqweights(q::AbstractQuadratureRule) = get_qnodes_and_qweights(q)[2]


"""
    struct GaussQuadrature{D, O} <: AbstractQuadratureRule{D}

Tabulated symmetric Gauss quadrature rule of order `O` for integration over `D`.
This is currently implemented on triangles by calling the Gmsh API.
"""
struct GaussQuadrature{D, O} <: AbstractQuadratureRule{D} 
    GaussQuadrature(ref::Type{AbstractReferenceShape}, order) = new{ref, order}()
    GaussQuadrature(ref, order) = new{typeof(ref), order}()
end

"""
    get_number_of_qnodes(q::GaussQuadrature)

Returns the total number of quadrature nodes (for a single element) 
associated with the quadrature rule `q`.
"""
function get_number_of_qnodes(q::GaussQuadrature)
    _, w = get_qnodes_and_qweights(q)
    return length(w)
end

"""
    get_qrule_for_reference_shape(ref, order)

Given a `ref`erence shape and a desired quadrature `order`, return
an appropiate quadrature rule.
"""
function get_qrule_for_reference_shape(ref, order)
    if ref isa ReferenceTriangle
        # This orders contains points outside the
        # reference triangle, so they can't be used
        forbidden_orders = SVector(11, 15, 16, 18, 20)
        if order ∉ forbidden_orders
            return GaussQuadrature(ref, order)
        end
    end
    error("no appropriate quadrature rule found.")
end

"""
    get_qrule_for_element(E, order)

Given an element type `E`, return an appropriate quadrature of order `order`.
"""
function get_qrule_for_element(E, order)
    return get_qrule_for_reference_shape(getdomain(E), order)
end

@generated function 
    get_qnodes_and_qweights(q::GaussQuadrature{<:ReferenceTriangle, O}) where {O}
    if O == 5
        x,w = gauss5_qnodes_and_qweights()
    elseif O == 7
        x,w = gauss7_qnodes_and_qweights()
    else
        element_name = "Triangle"
        qrule_name = "Gauss$O"
        x, w = get_qrule_from_gmsh(element_name, qrule_name)
        @assert (length(x) == length(w)) && (length(x) > 0)
    end
    return :($x,$w)
end

function gauss5_qnodes_and_qweights()
    #*! Quadrature rule for an interpolation of order 5 on the triangle *#
    #* 'Symmetric quadrature rules for simplexes based on sphere close packed
    #*  lattice arrangements', D.M. Williams, L. Shunn and A. Jameson *#    
    TRIANGLE_G5N10 = (
    (SVector(3.333333333333333e-01, 3.333333333333333e-01), 1.007714942923651e-01),
    (SVector(5.556405266979300e-02, 8.888718946604139e-01), 2.097775649832452e-02),
    (SVector(8.888718946604139e-01, 5.556405266979300e-02), 2.097775649832452e-02),
    (SVector(5.556405266979300e-02, 5.556405266979300e-02), 2.097775649832452e-02),
    (SVector(6.342107477457230e-01, 7.025554051838412e-02), 5.604920603544356e-02),
    (SVector(7.025554051838412e-02, 6.342107477457230e-01), 5.604920603544356e-02),
    (SVector(2.955337117358930e-01, 7.025554051838412e-02), 5.604920603544356e-02),
    (SVector(7.025554051838412e-02, 2.955337117358930e-01), 5.604920603544356e-02),
    (SVector(2.955337117358930e-01, 6.342107477457230e-01), 5.604920603544356e-02),
    (SVector(6.342107477457230e-01, 2.955337117358930e-01), 5.604920603544356e-02))
    x = [qdata[1] for qdata in TRIANGLE_G5N10]
    w = [qdata[2] for qdata in TRIANGLE_G5N10]
    return x,w
end

function gauss7_qnodes_and_qweights()
    #*! Quadrature rule for an interpolation of order 7 on the triangle *#
    #* 'Symmetric quadrature rules for simplexes based on sphere close packed
    #*  lattice arrangements', D.M. Williams, L. Shunn and A. Jameson *#
    TRIANGLE_G7N15 = (
    (SVector(3.587087769573400e-02, 9.282582446085320e-01), 8.957727506151512e-03),
    (SVector(9.282582446085320e-01, 3.587087769573400e-02), 8.957727506151512e-03),
    (SVector(3.587087769573400e-02, 3.587087769573400e-02), 8.957727506151512e-03),
    (SVector(2.417293957679670e-01, 5.165412084640659e-01), 6.385609794063259e-02),
    (SVector(5.165412084640659e-01, 2.417293957679670e-01), 6.385609794063259e-02),
    (SVector(2.417293957679670e-01, 2.417293957679670e-01), 6.385609794063259e-02),
    (SVector(4.743087877770790e-01, 5.138242444584196e-02), 3.810303119276755e-02),
    (SVector(5.138242444584196e-02, 4.743087877770790e-01), 3.810303119276755e-02),
    (SVector(4.743087877770790e-01, 4.743087877770790e-01), 3.810303119276755e-02),
    (SVector(7.511836311064840e-01, 4.731248701171598e-02), 2.787490501355754e-02),
    (SVector(4.731248701171598e-02, 7.511836311064840e-01), 2.787490501355754e-02),
    (SVector(2.015038818818000e-01, 4.731248701171598e-02), 2.787490501355754e-02),
    (SVector(4.731248701171598e-02, 2.015038818818000e-01), 2.787490501355754e-02),
    (SVector(2.015038818818000e-01, 7.511836311064840e-01), 2.787490501355754e-02),
    (SVector(7.511836311064840e-01, 2.015038818818000e-01), 2.787490501355754e-02))
    x = [qdata[1] for qdata in TRIANGLE_G7N15]
    w = [qdata[2] for qdata in TRIANGLE_G7N15]
    return x,w
end
