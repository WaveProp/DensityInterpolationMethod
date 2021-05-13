using Test
using DensityInterpolationMethod
using DensityInterpolationMethod.Mesh
using DensityInterpolationMethod.Integration

import DensityInterpolationMethod.Mesh: ReferenceTriangle3
import DensityInterpolationMethod.Integration: get_qrule_for_reference_shape
include("../test_utils.jl")

# Tolerances
const TOL1 = 1e-10

@testset "Quadrature rules in Reference Triangle" begin
    # Monomials
    function getmonomial(n1, n2)
        f(x) = x[1]^n1 * x[2]^n2
        return f
    end

    # Exact integral
    iexact(n1, n2) = factorial(n1) * factorial(n2) / factorial(n1 + n2 + 2)

    function test_qrule(order, ref_shape, tolerance)
        qrule = get_qrule_for_reference_shape(ref_shape, order)
        for n1 in 0:order
            for n2 in 0:order-n1
                mon = getmonomial(n1, n2)
                integral = integrate(mon, qrule)
                @test relative_error(integral, iexact(n1, n2)) < tolerance
            end
        end
    end

    refshape = ReferenceTriangle3()
    for i in 1:QRULE_MAX_ORDER
        @testset "Qrule order $i" begin
            test_qrule(i, refshape, TOL1)
        end
    end
end

