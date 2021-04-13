using Test
using SafeTestsets

@safetestset "Quadrature tests" begin include("quadraturetest.jl") end
@safetestset "Mesh integration tests" begin include("meshintegrationtest.jl") end
@safetestset "GlobalQuadrature integration tests" begin include("gquadinttest.jl") end