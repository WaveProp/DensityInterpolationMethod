using Test
using LinearAlgebra
using DensityInterpolationMethod
using DensityInterpolationMethod.IO
using DensityInterpolationMethod.Integration
include("../test_utils.jl")

# Tolerances
const TOL1 = 4e-2
const TOL2 = 1e-2
const TOL3 = 6e-4

@testset "Sphere area" begin
    mesh_filename = "meshes/sphere1.geo"
    sph_area = 4pi    # unit sphere area
    ftest(_) = 1      # for computing area

    @testset "Element order 1" begin
        ELEM_ORDER = 1
        mesh = read_gmsh_geo(mesh_filename, h=HMAX, order=ELEM_ORDER, verbosity=false);
        
        tolerances = [TOL1, TOL1, TOL1, TOL1]
        for i in 1:QRULE_MAX_ORDER
            area = integrate(mesh, ftest, order=i)
            @test relative_error(area, sph_area) < tolerances[i]
        end
    end

    @testset "Element order 2" begin
        ELEM_ORDER = 2
        mesh = read_gmsh_geo(mesh_filename, h=HMAX, order=ELEM_ORDER, verbosity=false);

        tolerances = [TOL2, TOL3, TOL3, TOL3]
        for i in 1:QRULE_MAX_ORDER
            area = integrate(mesh, ftest, order=i)
            @test relative_error(area, sph_area) < tolerances[i]
        end
    end
end

@testset "Flux surface integrals" begin
    mesh_filename = "meshes/sphere1.geo"

    # (normalized) Electric field of unit point charge centered
    # at the origin
    ftest(x) = x / (4pi * norm(x)^3)      
    exactcharge = 1          # electric charge at the origin

    @testset "Element order 1" begin
        ELEM_ORDER = 1
        mesh = read_gmsh_geo(mesh_filename, h=HMAX, order=ELEM_ORDER, verbosity=false);

        tolerances = [TOL1, TOL3, TOL3, TOL3]
        for i in 1:QRULE_MAX_ORDER
            charge = integrateflux(mesh, ftest, order=i)
            @test relative_error(charge, exactcharge) < tolerances[i]
        end
    end

    @testset "Element order 2" begin
        ELEM_ORDER = 2
        mesh = read_gmsh_geo(mesh_filename, h=HMAX, order=ELEM_ORDER, verbosity=false);
        
        tolerances = [TOL2, TOL3, TOL3, TOL3]
        for i in 1:QRULE_MAX_ORDER
            charge = integrateflux(mesh, ftest, order=i)
            @test relative_error(charge, exactcharge) < tolerances[i]
        end
    end
end