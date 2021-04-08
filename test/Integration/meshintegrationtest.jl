using Test
using DensityInterpolationMethod
using DensityInterpolationMethod.IO
using DensityInterpolationMethod.Integration
include("../test_utils.jl")

# Constants
const sph_area = 4pi    # unit sphere area
const HMAX = 0.5        # max mesh size

# Tolerances
const TOL1 = 4e-2
const TOL2 = 1e-2
const TOL3 = 5e-4

@testset "Sphere area" begin
    mesh_filename = "meshes/sphere1.geo"
    ftest(_) = 1      # for computing area

    @testset "Element order 1" begin
        ELEM_ORDER = 1
        mesh = read_gmsh_geo(mesh_filename, h=HMAX, order=ELEM_ORDER, verbosity=false);
        
        area = integrate(mesh, ftest, order=1)
        @test relative_error(area, sph_area) < TOL1

        area = integrate(mesh, ftest, order=2)
        @test relative_error(area, sph_area) < TOL1

        area = integrate(mesh, ftest, order=3)
        @test relative_error(area, sph_area) < TOL1

        area = integrate(mesh, ftest, order=4)
        @test relative_error(area, sph_area) < TOL1
    end

    @testset "Element order 2" begin
        ELEM_ORDER = 2
        mesh = read_gmsh_geo(mesh_filename, h=HMAX, order=ELEM_ORDER, verbosity=false);
        
        area = integrate(mesh, ftest, order=1)
        @test relative_error(area, sph_area) < TOL2

        area = integrate(mesh, ftest, order=2)
        @test relative_error(area, sph_area) < TOL3

        area = integrate(mesh, ftest, order=3)
        @test relative_error(area, sph_area) < TOL3

        area = integrate(mesh, ftest, order=4)
        @test relative_error(area, sph_area) < TOL3
    end
end