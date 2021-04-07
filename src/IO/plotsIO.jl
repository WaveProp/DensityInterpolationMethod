"""
    File that contains various recipes for plotting meshes, elements
    and data.
"""

"""
    surfacemeshplot(mesh:GenericMesh; npoints=5)
    surfacemeshplot!(mesh:GenericMesh; npoints=5)

Function that plots the surface of each element of a mesh.
It is possible that some elements may not be displayed properly, 
refer to https://github.com/plotly/plotly.py/issues/2194.
"""
@userplot SurfaceMeshPlot
@recipe function f(h::SurfaceMeshPlot; npoints=5)
    mesh = h.args[1]
    seriestype := :mesh3d

    coords_dict = Dict()
    coords_dict[FlatTriangleElement] = 
        [Point2D(u1, u2) for u1 in range(0, 1, length=npoints) 
                         for u2 in range(0, 1-u1, length=npoints)]
    coords_dict[QuadraticTriangleElement] = 
        coords_dict[FlatTriangleElement]

    for el in getelements(mesh)
        coords = coords_dict[typeof(el)]
        @series begin
            PlotElement(), el, coords
        end
    end
end

# for plotting surface elements 
struct PlotElement end
@recipe function f(::PlotElement, el::LagrangeElement, coords) 
    pt_list = [el(c) for c in coords]
    return pt_list
end

"""
    meshplot(mesh:GenericMesh; centers=0, normals=0, tangents=0)
    meshplot!(mesh:GenericMesh; centers=0, normals=0, tangents=0)

Function that plots the nodes of each element of a mesh. Optionally, it can plot
the center, the normal vector and the tangent vectors of each element.
"""
@userplot MeshPlot
@recipe function f(h::MeshPlot; nodes=true, centers=0, normals=0, tangents=0)
    mesh = h.args[1]
    plot_centers = centers > 0
    plot_normals = normals > 0
    plot_tangents = tangents > 0
    legend := false
    
    # Plot elements nodes
    markersize := 1
    markershape := :circle
    if nodes
        for el in getelements(mesh)
            @series begin
                el
            end
        end
    end

    # Plot centers/normals/tangents
    if plot_centers || plot_normals || plot_tangents
        c_list = []
        n_list = []
        jac_list = []

        # Get data
        for el in getelements(mesh)
            el_center, jac, _, n = getelementdata(el, getcenter(el))
            push!(c_list, el_center)
            push!(n_list, n .* normals)
            push!(jac_list, jac .* tangents)
        end

        # Plot centers
        if plot_centers
            markershape := :diamond
            for c in c_list
                @series begin 
                    c 
                end
            end
        end

        # Plot normals
        if plot_normals
            markershape := :rect
            for (c, n) in zip(c_list, n_list)
                @series begin
                    PlotQuiver(), c, n
                end
            end
        end

        # Plot tangents
        if plot_tangents
            markershape := :rect
            for (c, jac) in zip(c_list, jac_list)
                @series begin
                    PlotQuiver(), c, jac[:,1]
                end
                @series begin
                    PlotQuiver(), c, jac[:,2]
                end
            end
        end
    end
end

# for plotting vector fields (quiver plot)
struct PlotQuiver end
@recipe function f(::PlotQuiver, point::T, vector::T) where {T<:Point3D}
    sum = point + vector
    @series begin
        markershape := :none
        seriestype := :path3d
        [point, sum]
    end
    @series begin
        sum
    end
end

@recipe function f(pt::Point3D) 
    return [pt[1]], [pt[2]], [pt[3]]
end

@recipe function f(pts::AbstractVector{Point3D}) 
    x = [p[1] for p in pts]
    y = [p[2] for p in pts]
    z = [p[3] for p in pts]
    return x, y, z
end

@recipe function f(::Type{FlatTriangleElement}, el::FlatTriangleElement)
    n = getnodes(el)
    # Reorder nodes for visualization purposes
    nodList = (n[1], n[2], n[3], n[1]) 
    ptList = [el(nod) for nod in nodList]
    return ptList
end

@recipe function f(::Type{QuadraticTriangleElement}, el::QuadraticTriangleElement)
    n = getnodes(el)
    # Reorder nodes for visualization purposes
    nodList = (n[1], n[4], n[2], n[5], n[3], n[6], n[1])   
    ptList = [el(nod) for nod in nodList]
    return ptList
end

