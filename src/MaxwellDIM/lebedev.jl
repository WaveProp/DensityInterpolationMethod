"""
Definition of Lebedev points on the unit sphere.
"""

"""
    get_sphere_sources_lebedev(nsources, radius, center)

Returns the Lebedev points on the sphere of radius `radius`
and center `center`.
"""
function get_sphere_sources_lebedev(nsources, radius, center)
    lpts = lebedev_points(nsources)
    Xs = Point3D[]
    for pt in lpts
        push!(Xs, radius*pt .+ center)
    end
    return Xs
end

"""
    lebedev_points(n)

Return the Lebedev points on the unit sphere.    
"""
function lebedev_points(n::Int)
    pts = Vector{Point3D}()
    if n<=6
        push!(pts,_sph_pt(0,90))
        push!(pts,_sph_pt(180,90))
        push!(pts,_sph_pt(90,90))
        push!(pts,_sph_pt(-90,90))
        push!(pts,_sph_pt(90,0))
        push!(pts,_sph_pt(90,180))
        return pts
    elseif n<=14
        push!(pts,lebedev_points(6)...)
        push!(pts,_sph_pt(45,   54.735610317245346))
        push!(pts,_sph_pt(45,   125.264389682754654))
        push!(pts,_sph_pt(-45,  54.735610317245346))
        push!(pts,_sph_pt(-45,  125.264389682754654))
        push!(pts,_sph_pt(135,  54.735610317245346))
        push!(pts,_sph_pt(135,  125.264389682754654))
        push!(pts,_sph_pt(-135, 54.735610317245346))
        push!(pts,_sph_pt(-135, 125.264389682754654))
        return pts
    elseif n<=26
        push!(pts,_sph_pt(0,90))
        push!(pts,_sph_pt(180,90))
        push!(pts,_sph_pt(90,90))
        push!(pts,_sph_pt(-90,90))
        push!(pts,_sph_pt(90,0))
        push!(pts,_sph_pt(90,180))
        push!(pts,_sph_pt(90,45))
        push!(pts,_sph_pt(90,135))
        push!(pts,_sph_pt(-90,45))
        push!(pts,_sph_pt(-90,135))
        push!(pts,_sph_pt(0,45))
        push!(pts,_sph_pt(0,135))
        push!(pts,_sph_pt(180,45))
        push!(pts,_sph_pt(180,135))
        push!(pts,_sph_pt(45,90))
        push!(pts,_sph_pt(-45,90))
        push!(pts,_sph_pt(135,90))
        push!(pts,_sph_pt(-135,90))
        push!(pts,_sph_pt(45,54.735610317245346))
        push!(pts,_sph_pt(45,125.264389682754654))
        push!(pts,_sph_pt(-45,54.735610317245346))
        push!(pts,_sph_pt(-45,125.264389682754654))
        push!(pts,_sph_pt(135,54.735610317245346))
        push!(pts,_sph_pt(135,125.264389682754654))
        push!(pts,_sph_pt(-135,54.735610317245346))
        push!(pts,_sph_pt(-135,125.264389682754654))
        return pts
    elseif n<=38
        push!(pts,  _sph_pt(0.000000000000000,    90.000000000000000))
        push!(pts,_sph_pt( 180.000000000000000,    90.000000000000000))
        push!(pts,_sph_pt(  90.000000000000000,    90.000000000000000))
        push!(pts,_sph_pt( -90.000000000000000,    90.000000000000000))
        push!(pts,_sph_pt(  90.000000000000000,     0.000000000000000))
        push!(pts,_sph_pt(  90.000000000000000,   180.000000000000000))
        push!(pts,_sph_pt(  45.000000000000000,    54.735610317245346))
        push!(pts,_sph_pt(  45.000000000000000,   125.264389682754654))
        push!(pts,_sph_pt( -45.000000000000000,    54.735610317245346))
        push!(pts,_sph_pt( -45.000000000000000,   125.264389682754654))
        push!(pts,_sph_pt( 135.000000000000000,    54.735610317245346))
        push!(pts,_sph_pt( 135.000000000000000,   125.264389682754654))
        push!(pts,_sph_pt(-135.000000000000000,    54.735610317245346))
        push!(pts,_sph_pt(-135.000000000000000,   125.264389682754654))
        push!(pts,_sph_pt(  62.632194841377327,    90.000000000000000))
        push!(pts,_sph_pt( -62.632194841377327,    90.000000000000000))
        push!(pts,_sph_pt( 117.367805158622687,    90.000000000000000))
        push!(pts,_sph_pt(-117.367805158622687,    90.000000000000000))
        push!(pts,_sph_pt(  27.367805158622673,    90.000000000000000))
        push!(pts,_sph_pt( -27.367805158622673,    90.000000000000000))
        push!(pts,_sph_pt( 152.632194841377355,    90.000000000000000))
        push!(pts,_sph_pt(-152.632194841377355,    90.000000000000000))
        push!(pts,_sph_pt(   0.000000000000000,    27.367805158622673))
        push!(pts,_sph_pt(   0.000000000000000,   152.632194841377355))
        push!(pts,_sph_pt( 180.000000000000000,    27.367805158622673))
        push!(pts,_sph_pt( 180.000000000000000,   152.632194841377355))
        push!(pts,_sph_pt(   0.000000000000000,    62.632194841377327))
        push!(pts,_sph_pt(   0.000000000000000,   117.367805158622687))
        push!(pts,_sph_pt( 180.000000000000000,    62.632194841377327))
        push!(pts,_sph_pt( 180.000000000000000,   117.367805158622687))
        push!(pts,_sph_pt(  90.000000000000000,    27.367805158622673))
        push!(pts,_sph_pt(  90.000000000000000,   152.632194841377355))
        push!(pts,_sph_pt( -90.000000000000000,    27.367805158622673))
        push!(pts,_sph_pt( -90.000000000000000,   152.632194841377355))
        push!(pts,_sph_pt(  90.000000000000000,    62.632194841377327))
        push!(pts,_sph_pt(  90.000000000000000,   117.367805158622687))
        push!(pts,_sph_pt( -90.000000000000000,    62.632194841377327))
        push!(pts,_sph_pt( -90.000000000000000,   117.367805158622687))
        return pts
    elseif n<=50
        push!(pts,_sph_pt(  0.000000000000000,    90.000000000000000))
        push!(pts,_sph_pt( 180.000000000000000,    90.000000000000000))
        push!(pts,_sph_pt(  90.000000000000000,    90.000000000000000))
        push!(pts,_sph_pt( -90.000000000000000,    90.000000000000000))
        push!(pts,_sph_pt(  90.000000000000000,     0.000000000000000))
        push!(pts,_sph_pt(  90.000000000000000,   180.000000000000000))
        push!(pts,_sph_pt(  90.000000000000000,    45.000000000000000))
        push!(pts,_sph_pt(  90.000000000000000,   135.000000000000000))
        push!(pts,_sph_pt( -90.000000000000000,    45.000000000000000))
        push!(pts,_sph_pt( -90.000000000000000,   135.000000000000000))
        push!(pts,_sph_pt(   0.000000000000000,    45.000000000000000))
        push!(pts,_sph_pt(   0.000000000000000,   135.000000000000000))
        push!(pts,_sph_pt( 180.000000000000000,    45.000000000000000))
        push!(pts,_sph_pt( 180.000000000000000,   135.000000000000000))
        push!(pts,_sph_pt(  45.000000000000000,    90.000000000000000))
        push!(pts,_sph_pt( -45.000000000000000,    90.000000000000000))
        push!(pts,_sph_pt( 135.000000000000000,    90.000000000000000))
        push!(pts,_sph_pt(-135.000000000000000,    90.000000000000000))
        push!(pts,_sph_pt(  45.000000000000000,    54.735610317245346))
        push!(pts,_sph_pt(  45.000000000000000,   125.264389682754654))
        push!(pts,_sph_pt( -45.000000000000000,    54.735610317245346))
        push!(pts,_sph_pt( -45.000000000000000,   125.264389682754654))
        push!(pts,_sph_pt( 135.000000000000000,    54.735610317245346))
        push!(pts,_sph_pt( 135.000000000000000,   125.264389682754654))
        push!(pts,_sph_pt(-135.000000000000000,    54.735610317245346))
        push!(pts,_sph_pt(-135.000000000000000,   125.264389682754654))
        push!(pts,_sph_pt(  45.000000000000000,    25.239401820678911))
        push!(pts,_sph_pt(  45.000000000000000,   154.760598179321079))
        push!(pts,_sph_pt( -45.000000000000000,    25.239401820678911))
        push!(pts,_sph_pt( -45.000000000000000,   154.760598179321079))
        push!(pts,_sph_pt( 135.000000000000000,    25.239401820678911))
        push!(pts,_sph_pt( 135.000000000000000,   154.760598179321079))
        push!(pts,_sph_pt(-135.000000000000000,    25.239401820678911))
        push!(pts,_sph_pt(-135.000000000000000,   154.760598179321079))
        push!(pts,_sph_pt(  71.565051177077990,    72.451599386207704))
        push!(pts,_sph_pt( -71.565051177077990,    72.451599386207704))
        push!(pts,_sph_pt(  71.565051177077990,   107.548400613792296))
        push!(pts,_sph_pt( -71.565051177077990,   107.548400613792296))
        push!(pts,_sph_pt( 108.434948822922010,    72.451599386207704))
        push!(pts,_sph_pt(-108.434948822922010,    72.451599386207704))
        push!(pts,_sph_pt( 108.434948822922010,   107.548400613792296))
        push!(pts,_sph_pt(-108.434948822922010,   107.548400613792296))
        push!(pts,_sph_pt(  18.434948822922017,    72.451599386207704))
        push!(pts,_sph_pt( 161.565051177078004,    72.451599386207704))
        push!(pts,_sph_pt(  18.434948822922017,   107.548400613792296))
        push!(pts,_sph_pt( 161.565051177078004,   107.548400613792296))
        push!(pts,_sph_pt( -18.434948822922017,    72.451599386207704))
        push!(pts,_sph_pt(-161.565051177078004,    72.451599386207704))
        push!(pts,_sph_pt( -18.434948822922017,   107.548400613792296))
        push!(pts,_sph_pt(-161.565051177078004,   107.548400613792296))
        return pts
    else
        error("Unable to return $n Lebedev points")       
    end
end

function _sph_pt(theta,phi,r=1, center=[0 0 0])
    # convert to radians    
    theta = theta * pi/180
    phi   = phi   * pi/180
    # perform translation and dilation
    x = center[1] + r*sin(phi)*cos(theta)
    y = center[2] + r*sin(phi)*sin(theta)
    z = center[3] + r*cos(phi)
    return Point3D(x,y,z)
end