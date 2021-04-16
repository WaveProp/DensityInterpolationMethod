"""
Definition of Green tensor for Maxwell's equations.
"""

function single_layer_kernel(x, y, k)  
    x==y && return zero(Point3D)
    rvec = x - y
    r = norm(rvec)
    g   = 1/(4π)/r * exp(im*k*r)
    gp  = im*k*g - g/r
    gpp = im*k*gp - gp/r + g/r^2
    ID    = SMatrix{3,3,Float64,9}(1,0,0,0,1,0,0,0,1)
    RRT   = rvec*transpose(rvec) # rvec ⊗ rvecᵗ
    return  g*ID + 1/k^2*(gp/r*ID + (gpp/r^2 - gp/r^3)*RRT)
end

# Double Layer Kernel
# n × ∇ × G = γ₁ G
function double_layer_kernel(x, y, k, ny) 
    x==y && return zero(Point3D)
    rvec = x - y
    r = norm(rvec)
    g   = 1/(4π)/r * exp(im*k*r)
    gp  = im*k*g - g/r
    ID    = SMatrix{3,3,Float64,9}(1,0,0,0,1,0,0,0,1)
    ncross = transpose(SMatrix{3,3,Float64,9}(0,-ny[3],ny[2],
                                              ny[3],0,-ny[1],
                                              -ny[2],ny[1],0))
    rcross = transpose(SMatrix{3,3,Float64,9}(0,-rvec[3],rvec[2],
                                              rvec[3],0,-rvec[1],
                                              -rvec[2],rvec[1],0))
    return -gp/r*ncross*rcross
    #return gp/r*rcross*ncross   # CHECK!!!!!
end

function single_layer_kernel_v2(x, y, k, ϕy)  
    x==y && return zero(Point3D)
    rvec = x - y
    r = norm(rvec)
    rhat = rvec/r
    kr = k*r
    kr2 = kr^2
    g = 1/(4π)/r * exp(im*k*r)

    result = (1 + im/kr - 1/kr2)*ϕy
    result += (-1 - 3im/kr + 3/kr2)*dot(ϕy, rhat)*rhat
    result *= g
    return result
end

# Double Layer Kernel
# n × ∇ × G = γ₁ G
function double_layer_kernel_v2(x, y, k, ny, ϕy) 
    x==y && return zero(Point3D)
    rvec = x - y
    r = norm(rvec)
    g = 1/(4π)/r * exp(im*k*r)
    gp  = (im*k - 1/r)*g
    result = gp/r*cross(ny, cross(rvec, ϕy))
    return result
end

# test
if false
    k = 10
    x = Point3D(1, 4, 5)
    y = Point3D(-4, 4, 0)
    ny = Point3D(1, 1, 1)
    ny = ny / norm(ny)
    ϕy = cross(ny, Point3D(-6, 1, -4))

    SL1 = single_layer_kernel(x, y, k)*ϕy  
    SL2 = single_layer_kernel_v2(x, y, k, ϕy)  
    @assert isapprox(SL1, SL2) 
    @btime single_layer_kernel($x, $y, $k)*$ϕy      # 51.797 ns (0 allocations: 0 bytes)
    @btime single_layer_kernel_v2($x, $y, $k, $ϕy)  # 31.568 ns (0 allocations: 0 bytes)

    DL1 = double_layer_kernel(x, y, k, ny)*ϕy  
    DL2 = double_layer_kernel_v2(x, y, k, ny, ϕy) 
    @assert isapprox(-DL1, DL2)
    @btime double_layer_kernel($x, $y, $k, $ny)*$ϕy        # 46.571 ns (0 allocations: 0 bytes)
    @btime double_layer_kernel_v2($x, $y, $k, $ny, $ϕy)    # 30.037 ns (0 allocations: 0 bytes)
end