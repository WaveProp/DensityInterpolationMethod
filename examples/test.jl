"""
File for testing stuff
"""
module holamod
using StaticPolynomials
export jacobian

jacobian(x::Int) = 1
end

using .holamod



