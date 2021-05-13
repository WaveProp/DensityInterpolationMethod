# Constants
const HMAX = 0.5             # max mesh size
const QRULE_MAX_ORDER = 10    # max quadrature rule order

function relative_error(approx, exact)
    return abs((approx - exact) / exact)
end