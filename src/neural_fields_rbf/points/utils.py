import os
import pickle

DIR, _ = os.path.split(os.path.realpath(__file__))
CART_STENCILS = os.path.join(DIR, "cartesian_stencil_sizes.pickle")
with open(CART_STENCILS, "rb") as f:
    cart_sizes = pickle.load(f)
HEX_STENCILS = os.path.join(DIR, "hex_stencil_sizes.pickle")
with open(HEX_STENCILS, "rb") as f:
    hex_sizes = pickle.load(f)


def _smallest_greater_than(k_min, sizes):
    for k in sizes:
        if k >= k_min:
            break
    return k


def cart_stencil_min(k_min: int) -> int:
    """
    Returns the smallest stencil size bigger than k_min
    for which a Cartesian grid has unambiguous stencils.
    """
    return _smallest_greater_than(k_min, cart_sizes)


def hex_stencil_min(k_min: int) -> int:
    """
    Returns the smallest stencil size bigger than k_min
    for which a hex grid has unambiguous stencils.
    """
    return _smallest_greater_than(k_min, hex_sizes)


def poly_stencil_min(deg: int) -> int:
    """
    Returns the number of polynomial basis terms in 2D.
    The stencil size must be at least this big for the interpolation matrix
    to be unisolvent in theory.
    """
    return ((deg + 1) * (deg + 2)) // 2


def get_stencil_size(deg: int, stability_factor: float = 1.1):
    """Recommnded minimum stencil size for scattered nodes in 2D.

    Find the minimum number of points required to invert the
    local RBF collocation matrix of the specified degree. Then
    multiply that by the specified stability factor and choose
    the smallest hex stencil size larger than the result.
    """
    return hex_stencil_min(stability_factor * poly_stencil_min(deg))
