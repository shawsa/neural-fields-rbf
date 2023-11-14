"""
Utilities for multidimensional polynomials.

RBF methods can included polynomial basis terms to increase accuracy.
Polynomial basis terms are also necessary to ensure invertability of
RBF matrices based on polyharmonic spline RBFs.
"""

from functools import cache, reduce
from math import comb, perm
import numpy as np
import operator
from typing import Collection, Generator


class Monomial:
    def __init__(self, *pows: tuple[int]):
        self.pows = pows

    def __call__(self, points: np.ndarray[float]):
        """
        Apply the polynomial powers to an array of points, pointwise.
        """
        shape = points.shape
        if len(self.pows) == 1:
            points = points.reshape(*shape, 1)
        ret = reduce(
            operator.mul,
            (x**p for x, p in zip(points.T, self.pows)),
        )
        if len(self.pows) == 1:
            return ret.flatten()
        return ret

    def diff(self, points: np.ndarray[float], *orders: Collection[int]):
        shape = points.shape
        if len(self.pows) == 1:
            points = points.reshape(*shape, 1)
        new_pows = []
        coeff = 1
        for o, p in zip(orders, self.pows):
            new_pows.append(p - o)
            coeff *= perm(p, o)
        ret = coeff * reduce(
            operator.mul,
            (x**max(p, 0) for x, p in zip(points.T, new_pows)),
        )
        if len(self.pows) == 1:
            return ret.flatten()
        return ret


PolyPowGen = Generator[Monomial, None, None]


@cache
def poly_basis_dim(dim: int, deg: int) -> int:
    """
    The number of polynomial basis terms up to a given degree for a given
    dimension.
    """
    return comb(dim + deg, deg)


def poly_powers_of_deg(dim: int, deg: int) -> PolyPowGen:
    """
    Find all combinations of exponents of polynomials of the specified degree for
    the specified dimension. Return a generator that yields tuples of the exponents.
    """
    if dim == 1:
        yield Monomial(deg)
    else:
        for d in range(deg + 1):
            yield from (
                Monomial(*poly.pows, d) for poly in poly_powers_of_deg(dim - 1, deg - d)
            )


def poly_powers_gen(dim: int, max_deg: int) -> PolyPowGen:
    """
    Find all combinations of exponents of polynomials of at most the specified degree
    for the specified dimension. Return a generator that yields tuples of the exponents.
    """
    for deg in range(max_deg + 1):
        yield from poly_powers_of_deg(dim, deg)


@cache
def poly_powers(dim: int, max_deg: int) -> tuple[Monomial]:
    """
    Cached version of poly_powers_gen that returns tuple.
    More efficient for repeated calls with same inputs.
    """
    return tuple(poly_powers_gen(dim, max_deg))
