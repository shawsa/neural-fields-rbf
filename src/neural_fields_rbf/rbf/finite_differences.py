from .rbf import RBF, PHS
from .poly_utils import Monomial
from .linear_functional import LinearFunctional

import numpy as np


class Derivative1D(LinearFunctional):
    def rbf_op(self, rbf: RBF, r: float, d: np.ndarray[float]) -> float:
        return d * rbf.dr_div_r(r)

    def poly_op(self, poly: Monomial, d: np.ndarray[float]) -> float:
        return poly.diff(d, 1)

    @property
    def scale_power(self) -> np.ndarray[int]:
        return 1


class Laplacian(LinearFunctional):
    """Not working in higher than 1D yet."""

    def __init__(self, dim: int):
        self.dim = dim

    def rbf_op(self, rbf: RBF, r: float, d: np.ndarray[float]) -> float:
        return rbf.d2r(r) + (self.dim - 1) * rbf.dr_div_r(r)

    def poly_op(self, poly: Monomial, d: np.ndarray[float]) -> float:
        return sum(
            poly.diff(d, *[2 * (i == j) for i in range(self.dim)])
            for j in range(self.dim)
        )

    @property
    def scale_power(self) -> np.ndarray[int]:
        return 2
