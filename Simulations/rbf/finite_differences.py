
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
    def scale_powers(self) -> np.ndarray[int]:
        return np.array([1])


class Laplacian(LinearFunctional):

    def __init__(self, dim: int):
        self.dim = dim

    def rbf_op(self, rbf: RBF, r: float, d: np.ndarray[float]) -> float:
        return rbf.d2r(r) + (self.dim-1)*rbf.dr_div_r(r)

    def poly_op(self, poly: Monomial, d: np.ndarray[float]) -> float:
        return poly.diff(d, *[2 for _ in range(self.dim)])

    @property
    def scale_powers(self) -> np.ndarray[int]:
        return np.array([2 for _ in range(self.dim)])
