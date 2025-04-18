from abc import ABC, abstractmethod, abstractproperty
from functools import reduce
import numpy as np
import numpy.linalg as la
from operator import mul
from .poly_utils import Monomial, poly_basis_dim, poly_powers_gen
from .rbf import RBF, pairwise_dist, pairwise_diff
from .stencil import Stencil
from typing import Callable


class LinearFunctional(ABC):
    """
    A base class for linear functionals.
    Be sure to account for stencil scaling.
    """

    # this call signature may need to change
    # currenty, this is ideal for differential functionals
    @abstractmethod
    def rbf_op(self, rbf: RBF, r: float, d: np.ndarray[float]) -> float:
        raise NotImplementedError

    @abstractmethod
    def poly_op(self, poly: Monomial, d: np.ndarray[float]) -> float:
        raise NotImplementedError

    @abstractproperty
    def scale_power(self) -> int:
        raise NotImplementedError


class FunctionalStencil(Stencil):
    def __init__(
        self,
        points: np.ndarray[float],
    ):
        super(FunctionalStencil, self).__init__(points)
        self.center = points[0]

    def weights(self, rbf: RBF, op: LinearFunctional, poly_deg: int):
        d = self.pairwise_diff[0]
        r = self.dist_mat[0]
        mat = self.interpolation_matrix(rbf, poly_deg)
        rhs = np.zeros_like(mat[0])
        rhs[: len(self.points)] = op.rbf_op(rbf, r, d).ravel()
        rhs[len(self.points) :] = np.array(
            [
                op.poly_op(poly, np.zeros(self.dim))
                for poly in poly_powers_gen(self.dim, poly_deg)
            ]
        )
        weights = la.solve(mat, rhs)
        return weights[: len(self.points)] / (self.scaling**op.scale_power)
