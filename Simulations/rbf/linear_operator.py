from abc import ABC, abstractmethod
import numpy as np
import numpy.linalg as la
from .poly_utils import Monomial, poly_basis_dim, poly_powers_gen
from .rbf import RBF, pairwise_dist, pairwise_diff
from .stencil import Stencil
from typing import Callable


class LinearOperator(ABC):
    @abstractmethod
    def rbf_op(self, rbf: RBF, r: float, d: np.ndarray[float]) -> float:
        raise NotImplementedError

    @abstractmethod
    def poly_op(self, poly: Monomial, d: np.ndarray[float]) -> float:
        raise NotImplementedError


class OperatorStencil(Stencil):
    def __init__(
        self,
        points: np.ndarray[float],
    ):
        super(OperatorStencil, self).__init__(points)
        self.center = points[0]

    def weights(self, rbf: RBF, op: LinearOperator, poly_deg: int):
        d = self.pairwise_diff[0]
        r = self.dist_mat[0]
        mat = self.interpolation_matrix(rbf, poly_deg)
        rhs = np.zeros_like(mat[0])
        rhs[: len(self.points)] = op.rbf_op(rbf, r, d).ravel()
        rhs[len(self.points) :] = np.array(
            [
                op.poly_op(poly, np.array([0.0]))
                for poly in poly_powers_gen(self.dim, poly_deg)
            ]
        ).T
        weights = la.solve(mat, rhs)
        return weights[: len(self.points)]
