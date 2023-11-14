"""
Stencil class for RBF interpolation.
"""

from functools import cache
import numpy as np
import numpy.linalg as la
from .poly_utils import poly_basis_dim, poly_powers
from .rbf import RBF, pairwise_diff, pairwise_dist


class Stencil:
    def __init__(
        self,
        points: np.ndarray[float],
    ):
        shape = points.shape
        if len(shape) == 1:
            self.dim = 1
        else:
            assert len(shape) == 2
            self.dim = shape[1]
        self.points = points

        self.upper_bounds = np.max(points, axis=0)
        self.lower_bounds = np.min(points, axis=0)

    @property
    @cache
    def num_points(self):
        return len(self.points)

    def shift_points(self, points: np.ndarray) -> np.ndarray:
        mids = (self.upper_bounds + self.lower_bounds) / 2
        widths = self.upper_bounds - self.lower_bounds
        return (points - mids) / widths

    @property
    @cache
    def scaled_points(self):
        return self.shift_points(self.points)

    @property
    def pairwise_diff(self) -> np.ndarray[float]:
        points = self.scaled_points
        return pairwise_diff(points, points)

    @property
    def dist_mat(self) -> np.ndarray[float]:
        points = self.scaled_points
        return pairwise_dist(points, points)

    def rbf_mat(self, rbf: RBF):
        return rbf(self.dist_mat)

    def poly_mat(self, poly_deg: int) -> np.ndarray[float]:
        P = np.ones((self.num_points, poly_basis_dim(self.dim, poly_deg)))
        for index, poly in enumerate(poly_powers(self.dim, max_deg=poly_deg)):
            P[:, index] = poly(self.scaled_points)
        return P

    def interpolation_matrix(self, rbf: RBF, poly_deg: int = -1) -> np.ndarray[float]:
        P = self.poly_mat(poly_deg=poly_deg)
        basis_size = P.shape[1]
        zeros = np.zeros((basis_size, basis_size))
        return np.block([[self.rbf_mat(rbf=rbf), P], [P.T, zeros]])
