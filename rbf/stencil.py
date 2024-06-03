"""
Stencil class for RBF interpolation.
"""

import numpy as np
from .poly_utils import poly_basis_dim, poly_powers
from .rbf import RBF, pairwise_diff, pairwise_dist


class Stencil:
    def __init__(
        self,
        points: np.ndarray[float],
        center: (np.ndarray[float] | None) = None,
    ):
        shape = points.shape
        if len(shape) == 1:
            self.dim = 1
        else:
            assert len(shape) == 2
            self.dim = shape[1]
        self.points = points

        if center is None:
            self.center = self.points[0]
        else:
            self.center = center
        self.scaled_points = self.points - self.center
        self.scale_factor = np.max(np.abs(self.points - self.center))
        self.scaled_points /= self.scale_factor

    def shift_and_scale(self, points: np.ndarray[float]) -> np.ndarray[float]:
        return 1/self.scale_factor * (points - self.center)

    @property
    def num_points(self):
        return len(self.points)

    @property
    def pairwise_diff(self) -> np.ndarray[float]:
        return pairwise_diff(self.scaled_points, self.scaled_points)

    @property
    def dist_mat(self) -> np.ndarray[float]:
        return pairwise_dist(self.scaled_points, self.scaled_points)

    def rbf_mat(self, rbf: RBF):
        return rbf(self.dist_mat)

    def poly_mat(self, poly_deg: int) -> np.ndarray[float]:
        P = np.ones((self.num_points, poly_basis_dim(self.dim, poly_deg)))
        for index, poly in enumerate(poly_powers(self.dim, max_deg=poly_deg)):
            P[:, index] = poly(self.scaled_points)
        return P

    def interpolation_matrix(self, rbf: RBF, poly_deg: int = -1) -> np.ndarray[float]:
        if poly_deg == -1:
            return self.rbf_mat(rbf=rbf)
        P = self.poly_mat(poly_deg=poly_deg)
        basis_size = P.shape[1]
        zeros = np.zeros((basis_size, basis_size))
        return np.block([[self.rbf_mat(rbf=rbf), P], [P.T, zeros]])
