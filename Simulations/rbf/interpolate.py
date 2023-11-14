"""
An RBF interpolant class.
"""

import numpy as np
import numpy.linalg as la
from .poly_utils import poly_powers
from .rbf import RBF, PHS, pairwise_dist
from scipy.spatial import KDTree
from .stencil import Stencil


class Interpolator:
    def __init__(
        self, *, stencil: Stencil, fs: np.ndarray, rbf: RBF, poly_deg: int = -1
    ):
        assert len(fs) == stencil.num_points
        self.stencil = stencil
        self.rbf = rbf
        self.poly_deg = poly_deg
        self.find_weights(fs)

    def find_weights(self, fs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        A = self.stencil.interpolation_matrix(rbf=self.rbf, poly_deg=self.poly_deg)
        ys = np.zeros(A.shape[0])
        ys[: self.stencil.num_points] = fs
        try:
            weights = la.solve(A, ys)
        except la.LinAlgError:
            raise ValueError(
                """Interpolation matrix is singular.
                I haven't yet implemented the Schur decomp solve yet.
                For now, use shape parameter RBFs."""
            )
        self.rbf_weights = weights[: self.stencil.num_points]
        self.poly_weights = weights[self.stencil.num_points :]

    def __call__(self, z):
        z = self.stencil.shift_points(z)
        rbf_val = sum(
            w * self.rbf(la.norm(z - point))
            for w, point in zip(self.rbf_weights, self.stencil.scaled_points)
        )
        poly_val = sum(
            w * poly(z)
            for w, poly in zip(
                self.poly_weights,
                poly_powers(dim=self.stencil.dim, max_deg=self.poly_deg),
            )
        )
        return rbf_val + poly_val


class LocalInterpolator:
    def __init__(
        self,
        *,
        points: np.ndarray[float],
        fs: np.ndarray[float],
        rbf: RBF,
        poly_deg: int,
        stencil_size: int
    ):
        self.points = points
        self.fs = fs
        self.rbf = rbf
        self.poly_deg = poly_deg
        self.stencil_size = stencil_size

        self.kdt = KDTree(points)
        self.generate_stencils()
        self.interpolate_on_stencils()

    def generate_stencils(self):
        self.stencils = []
        for point in self.points:
            _, neighbor_indices = self.kdt.query(point, self.stencil_size)
            self.stencils.append(neighbor_indices)

    def interpolate_on_stencils(self):
        self.local_interpolants = []
        for stencil in self.stencils:
            self.local_interpolants.append(
                Interpolator(
                    stencil=Stencil(self.points[stencil]),
                    fs=self.fs[stencil],
                    rbf=self.rbf,
                    poly_deg=self.poly_deg,
                )
            )

    def __call__(self, eval_point: np.ndarray) -> float:
        _, index = self.kdt.query(eval_point, 1)
        return self.local_interpolants[index](eval_point)


def interpolate(
    points: np.ndarray, fs: np.ndarray, rbf: RBF = PHS(3), poly_deg: int = 2
):
    return Interpolator(stencil=Stencil(points), fs=fs, rbf=rbf, poly_deg=poly_deg)
