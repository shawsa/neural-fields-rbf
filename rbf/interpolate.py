"""
Radial basis function interpolation classes.
"""

import numpy as np
import numpy.linalg as la
from scipy.spatial import distance_matrix, KDTree

from .poly_utils import poly_powers
from .rbf import RBF, PHS
from .stencil import Stencil


class Interpolator:
    def __init__(
        self, *, stencil: Stencil, fs: np.ndarray, rbf: RBF, poly_deg: int = -1
    ):
        assert len(fs) == stencil.num_points
        self.points = stencil.points
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
            raise ValueError("RBF matrix singular")

        self.rbf_weights = weights[: self.stencil.num_points]
        self.poly_weights = weights[self.stencil.num_points :]

    def point_eval(self, *coords: list[float]) -> float:
        """Evaluate the RBF interpolant"""
        z = self.stencil.shift_and_scale(np.array(coords))
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

    def batch_eval(self, zs: np.ndarray[float]) -> np.ndarray[float]:
        """Evaluate the RBF interpolant. zs is a matrix where each row
        is an evaluation point."""
        zs_shifted = self.stencil.shift_and_scale(zs)
        rbf_val = (
            self.rbf(distance_matrix(zs_shifted, self.stencil.scaled_points))
        ) @ self.rbf_weights
        poly_val = sum(
            w * poly(zs_shifted)
            for w, poly in zip(
                self.poly_weights,
                poly_powers(dim=self.stencil.dim, max_deg=self.poly_deg),
            )
        )
        return rbf_val + poly_val

    def __call__(self, *args):
        """Evaluate the RBF interpolant. This is a convenicence
        function for interactive use, and dispatches to either
        self.point_eval or self.batch_eval. In production, use
        these instead as they have fixed type arguments.
        """

        if len(args) > 1:
            return self.point_eval(*args)
        z = args[0]
        if isinstance(z, list):
            return self.point_eval(*z)
        assert isinstance(
            z, np.ndarray
        ), f"Type {z=} not recognized, use list or np.array"
        if z.shape == self.points[0].shape:
            return self.point_eval(*z)
        return self.batch_eval(z)


class LocalInterpolator(Interpolator):
    def __init__(
        self,
        *,
        points: np.ndarray[float],
        fs: np.ndarray[float],
        rbf: RBF,
        poly_deg: int,
        stencil_size: int,
    ):
        self.points = points
        self.fs = fs
        self.rbf = rbf
        self.poly_deg = poly_deg
        self.stencil_size = stencil_size
        self.generate_stencils()
        self.form_interpolants()

    def generate_stencils(self):
        self.kdt = KDTree(self.points)
        self.stencils = []
        for point in self.points:
            _, neighbor_indices = self.kdt.query(point, self.stencil_size)
            self.stencils.append(neighbor_indices)

    def form_interpolants(self):
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

    def point_eval(self, *coords: list[float]) -> float:
        point = np.array(coords)
        _, index = self.kdt.query(point, 1)
        return self.local_interpolants[index].point_eval(*coords)

    def batch_eval(self, points: np.ndarray[float]) -> np.ndarray[float]:
        ret = np.zeros(len(points))
        _, indices = self.kdt.query(points)
        for index, approx in enumerate(self.local_interpolants):
            mask = indices == index
            ret[mask] = approx.batch_eval(points[mask])
        return ret


class LocalInterpolator1D(LocalInterpolator):
    def generate_stencils(self):
        """In 1D, we cannot use a KD-tree, but finding neighbors is simple."""
        self.stencils = []
        for point in self.points:
            lst = [(index, abs(x - point)) for index, x in enumerate(self.points)]
            lst.sort(key=lambda tup: tup[1])
            self.stencils.append(
                np.array([tup[0] for tup in lst[: self.stencil_size]], dtype=int)
            )

    def __call__(self, eval_point: float) -> float:
        index = np.argmin(np.abs(eval_point - self.points))
        return self.local_interpolants[index](eval_point)


def interpolate(
    points: np.ndarray, fs: np.ndarray, rbf: RBF = PHS(3), poly_deg: int = 2
):
    return Interpolator(stencil=Stencil(points), fs=fs, rbf=rbf, poly_deg=poly_deg)
