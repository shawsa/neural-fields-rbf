"""
Python classes for holding point clouds and point generation.
"""

from abc import ABC, abstractmethod
import numpy as np
import numpy.linalg as la
from scipy.spatial import KDTree
from typing import Callable


class RepulsionKernel(ABC):
    """Used to redistribute nodes to make them more regular."""

    @abstractmethod
    def __call__(self, direction: np.ndarray):
        raise NotImplementedError


class GaussianRepulsionKernel(RepulsionKernel):
    def __init__(self, height: float, shape: float):
        self.shape = shape
        self.height = height

    def __call__(self, points: np.ndarray):
        ret = points.copy()
        mags = la.norm(ret, axis=-1)
        mags = self.height * np.exp(-((mags / self.shape) ** 2)) / mags
        ret[..., 0] *= mags
        ret[..., 1] *= mags
        return ret


class ConstRepulsionKernel(RepulsionKernel):
    def __init__(self, const: float):
        self.const = const

    def __call__(self, points: np.ndarray):
        ret = points.copy()
        mags = la.norm(ret, axis=-1)
        ret[..., 0] /= mags
        ret[..., 1] /= mags
        return self.const * ret


class PointCloud:
    def __init__(
        self,
        points: np.ndarray,
        num_interior: int,
        num_boundary: int,
        num_ghost: int = 0,
    ):
        assert num_interior + num_boundary + num_ghost == len(points)
        self.all_points = points
        self.num_interior = num_interior
        self.num_boundary = num_boundary
        self.num_ghost = num_ghost

        self.points = points[: num_interior + num_boundary]

    @property
    def inner(self):
        return self.points[: self.num_interior]

    @inner.setter
    def inner(self, value):
        self.points[: self.num_interior] = value

    @property
    def boundary(self):
        return self.points[self.num_interior : self.num_interior + self.num_boundary]

    @property
    def ghost(self):
        start = self.num_interior + self.num_boundary
        return self.all_points[start : start + self.num_ghost]

    @boundary.setter
    def boundary(self, value):
        self.points[self.num_interior :] = value

    def settle(
        self,
        *,
        kernel: RepulsionKernel,
        rate: float,
        num_neighbors: int,
        force: Callable = None,
        use_ghost: bool = False,
        repeat=1,
    ):
        points = self.points
        if use_ghost:
            points = self.all_points
        kdt = KDTree(points)
        # update = np.zeros_like(self.inner)
        # for index, point in enumerate(self.inner):
        #     _, neighbors_indices = kdt.query(point, num_neighbors + 1)
        #     neighbors = points[neighbors_indices][1:]
        #     update[index] = np.average(kernel(neighbors - point), axis=0)
        _, neighbors_indices = kdt.query(self.inner, num_neighbors + 1)
        neighbors = points[neighbors_indices][:, 1:]
        update = np.average(kernel(neighbors - self.inner[:, np.newaxis, :]), axis=1)
        self.inner -= rate * update
        if force is not None:
            self.inner += force(self.inner)
