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

    def __call__(self, direction: np.ndarray):
        r = la.norm(direction)
        vec = direction / r
        return vec * self.height * np.exp(-((r / self.shape) ** 2))


class PointCloud:
    def __init__(self, points: np.ndarray, num_interior: int, num_boundary: int):
        assert num_interior + num_boundary == len(points)
        self.points = points
        self.num_interior = num_interior
        self.num_boundary = num_boundary

    @property
    def inner(self):
        return self.points[: self.num_interior]

    @inner.setter
    def inner(self, value):
        self.points[: self.num_interior] = value

    @property
    def boundary(self):
        return self.points[self.num_interior :]

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
    ):
        kdt = KDTree(self.points)
        update = np.empty_like(self.inner)
        for index, point in enumerate(self.inner):
            _, neighbors_indices = kdt.query(point, num_neighbors)
            neighbors = self.points[[id for id in neighbors_indices if id != index]]
            update[index] = sum(kernel(x2 - point) for x2 in neighbors) / len(neighbors)
            if force is not None:
                update[index] -= force(point)
        self.inner -= rate * update
