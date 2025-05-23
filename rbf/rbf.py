"""
Classes for radial basis functions.
"""

from abc import ABC, abstractmethod, abstractstaticmethod
import numpy as np
import numpy.linalg as la
from scipy.spatial import distance_matrix


def pairwise_diff(p1: np.ndarray[float], p2: np.ndarray[float]) -> np.ndarray[float]:
    """Return a matrix of the poirwise differences between two
    vectors of points.
    The points must be of the same dimension.
    """
    return np.array([[(x - y) for x in p1] for y in p2])


# def pairwise_dist(p1: np.ndarray[float], p2: np.ndarray[float]) -> np.ndarray[float]:
#     """Return a matrix of the pairwise distances between two
#     vectors of points.
#     The points must be of the same dimension.
#     """
#     return np.array([[la.norm(x - y) for x in p1] for y in p2])


pairwise_dist = distance_matrix


class RBF(ABC):
    @abstractmethod
    def __call__(self, r):
        raise NotImplementedError

    @abstractmethod
    def dr(self, r):
        raise NotImplementedError

    @abstractmethod
    def d2r(self, r):
        raise NotImplementedError

    @abstractmethod
    def dr_div_r(self, r):
        raise NotImplementedError

    @abstractmethod
    def __repr__(self):
        raise NotImplementedError


class OddPHS(RBF):
    def __init__(self, deg: int):
        assert type(deg) is int
        assert deg > 0
        assert deg % 2 == 1
        self.deg = deg

    def __call__(self, r):
        return r**self.deg

    def dr(self, r):
        return self.deg * r ** (self.deg - 1)

    def d2r(self, r):
        return self.deg * (self.deg - 1) * r ** (self.deg - 2)

    def dr_div_r(self, r):
        return self.deg * r ** (self.deg - 2)

    def __repr__(self):
        return f"r**{self.deg}"


class EvenPHS(RBF):
    def __init__(self, deg: int):
        assert type(deg) is int
        assert deg > 0
        self.deg = deg
        self.even = deg % 2 == 0

    def __call__(self, r):
        # account for removeable singularity at r = 0
        ret = np.empty(r.shape)
        mask = np.abs(r) < 1e-15
        ret[mask] = 0
        ret[~mask] = r[~mask] ** self.deg * np.log(r[~mask])
        return ret

    def dr(self, r):
        raise NotImplementedError

    def d2r(r):
        raise NotImplementedError

    def dr_div_r(self, r):
        raise NotImplementedError

    def __repr__(self):
        return f"r**{self.deg}*log(r)"


def PHS(deg: int) -> RBF:
    if deg % 2 == 0:
        return EvenPHS(deg)
    return OddPHS(deg)


class Gaussian(RBF):
    def __init__(self, shape: float):
        self.shape = shape

    def __call__(self, r):
        return np.exp(-(r**2) / self.shape)

    def dr(r):
        raise NotImplementedError

    def d2r(r):
        raise NotImplementedError

    def dr_div_r(r):
        raise NotImplementedError

    def __repr__(self):
        return f"Gaussian(shape={self.shape})"
