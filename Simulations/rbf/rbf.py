"""
Classes for radial basis functions.
"""

from abc import ABC, abstractmethod, abstractstaticmethod
import numpy as np
import numpy.linalg as la


def pairwise_diff(p1: np.ndarray[float], p2: np.ndarray[float]) -> np.ndarray[float]:
    """Return a matrix of the poirwise differences between two
    vectors of points.
    The points must be of the same dimension.
    """
    return np.array([[(x - y) for x in p1] for y in p2])


def pairwise_dist(p1: np.ndarray[float], p2: np.ndarray[float]) -> np.ndarray[float]:
    """Return a matrix of the pairwise distances between two
    vectors of points.
    The points must be of the same dimension.
    """
    return np.array([[la.norm(x - y) for x in p1] for y in p2])


class RBF(ABC):
    @abstractmethod
    def __call__(r):
        raise NotImplementedError

    @abstractmethod
    def dr(r):
        raise NotImplementedError

    @abstractmethod
    def dr_div_r(r):
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

    def dr_div_r(self, r):
        return self.deg * r ** (self.deg - 2)


class EvenPHS(RBF):
    def __init__(self, deg: int):
        assert type(deg) is int
        assert deg > 0
        self.deg = deg
        self.even = deg % 2 == 0

    def __call__(self, r):
        # account for removeable singularity at r = 0
        ret = np.empty(r.shape)
        mask = ret == 0
        ret[mask] = 0
        ret[~mask] = r[~mask]**self.deg * np.log(r[~mask])
        return ret

    def dr(self, r):
        raise NotImplementedError


def PHS(deg: int) -> RBF:
    if deg % 2 == 0:
        return EvenPHS(deg)
    return OddPHS(deg)
