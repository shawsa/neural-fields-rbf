
from .rbf import RBF
from .linear_functional import LinearFunctional
from .poly_utils import Monomial
from rbf.quad_lib import get_right_triangle_integral_function

from itertools import pairwise
import numpy as np
import numpy.linalg as la


def orthogonal_projection(O, A, B):
    """Project the point O onto the line through A and B"""
    rel_O = O - A
    rel_B = B - A
    proj = np.dot(rel_O, rel_B) / np.dot(rel_B, rel_B) * rel_B
    return proj + A


def area_sign(A, B, C):
    arr1 = np.array([A[1] - B[1], B[0] - A[0]])
    arr2 = np.array([C[0] - A[0], C[1] - A[1]])
    return np.sign(np.dot(arr1, arr2))


# This is a linear functional, but the current call signatures of the
# linear functional interface are incompatable
class TriangleQuad:
    def __init__(self, A, B, C, rbf: RBF):
        self.A = A
        self.B = B
        self.C = C
        self.func = get_right_triangle_integral_function(rbf)

    @property
    def scale_power(self) -> np.ndarray[int]:
        return 1

    def rbf_op(self, rbf_center: np.ndarray) -> float:
        area = 0
        for X, Y in pairwise([self.A, self.B, self.C, self.A]):
            Z = orthogonal_projection(rbf_center, X, Y)
            a = la.norm(rbf_center - Z)
            b = la.norm(X - Z)
            # analytically integrate constant
            area += area_sign(rbf_center, X, Z) * self.func(a, b)
            b = la.norm(Y - Z)
            area += area_sign(rbf_center, Z, Y) * self.func(a, b)
        area *= area_sign(self.A, self.B, self.C)
        return area

    def poly_op(self, poly: Monomial) -> float:
        ...
