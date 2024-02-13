from dataclasses import dataclass
from itertools import pairwise, starmap
import numpy as np
import numpy.linalg as la
from .poly_utils import Monomial
from scipy.integrate import fixed_quad
from scipy.spatial import Delaunay
from typing import Callable


"""Need to unify poly integrate and rbf integrate classes."""

# rbf stuff


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


def get_right_triangles(A, B, C, O):
    triangles = []
    for X, Y in pairwise([A, B, C, A]):
        Z = orthogonal_projection(O, X, Y)
        triangles.append([O, X, Z])
        triangles.append([O, Z, Y])
    return triangles


def integrate_triangle(
    O, A, B, C, right_triangle_integrate: Callable[[float, float], float]
):
    """Integrate the function phi(|X - O|) over the triangular patch ABC"""
    area = 0
    for X, Y in pairwise([A, B, C, A]):
        Z = orthogonal_projection(O, X, Y)
        a = la.norm(O - Z)
        b = la.norm(X - Z)
        # analytically integrate constant
        area += area_sign(O, X, Z) * right_triangle_integrate(a, b)
        b = la.norm(Y - Z)
        area += area_sign(O, Z, Y) * right_triangle_integrate(a, b)
    area *= area_sign(A, B, C)
    return area


# poly stuff


@dataclass
class Point:
    x: float
    y: float

    def __iter__(self):
        yield self.x
        yield self.y

    def __len__(self):
        return 2

    def __add__(self, point2):
        if isinstance(point2, Point):
            return Point(*(self.arr + point2.arr))
        return Point(*(self.arr + point2))

    def __mul__(self, scalar):
        return Point(scalar * self.x, scalar * self.y)

    def __rmul__(self, scalar):
        return self * scalar

    def __sub__(self, point2):
        return self + (-1 * point2)

    def __truediv__(self, scalar):
        return self * (1 / scalar)

    def __radd__(self, point2):
        return point2 + self

    def __rmul(self, scalar):
        return self * scalar

    @property
    def arr(self) -> np.ndarray[float]:
        return np.array([self.x, self.y])


def parameterize(A: Point, B: Point, t_start=-1, t_stop=1):
    x_slope = (B.x - A.x) / (t_stop - t_start)
    y_slope = (B.y - A.y) / (t_stop - t_start)

    def path(t: float) -> Point:
        return Point(A.x + (t - t_start) * x_slope, A.y + (t - t_start) * y_slope)

    return path, y_slope


@dataclass
class Triangle:
    A: Point
    B: Point
    C: Point

    def __iter__(self):
        yield self.A
        yield self.B
        yield self.C

    @property
    def points(self):
        return np.array([point.arr for point in self])

    @property
    def plot_points(self):
        return np.array([point.arr for point in self] + [self.A.arr])

    def __add__(self, point2):
        return Triangle(*(point + point2 for point in self))

    def __radd__(self, point2):
        return point2 + self

    def __mul__(self, scalar):
        return Triangle(*(point * scalar for point in self))

    def __rmul(self, scalar):
        return self * scalar

    def __sub__(self, point2):
        return self + (-1 * point2)

    def __div__(self, scalar):
        return self * 1 / scalar

    def __truediv__(self, scalar):
        return self * (1 / scalar)

    @property
    def centroid(self):
        return (1 / 3) * np.array(
            [self.A.x + self.B.x + self.C.x, self.A.y + self.B.y + self.C.y]
        )

    @property
    def edges(self):
        return ((self.A, self.B), (self.B, self.C), (self.C, self.A))

    @property
    def sign(self):
        """+1, or -1 based on orientation"""
        return np.sign(
            np.dot(
                np.array([self.A.y - self.B.y, self.B.x - self.A.x]),
                np.array([self.C.x - self.A.x, self.C.y - self.A.y]),
            )
        )

    def boundary_quad(self, func: Callable, order=10):
        t_start = -1
        t_stop = 1
        total = 0
        for A, B in self.edges:
            path, y_slope = parameterize(A, B)
            total += (
                y_slope
                * fixed_quad(lambda t: func(*path(t)), t_start, t_stop, n=order)[0]
            )
        return self.sign * total

    def poly_quad(self, poly: Monomial, order: int = 10):
        func = poly.adiff_x()
        return self.boundary_quad(func, order=order)

    def rbf_quad(
        self,
        rbf_center: np.ndarray[float],
        right_triangle_integrate: Callable[[float, float], float],
    ) -> float:
        return integrate_triangle(
            rbf_center, self.A.arr, self.B.arr, self.C.arr, right_triangle_integrate
        )


def triangle(mat: np.ndarray[float]) -> Triangle:
    assert mat.ndim == 2
    assert mat.shape == (3, 2)
    return Triangle(*starmap(Point, mat))


def circumradius(points: np.ndarray[float]) -> float:
    a = la.norm(points[0] - points[1])
    b = la.norm(points[1] - points[2])
    c = la.norm(points[2] - points[0])
    return (
        a * b * c / np.sqrt((a + b + c) * (b + c - a) * (c + a - b) * (a + b - c))
    )


def delaunay_covering_radius(mesh: Delaunay):
    h = 0
    for mesh_indices in mesh.simplices:
        tri = mesh.points[mesh_indices]
        h = max(h, circumradius(tri))
    return h
