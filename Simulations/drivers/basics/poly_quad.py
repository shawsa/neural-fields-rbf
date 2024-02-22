"""Test code for integrating polynomials over triangles."""

from dataclasses import dataclass
from itertools import starmap
import matplotlib.pyplot as plt
import numpy as np
from typing import Callable
from scipy.integrate import fixed_quad

from scipy.spatial import Delaunay

from sympy.abc import x, y
import sympy as sym


@dataclass
class Point:
    x: float
    y: float

    def __iter__(self):
        yield self.x
        yield self.y

    def __len__(self):
        return 2


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

    @property
    def edges(self):
        return ((self.A, self.B), (self.B, self.C), (self.C, self.A))

    @property
    def sign(self):
        """+1, or -1 based on orientation"""
        return np.sign(np.dot(np.array([self.A.y - self.B.y, self.B.x - self.A.x]),
                              np.array([self.C.x - self.A.x, self.C.y - self.A.y])))

    def boundary_integrate(self, func: Callable, order=10):
        t_start = -1
        t_stop = 1
        total = 0
        for A, B in self.edges:
            path, y_slope = parameterize(A, B)
            total += (
                y_slope
                * fixed_quad(lambda t: func(*path(t)), t_start, t_stop, n=order)[0]
            )
        return self.sign*total


expr = x**5 * y**7

foo = sym.lambdify((x, y), sym.integrate(expr, x))

exact = sym.integrate(sym.integrate(expr, (x, 0, 1)), (y, 0, 1))


np.random.seed(0)
corners = np.array(
    [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
    ],
    dtype=float)
points = np.random.rand(20, 2)
points = np.concatenate((points, corners))
mesh = Delaunay(points)
plt.triplot(*points.T, mesh.simplices)

total = 0
for simplex_indices in mesh.simplices:
    points = starmap(Point, mesh.points[simplex_indices])
    tri = Triangle(*points)
    total += tri.boundary_integrate(foo)

print(f"{float(exact)=}, \t{total=}")
print(f"error = {abs((total - exact)/exact)}")
