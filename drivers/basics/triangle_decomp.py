"""
I have concluded that the approach in Sommariva and Vianello (2006) is
too difficult to generalize even with Sympy.

Look into Reeger and Fornberg (2016) instead. They seem to have
addressed the issue.
"""

from rbf.poly_utils import Monomial
from itertools import pairwise
import matplotlib.pyplot as plt
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


def get_right_triangles(A, B, C, O):
    triangles = []
    for X, Y in pairwise([A, B, C, A]):
        Z = orthogonal_projection(O, X, Y)
        triangles.append([O, X, Z])
        triangles.append([O, Z, Y])
    return triangles


# visualize triangles

A = np.array((1, 1), dtype=float)
B = np.array((5, 2), dtype=float)
C = np.array((0, 7), dtype=float)

O = np.array((3, 5), dtype=float)
# O = np.array((2, 3), dtype=float)
# O = np.array((7, 3), dtype=float)

D = orthogonal_projection(O, A, B)
E = orthogonal_projection(O, B, C)
F = orthogonal_projection(O, C, A)

fig, axes = plt.subplots(1, 2, figsize=(20, 10))
for ax in axes:
    xs, ys = map(list, zip(A, B, C, A))
    ax.plot(xs, ys, "k-")
    for X in [O, D, E, F]:
        ax.plot(*X, "k.")
    ax.axis("square")

triangles = get_right_triangles(A, B, C, O)
poly_kwargs = {
    "edgecolor": "k",
    "linestyle": "-",
    "alpha": 0.5,
}
for triangle in triangles:
    if area_sign(*triangle) > 0:
        axes[0].add_patch(plt.Polygon(np.array(triangle), facecolor="b", **poly_kwargs))
    else:
        axes[1].add_patch(plt.Polygon(np.array(triangle), facecolor="r", **poly_kwargs))


# test on constant function
def true_area(A, B, C):
    return abs(la.det(np.array([B - A, C - A]))) / 2


def area_test(O, A, B, C):
    area = 0
    for X, Y in pairwise([A, B, C, A]):
        Z = orthogonal_projection(O, X, Y)
        a = la.norm(O - Z)
        b = la.norm(X - Z)
        # analytically integrate constant
        area += area_sign(O, X, Z) * 0.5 * a * b
        b = la.norm(Y - Z)
        area += area_sign(O, Z, Y) * 0.5 * a * b
    area *= area_sign(A, B, C)
    return area


for _ in range(100_000):
    O, A, B, C = np.random.rand(4, 2)
    assert abs(true_area(A, B, C) - area_test(O, A, B, C)) < 1e-15


# polynomial testing
def line(A, B, x):
    return A[1] + (x - A[0])*(B[1] - A[1])/(B[0] - A[0])


def poly_int(A, B, C, poly: Monomial):
    L, M, R = sorted([A, B, C], key=lambda point: point[0])
