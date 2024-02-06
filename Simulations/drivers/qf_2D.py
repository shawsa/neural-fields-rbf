import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from rbf.geometry import Triangle, triangle
from rbf.points import UnitSquare
from rbf.poly_utils import Monomial, poly_powers_gen
from rbf.quad_lib import get_right_triangle_integral_function
from rbf.rbf import RBF, PHS
from rbf.stencil import Stencil
from scipy.integrate import fixed_quad
from scipy.spatial import Delaunay
from typing import Callable


"""
Bugs:
    Only works for stencil center = 0, 0
    scaling seems to work, but may be confounding the problem
"""


class QuadStencil(Stencil):
    def __init__(self, points: np.ndarray[float], element: Triangle):
        super(QuadStencil, self).__init__(points, center=np.array([0.0, 0.0]))
        self.element = element

    def weights(self, rbf: RBF, poly_deg: int):
        right_triangle_integrate = get_right_triangle_integral_function(rbf)
        mat = self.interpolation_matrix(rbf, poly_deg)
        rhs = np.zeros_like(mat[0])
        rhs[: len(self.points)] = np.array(
            [
                self.element.rbf_quad(point, right_triangle_integrate)
                for point in self.points
            ]
        )

        rhs[len(self.points) :] = np.array(
            [
                self.element.poly_quad(poly)
                for poly in poly_powers_gen(self.dim, poly_deg)
            ]
        )
        weights = la.solve(mat, rhs)
        # return weights[: len(self.points)] / self.scaling**2
        return mat, rhs, weights[: len(self.points)] / self.scaling**2


if __name__ == "__main__":
    n = 11
    points = UnitSquare(n, verbose=True).points
    # X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
    # points = np.array([X.ravel(), Y.ravel()]).T
    mesh = Delaunay(points)
    rbf = PHS(7)
    poly_deg = 2

    total = 0
    weights_list = []
    for tri_indices in mesh.simplices:
        tri = triangle(mesh.points[tri_indices])
        stencil = QuadStencil(points, tri)
        stencil.scaling = 1.0
        # total += stencil.weights(rbf, poly_deg)
        mat, rhs, weights = stencil.weights(rbf, poly_deg)
        weights_list.append(weights)
        total += weights

    plt.triplot(*points.T, mesh.simplices)
    plt.plot(*points.T, "k.")
    for w, (x, y) in zip(total, points):
        plt.text(x, y, f"{w:.2}")
    plt.title("Quad Weights For Scattered Nodes")

    xs, ys = points.T
    print(sum(total) - 1)
    print(total @ xs - 0.5)
    print(total @ ys - 0.5)
    print(total @ xs**2 - 1 / 3)
    print(total @ (xs * ys) - 1 / 4)
    print(total @ ys**2 - 1 / 3)

    import sympy as sym
    from sympy.abc import x, y

    for foo_sym in [
            1,
            x,
            y,
            x**2,
            x*y,
            y**2,
            sym.sin(x) * sym.sin(y),
            sym.cos(x) * sym.cos(y),
            sym.cos(x)**2 - y*sym.exp(y),
    ]:
        if foo_sym == 1:
            exact = 1
            approx = sum(total)
        else:
            foo = sym.lambdify((x, y), foo_sym)
            exact = float(sym.integrate(sym.integrate(foo_sym, (x, 0, 1)), (y, 0, 1)))
            approx = total @ foo(xs, ys)
        print("Function: " + sym.latex(foo_sym))
        print(f"Relative Error: {abs(exact-approx)/abs(exact)}\n")
