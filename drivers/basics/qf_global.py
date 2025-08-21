import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import Delaunay
import sympy as sym

from neural_fields_rbf.rbf.geometry import triangle
from neural_fields_rbf.points import UnitSquare
from neural_fields_rbf.rbf.quadrature import QuadStencil
from neural_fields_rbf.rbf import PHS

n = 7
points = UnitSquare(n**2, verbose=True).points
# X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
# points = np.array([X.ravel(), Y.ravel()]).T
rbf = PHS(3)
poly_deg = 2

mesh = Delaunay(points)
total = 0
weights_list = []
for tri_indices in mesh.simplices:
    tri = triangle(mesh.points[tri_indices])
    stencil = QuadStencil(points, tri)
    total += stencil.weights(rbf, poly_deg)

plt.figure()
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

x, y = sym.symbols("x y")

for foo_sym in [
    1,
    x,
    y,
    x**2,
    x * y,
    y**2,
    sym.sin(x) * sym.sin(y),
    sym.cos(x) * sym.cos(y),
    sym.cos(x) ** 2 - y * sym.exp(y),
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
