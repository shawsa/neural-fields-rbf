from math import ceil, sqrt
import matplotlib.pyplot as plt
import numpy as np
from rbf.points import UnitSquare
from rbf.quadrature import LocalQuad
from rbf.rbf import PHS
import sympy as sym

n = ceil(sqrt(2000))
points = UnitSquare(n, verbose=True).points
# X, Y = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
# points = np.array([X.ravel(), Y.ravel()]).T
rbf = PHS(3)
poly_deg = 3
stencil_size = 20

qf = LocalQuad(points, rbf, poly_deg, stencil_size)

plt.figure()
plt.triplot(*points.T, qf.mesh.simplices)
plt.plot(*points.T, "k.")
for w, (x, y) in zip(qf.weights, points):
    plt.text(x, y, f"{w:.2}")
plt.title("Quad Weights For Scattered Nodes")

print("Weight Stats")
print(f"Min: {np.min(qf.weights):.3E}")
print(f"Ave: {np.mean(qf.weights):.3E}")
print(f"Max: {np.max(qf.weights):.3E}")
print(f"Negative weight count: {np.sum(qf.weights < 0)}")
print(qf.weights[qf.weights < 0])

xs, ys = points.T
print(sum(qf.weights) - 1)
print(qf.weights @ xs - 0.5)
print(qf.weights @ ys - 0.5)
print(qf.weights @ xs**2 - 1 / 3)
print(qf.weights @ (xs * ys) - 1 / 4)
print(qf.weights @ ys**2 - 1 / 3)

x, y = sym.symbols("x y")

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
        approx = sum(qf.weights)
    else:
        foo = sym.lambdify((x, y), foo_sym)
        exact = float(sym.integrate(sym.integrate(foo_sym, (x, 0, 1)), (y, 0, 1)))
        approx = qf.weights @ foo(xs, ys)
    print("Function: " + sym.latex(foo_sym))
    print(f"Relative Error: {abs(exact-approx)/abs(exact)}\n")
