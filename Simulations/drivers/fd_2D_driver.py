import set_path

from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from rbf import RBF, PHS
from rbf.finite_differences import Laplacian
from rbf.linear_functional import FunctionalStencil, LinearFunctional
from rbf.poly_utils import Monomial
import sympy as sym

x_sym, y_sym = sym.symbols("x y")

sym_foo = sym.cos(x_sym-1)*sym.cos(y_sym-3)


foo = sym.lambdify((x_sym, y_sym), sym_foo)
L_foo = sym.lambdify((x_sym, y_sym), sym_foo.diff(x_sym, 2) + sym_foo.diff(y_sym, 2))


rbf = PHS(3)
poly_deg = 2

stencil_width = 3


hs = [2**-i for i in range(2, 20)]

assert stencil_width % 2 == 1
unshifted_points = list(
    product(*2 * (np.arange(-(stencil_width-1) // 2, (stencil_width+1) // 2),))
)
unshifted_points.sort(key=lambda point: la.norm(point))
unshifted_points = np.array(unshifted_points, dtype=float)
# unshifted_points += np.random.rand(*unshifted_points.shape)*1e-1
# unshifted_points -= unshifted_points[0]

print(f"Laplacian evaluated at 0, 0: {L_foo(0, 0)}")
print("Stencil:")
print(unshifted_points)

stencil = FunctionalStencil(unshifted_points)
weights = stencil.weights(rbf, Laplacian(dim=2), poly_deg)
plt.figure("Stencil")
for w, point in zip(weights, unshifted_points):
    plt.plot(*point, 'k.')
    plt.text(*point, str(np.round(w, 4)))
plt.xlim(-stencil_width / 1.9, stencil_width/1.9)
plt.ylim(-stencil_width / 1.9, stencil_width/1.9)
plt.title("Stencil and Weights")
plt.show()

errors = []
for h in hs:
    points = unshifted_points * h
    stencil = FunctionalStencil(points)
    weights = stencil.weights(rbf, Laplacian(dim=2), poly_deg)
    test = weights @ foo(*points.T)
    error = abs(test - L_foo(0, 0))
    print(f"approx = {test:.5f} \terror = {error:.5g}")
    errors.append(error)

plt.figure("Error")
plt.loglog(hs, errors, "k.-")
plt.xlabel("h")
plt.ylabel("Error")
plt.title(f"Finite Difference Test:\nrbf: {rbf}, poly degree: {poly_deg}")

start = hs[0]
end = 1e-3

for order in range(1, 5):
    plt.plot(
        [start, end], [errors[0], end**order], label=f"$\\mathcal{{O}}({order})$"
    )

plt.legend()
plt.show()
