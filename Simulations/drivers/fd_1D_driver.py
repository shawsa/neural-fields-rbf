import set_path

import matplotlib.pyplot as plt
import numpy as np
from rbf import RBF, PHS
from rbf.finite_differences import Derivative1D
from rbf.linear_functional import FunctionalStencil, LinearFunctional
from rbf.poly_utils import Monomial


def foo(x):
    return np.sin(x)


def d_foo(x):
    return np.cos(x)


rbf = PHS(3)
poly_deg = 4
k = 5
hs = [2 ** (-i) for i in range(20)]
assert k % 2 == 1

print(f"1st derivative evaluated at 0: {d_foo(0)}")
print("Stencil:")
unshifted_points = np.concatenate([np.arange(-k // 2 + 1, 0), np.arange((k + 1) // 2)])
print(unshifted_points)

errors = []
for h in hs:
    points = unshifted_points * h
    stencil = FunctionalStencil(points)
    weights = stencil.weights(rbf, Derivative1D(), poly_deg)
    error = abs(weights @ foo(points) - d_foo(0))
    errors.append(error)

plt.loglog(hs, errors, 'k.-')
plt.xlabel("h")
plt.ylabel("Error")
plt.title(f"Finite Difference Test:\nrbf: {rbf}, poly degree: {poly_deg}, stencil size: {k}")

start = 1
end = 1e-3

for order in range(1, 5):
    plt.plot([start, end], [errors[0], end**order], label=f"$\\mathcal{{O}}({order})$")

plt.legend()
plt.show()
