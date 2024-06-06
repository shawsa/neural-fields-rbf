"""
A driver testing 2D interpolation on scattered nodes.
"""
import matplotlib.pyplot as plt
import numpy as np

from rbf.rbf import PHS, Gaussian
from rbf.points import UnitSquare
from rbf.interpolate import Interpolator
from rbf.stencil import Stencil

figsize = (5, 4)

FILE_FUNC = "presentation_interp_func.png"
FILE_POINTS = "presentation_interp_points.png"
FILE_BASIS_PREFIX = "presentation_interp_basis_"
FILE_INTERP = "presentation_interp_interp.png"
FILE_ERROR = "presentation_interp_error.png"


def Frankes_function(x, y):
    """A common function to test multidimensional interpolation."""
    return (
        0.75 * np.exp(-1 / 4 * ((9 * x - 2) ** 2 + (9 * y - 2) ** 2))
        + 0.75 * np.exp(-1 / 49 * ((9 * x + 12) ** 2 + (9 * y + 1) ** 2))
        + 0.5 * np.exp(-1 / 4 * ((9 * x - 7) ** 2 + (9 * y - 3) ** 2))
        + 0.2 * np.exp(-((9 * x - 4) ** 2 + (9 * y - 7) ** 2))
    )


test_func = Frankes_function
xs_dense, ys_dense = np.meshgrid(np.linspace(0, 1, 401), np.linspace(0, 1, 401))
points_dense = np.block([[xs_dense.ravel()], [ys_dense.ravel()]]).T
fs_dense = Frankes_function(xs_dense, ys_dense)

N = 300

np.random.seed(0)
points = UnitSquare(N, verbose=True).points
fs = Frankes_function(*points.T)

# rbf = PHS(7)
rbf = Gaussian(shape=0.1)
poly_deg = 4
stencil = Stencil(points=points)
approx = Interpolator(stencil=stencil, fs=fs, rbf=rbf)

errors = (
    approx(np.array([xs_dense.ravel(), ys_dense.ravel()]).T).reshape(xs_dense.shape)
    - fs_dense
)
print(f"max error: {np.max(np.abs(errors)):.4E}")


fig = plt.figure(figsize=figsize)
plt.pcolormesh(xs_dense, ys_dense, fs_dense, cmap="jet")
plt.colorbar()
plt.axis("off")
plt.savefig(FILE_FUNC)

fig = plt.figure(figsize=figsize)
plt.plot(*points.T, "g.")
plt.pcolormesh(xs_dense, ys_dense, fs_dense, cmap="jet")
plt.colorbar()
plt.axis("off")
plt.savefig(FILE_POINTS)

point_indices = [0, 10, 25, 30]

for file_index, point_index in enumerate(point_indices):
    x, y = points[point_index]
    fig = plt.figure(figsize=figsize)
    plt.plot(*points.T, "g.")
    plt.plot(*points[point_index], "c*", markersize=10)
    plt.pcolormesh(
        xs_dense,
        ys_dense,
        rbf(np.sqrt(((xs_dense - x) ** 2 + (ys_dense - y) ** 2))),
        cmap="jet",
    )
    plt.colorbar()
    plt.axis("off")
    plt.savefig(FILE_BASIS_PREFIX + str(file_index) + ".png")

fig = plt.figure(figsize=figsize)
plt.pcolormesh(
    xs_dense, ys_dense, approx(points_dense).reshape(xs_dense.shape), cmap="jet"
)
plt.plot(*points.T, "g.")
plt.colorbar()
plt.axis("off")
plt.savefig(FILE_INTERP)

fig = plt.figure(figsize=figsize)
plt.pcolormesh(xs_dense, ys_dense, errors, cmap="jet")
plt.plot(*points.T, "g.")
plt.colorbar()
plt.axis("off")
plt.savefig(FILE_ERROR)
