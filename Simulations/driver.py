from functools import partial
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import numpy.linalg as la

from rbf.rbf import PHS
from rbf.points import UnitSquare
from rbf.interpolate import interpolate, LocalInterpolator

from scipy.special import roots_chebyt as cheb
from tqdm import tqdm


########################
# two test functions
########################
def Frankes_function(x, y):
    """A common function to test multidimensional interpolation."""
    return (
        0.75 * np.exp(-1 / 4 * ((9 * x - 2) ** 2 + (9 * y - 2) ** 2))
        + 0.75 * np.exp(-1 / 49 * ((9 * x + 12) ** 2 + (9 * y + 1) ** 2))
        + 0.5 * np.exp(-1 / 4 * ((9 * x - 7) ** 2 + (9 * y - 3) ** 2))
        + 0.2 * np.exp(-((9 * x - 4) ** 2 + (9 * y - 7) ** 2))
    )


def poly_test(x, y):
    """A polynomial test function."""
    return 4 - 3 * x + 2 * y - x**2 + x * y - y**2


# test_func = poly_test
test_func = Frankes_function


########################
# Dense sample points
########################
xs_dense, ys_dense = np.meshgrid(np.linspace(0, 1, 401), np.linspace(0, 1, 401))
points_dense = np.block([[xs_dense.ravel()], [ys_dense.ravel()]]).T
fs_dense = test_func(xs_dense, ys_dense)

########################
# RBF sample points
########################

xs, ys = UnitSquare(21, verbose=True).points.T

# Cartesian points
# xs, ys = np.meshgrid(
#     np.linspace(np.min(xs_dense), np.max(xs_dense), 11),
#     np.linspace(np.min(ys_dense), np.max(ys_dense), 11),
# )


# Chebshev points
# def cheb_nodes_1D(n, x_min, x_max):
#     points = (cheb(n)[0] + 1)/2
#     points = (x_max - x_min)*points + x_min
#     return points
#
#
# xs, ys = np.meshgrid(
#         cheb_nodes_1D(15, np.min(xs_dense), np.max(xs_dense)),
#         cheb_nodes_1D(15, np.min(ys_dense), np.max(ys_dense))
# )

# circle for singular testing
# angles = np.linspace(-np.pi, np.pi, 8, endpoint=False)
# xs = np.cos(angles)/2
# ys = np.sin(angles)/2

points = np.array([xs.ravel(), ys.ravel()]).T
# add extra points
# points = np.concatenate((points, [(0.45, 0.75), (0.45, 0.85), (0.55, 0.6)]))
# points = np.concatenate((points, [(0.45, 0.77)]))

fs = test_func(points[:, 0], points[:, 1])

########################
# interpolate
########################
rbf = PHS(3)
poly_deg = 3
stencil_size = 14
approx = LocalInterpolator(
    points=points, fs=fs, rbf=rbf, poly_deg=poly_deg, stencil_size=stencil_size
)
# approx = interpolate(points=points, fs=fs, rbf=rbf, poly_deg=poly_deg)
# approx = interpolate(points=points, fs=fs)

########################
# measure errors
########################
eval_point = np.array((0.21, 0.23))
print(f"eval point = {eval_point}")
print(f"Function value = {test_func(*eval_point)}")
print(f"Interpolant value = {approx(eval_point)}")
print(f"Interpolation error = {approx(eval_point) - test_func(*eval_point): .3E}")

errors = np.empty(ys_dense.shape)
for index, (x, y) in tqdm(
    enumerate(zip(xs_dense.ravel(), ys_dense.ravel())), total=len(xs_dense.ravel())
):
    errors.ravel()[index] = abs(approx(np.array((x, y))) - test_func(x, y))

print(f"max error over domain: {np.max(errors): .3E}")

plt.close()

fig = plt.figure("RBF interpolation error", figsize=(8, 4))
grid = matplotlib.gridspec.GridSpec(1, 7)
ax_function = fig.add_subplot(grid[0, :3])
ax_error = fig.add_subplot(grid[0, 3:6])
ax_color_bar = fig.add_subplot(grid[0, 6])

# plot function and points
ax_function.set_title("Test Function")
ax_function.pcolormesh(xs_dense, ys_dense, fs_dense)
ax_function.plot(points[:, 0], points[:, 1], "r.")

ax_error.set_title("Error")
ax_error.pcolormesh(xs_dense, ys_dense, errors)
error_color_norm = matplotlib.cm.ScalarMappable(
    matplotlib.colors.Normalize(vmin=0, vmax=np.max(errors))
)
plt.colorbar(error_color_norm, cax=ax_color_bar)
ax_error.plot(points[:, 0], points[:, 1], "r.")
plt.tight_layout()
plt.show()
