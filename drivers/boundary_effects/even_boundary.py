from collections import namedtuple
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from scipy.spatial import Delaunay
from scipy.stats import linregress
from tqdm import tqdm

from min_energy_points import UnitSquare
from rbf.geometry import delaunay_covering_radius_stats

# from rbf.points.utils import poly_stencil_min, hex_stencil_min
from rbf.points.utils import get_stencil_size
from rbf.interpolate import LocalInterpolator
from rbf.rbf import PHS


DIM = 2


def foo_even(x, y):
    return np.cos(2 * x / np.pi) * np.cos(4 * y / np.pi) + 1


def foo_odd(x, y):
    return np.sin(2 * x / np.pi) * np.sin(4 * y / np.pi) + 1


def gaussian(x, y):
    return np.exp(-((x - 0.05) ** 2 + (y - 0.05) ** 2))


# foo, foo_str = foo_even, "Cosines"
# foo, foo_str = foo_odd, "Sines"
foo, foo_str = gaussian, "Gaussian"


def trap_int(Z, area=1):
    hx, hy = [area / (n - 1) for n in Z.shape]
    res = np.sum(Z[1:-1, 1:-1])
    res += np.sum(Z[0, 1:-1]) / 2
    res += np.sum(Z[1, 1:-1]) / 2
    res += np.sum(Z[1:-1, 0]) / 2
    res += np.sum(Z[1:-1, -1]) / 2
    res += Z[0, 0] / 4
    res += Z[-1, 0] / 4
    res += Z[0, -1] / 4
    res += Z[-1, -1] / 4
    return res * hx * hy


def L1(Z, area=1):
    return trap_int(np.abs(Z), area=area)


def L2(Z, area=1):
    return np.sqrt(trap_int(Z**2, area=area))


def Loo(Z):
    return np.max(np.abs(Z))


# interp test
N = 500
poly_deg = 2
PHS_PAIRITY = 0

points = UnitSquare(N, verbose=True).points
mesh = Delaunay(points)
h, _ = delaunay_covering_radius_stats(mesh)

rbf_order = (2 * (poly_deg + 1) - DIM) + PHS_PAIRITY
# stencil_size = int(1.1 * poly_stencil_min(deg=poly_deg))
stencil_size = get_stencil_size(poly_deg, stability_factor=1.2)
rbf = PHS(rbf_order)

approx = LocalInterpolator(
    points=points,
    fs=foo_even(*points.T),
    rbf=rbf,
    poly_deg=poly_deg,
    stencil_size=stencil_size,
)

X, Y = np.meshgrid(*2 * (np.linspace(0, 1, 400),))
Fs = foo_even(X, Y)
errors = Fs - approx(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
max_err = Loo(errors)
print(f"L1_err = {L1(errors):.3E}")
print(f"L2_err = {L2(errors):.3E}")
print(f"max_err = {max_err:.3E}")

plt.figure()
plt.pcolormesh(X, Y, errors, cmap="jet", vmin=-max_err, vmax=max_err)
plt.colorbar()
plt.plot(*points.T, "k.")
plt.axis("equal")

plt.figure()
plt.pcolormesh(X, Y, np.log10(np.abs(errors)), cmap="jet")
plt.colorbar()
plt.plot(*points.T, "k.")
plt.axis("equal")

# convergence
Result = namedtuple(
    "Result",
    (
        "rbf",
        "poly_deg",
        "N",
        "h",
        "L1",
        "L2",
        "Loo",
        "L1_inner",
        "L2_inner",
        "Loo_inner",
    ),
)

PHS_PAIRITY = 0
poly_degs = [1, 2, 3, 4]
Ns = list(map(int, np.logspace(np.log10(5_000), np.log10(1_000), 11)))
repeats = 5

# X, Y = np.meshgrid(*2 * (np.linspace(0.1, 0.9, 400),))
X, Y = np.meshgrid(*2 * (np.linspace(0, 1, 400),))
X_inner, Y_inner = np.meshgrid(*2 * (np.linspace(0.1, 0.9, 400),))
Fs = foo_even(X, Y)
Fs_inner = foo_even(X_inner, Y_inner)

results = []
np.random.seed(0)
for repeat in tqdm(range(repeats), position=0, leave=False):
    for N in (tqdm_obj := tqdm(Ns, position=1, leave=False)):
        for poly_deg in tqdm(poly_degs[::-1], position=2, leave=False):
            rbf_order = (2 * (poly_deg + 1) - DIM) + PHS_PAIRITY
            rbf = PHS(rbf_order)
            stencil_size = get_stencil_size(poly_deg, stability_factor=1.5)
            points = UnitSquare(N=N).points
            mesh = Delaunay(points)
            h, _ = delaunay_covering_radius_stats(mesh)

            approx = LocalInterpolator(
                points=points,
                fs=foo_even(*points.T),
                rbf=rbf,
                poly_deg=poly_deg,
                stencil_size=stencil_size,
            )

            errors = Fs - approx(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
            errors_inner = Fs_inner - approx(
                np.c_[X_inner.ravel(), Y_inner.ravel()]
            ).reshape(X_inner.shape)
            Loo_err = Loo(errors)
            tqdm_obj.set_description(f"N={N}, max_err={Loo_err}")
            results.append(
                Result(
                    rbf=str(rbf),
                    poly_deg=poly_deg,
                    N=N,
                    h=h,
                    L1=L1(errors),
                    L2=L2(errors),
                    Loo=Loo_err,
                    L1_inner=L1(errors_inner, area=0.64),
                    L2_inner=L2(errors_inner, area=0.64),
                    Loo_inner=Loo(errors_inner),
                )
            )

hs = [res.h for res in results]
min_index, min_h = min(enumerate(hs), key=lambda pair: pair[1])
max_index, max_h = max(enumerate(hs), key=lambda pair: pair[1])

fig = plt.figure(figsize=(30, 10))
grid = gridspec.GridSpec(2, 3)
ax_1 = fig.add_subplot(grid[0, 0])
ax_2 = fig.add_subplot(grid[0, 1])
ax_oo = fig.add_subplot(grid[0, 2])
ax_1_inner = fig.add_subplot(grid[1, 0])
ax_2_inner = fig.add_subplot(grid[1, 1])
ax_oo_inner = fig.add_subplot(grid[1, 2])


colors = ["r", "b", "g", "c", "m", "y"] * 2
for ax, title, p, full in [
    (ax_1, "L1 errors", 1, True),
    (ax_2, "L2 errors", 2, True),
    (ax_oo, "max errors", "oo", True),
    (ax_1_inner, "L1 errors", 1, False),
    (ax_2_inner, "L2 errors", 2, False),
    (ax_oo_inner, "max errors", "oo", False),
]:
    for poly_deg, color in zip(poly_degs, colors):
        my_res = [res for res in results if res.poly_deg == poly_deg]
        hs = [res.h for res in my_res]
        if p == 1:
            if full:
                ax.set_ylabel("Error")
                errs = [res.L1 for res in my_res]
            else:
                errs = [res.L1_inner for res in my_res]
                ax.set_ylabel("Inner Error")
        elif p == 2:
            if full:
                errs = [res.L2 for res in my_res]
            else:
                errs = [res.L2_inner for res in my_res]
        elif p == "oo":
            errs = [res.Loo for res in my_res]
            if full:
                errs = [res.Loo for res in my_res]
            else:
                errs = [res.Loo_inner for res in my_res]
        ax.loglog(hs, errs, ".", color=color)
        lin = linregress(np.log(hs), np.log(errs))
        ax.plot(
            [min_h, max_h],
            [np.exp(lin.intercept) * h**lin.slope for h in [min_h, max_h]],
            "-",
            color=color,
            label=f"rbf={my_res[0].rbf}, deg={poly_deg}, $\\mathcal{{O}}({lin.slope:.2})$",
        )
    ax.set_xlabel("$h$")
    ax.set_title(title)
    ax.legend()

plt.suptitle(foo_str)
plt.savefig(f"media/boundary_test_{foo_str}_pairty={PHS_PAIRITY}.png")
