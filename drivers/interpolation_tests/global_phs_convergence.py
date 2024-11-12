from collections import namedtuple
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from scipy.spatial import Delaunay
from scipy.stats import linregress
from tqdm import tqdm

from min_energy_points import UnitSquare
from rbf.geometry import delaunay_covering_radius_stats
from rbf.interpolate import Interpolator, Stencil
from rbf.rbf import PHS

dim = 2

rbf_order = 6

poly_deg = (rbf_order + dim) // 2
rbf = PHS(rbf_order)


def foo(x, y):
    x0, y0 = 0.5, 0.5
    return np.exp(-((x - x0) ** 2) - (y - y0) ** 2)


def trap_int(Z):
    hx, hy = [1 / (n - 1) for n in Z.shape]
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


def L1(Z):
    return trap_int(np.abs(Z))


def L2(Z):
    return np.sqrt(trap_int(Z**2))


def Loo(Z):
    return np.max(np.abs(Z))


def predicted_order(rbf_order, dim, p):
    k = (rbf_order + dim) / 2
    n = dim
    if p == "oo":
        return 2 * k - n / 2
    return 2 * k + min(n / p - n / 2, 0)


# interp test
N = 200
points = UnitSquare(N, verbose=True).points
mesh = Delaunay(points)
h, _ = delaunay_covering_radius_stats(mesh)


approx = Interpolator(
    stencil=Stencil(points),
    fs=foo(*points.T),
    rbf=rbf,
    poly_deg=poly_deg,
)

X, Y = np.meshgrid(*2 * (np.linspace(0, 1, 400),))
Fs = foo(X, Y)
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
    ),
)

Ns = list(map(int, np.logspace(np.log10(5_000), np.log10(1_000), 5)))
repeats = 5

X, Y = np.meshgrid(*2 * (np.linspace(0.1, 0.9, 400),))
Fs = foo(X, Y)

results = []
for N in (tqdm_obj := tqdm(Ns, position=0)):
    for _ in tqdm(range(repeats), position=1, leave=False):
        points = UnitSquare(N=N).points
        mesh = Delaunay(points)
        h, _ = delaunay_covering_radius_stats(mesh)

        approx = Interpolator(
            stencil=Stencil(points),
            fs=foo(*points.T),
            rbf=rbf,
            poly_deg=poly_deg,
        )
        errors = Fs - approx(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
        Loo_err = Loo(errors)
        tqdm_obj.set_description(f"N={N}, max_err={Loo_err}")
        results.append(
            Result(
                rbf=rbf,
                poly_deg=poly_deg,
                N=N,
                h=h,
                L1=L1(errors),
                L2=L2(errors),
                Loo=Loo_err,
            )
        )

hs = [res.h for res in results]
min_index, min_h = min(enumerate(hs), key=lambda pair: pair[1])
max_index, max_h = max(enumerate(hs), key=lambda pair: pair[1])

fig = plt.figure(figsize=(30, 10))
grid = gridspec.GridSpec(1, 3)
ax_1 = fig.add_subplot(grid[0, 0])
ax_2 = fig.add_subplot(grid[0, 1])
ax_oo = fig.add_subplot(grid[0, 2])

for ax, title, errs, p in [
    (ax_1, "L1 errors", [res.L1 for res in results], 1),
    (ax_2, "L2 errors", [res.L2 for res in results], 2),
    (ax_oo, "max errors", [res.Loo for res in results], "oo"),
]:
    errs = [res.L1 for res in results]
    ax.loglog(hs, errs, ".")
    theory = predicted_order(rbf_order=rbf_order, dim=dim, p=p)
    lin = linregress(np.log(hs), np.log(errs))
    ax.plot(
        [min_h, max_h],
        [errs[min_index], errs[min_index] * (max_h / min_h) ** theory],
        "k:",
        label=f"theory - $\\mathcal{{O}}({theory})$",
    )
    ax.plot(
        [min_h, max_h],
        [np.exp(lin.intercept) * h**lin.slope for h in [min_h, max_h]],
        "k-",
        label=f"observed - $\\mathcal{{O}}({lin.slope:.2})$",
    )
    ax.set_xlabel("$h$")
    ax.set_title(title)
    ax.legend()

plt.suptitle(f"rbf={rbf}, poly_deg={poly_deg}")
plt.savefig(f"media/rbf={rbf}_poly_deg={poly_deg}.png")
