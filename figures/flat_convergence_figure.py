import json
from matplotlib.colors import TABLEAU_COLORS
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import numpy as np
# from scipy.spatial import Delaunay
from scipy.stats import linregress
import sympy as sym
# from tqdm import tqdm
from types import SimpleNamespace

from flat_convergence.manufactured import ManufacturedSolutionPeriodic
# from odeiter import TimeDomain_Start_Stop_MaxSpacing, RK4
# from odeiter.adams_bashforth import AB5
# from rbf.interpolate import LocalInterpolator
# from rbf.quadrature import LocalQuad
# from rbf.points import UnitSquare
# from rbf.rbf import PHS
# from neural_fields.kernels import Gaussian
# from neural_fields.scattered import (
#     NeuralFieldSparse,
#     FlatTorrusDistance,
# )

plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": True,
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
    }
)

FILE = "media/flat_convergence"
colors = {deg: color for deg, color in zip(range(10), TABLEAU_COLORS.keys())}

#############
#
# nf simulation
#
#############

width = 2 * np.pi
t0, tf = 0, 2 * np.pi

threshold = 0.5
gain = 5
weight_kernel_sd = 0.025
sol_sd = 1.1
path_radius = 0.2
epsilon = 0.1

sol = ManufacturedSolutionPeriodic(
    weight_kernel_sd=weight_kernel_sd,
    threshold=threshold,
    gain=gain,
    solution_sd=sol_sd,
    path_radius=path_radius,
    epsilon=epsilon,
    period=width,
)

# rbf = PHS(3)
# solver = AB5(seed=RK4(), seed_steps_per_step=2)
# delta_t = 1e-3
# N = 32_000
# poly_deg = 2
# stencil_size = 24
# 
# print(f"Generating {N=} points.")
# points = UnitSquare(N, verbose=True).points * width - width / 2
# mesh = Delaunay(points)
# print("Computing quadrature weights.")
# qf = LocalQuad(
#     points=points,
#     rbf=rbf,
#     poly_deg=poly_deg,
#     stencil_size=stencil_size,
#     verbose=True,
#     tqdm_kwargs={"position": 0, "leave": True},
# )
# print("Constructing weight matrix.")
# nf = NeuralFieldSparse(
#     qf=qf,
#     firing_rate=sol.firing,
#     weight_kernel=Gaussian(sigma=weight_kernel_sd).radial,
#     dist=FlatTorrusDistance(x_width=width, y_width=width),
#     sparcity_tolerance=1e-16,
#     verbose=True,
#     tqdm_kwargs={"position": 0, "leave": True},
# )
#
#
# def rhs(t, u):
#     return nf.rhs(t, u) + sol.rhs(*points.T, t)
#
#
# u0 = sol.exact(*points.T, 0)
# time = TimeDomain_Start_Stop_MaxSpacing(t0, tf, delta_t)
# max_err = 0
# print("Time steping.")
# for t, u in (
#     tqdm_obj_inner := tqdm(
#         zip(time.array, solver.solution_generator(u0, rhs, time)),
#         total=len(time.array),
#         position=0,
#         leave=True,
#     )
# ):
#     my_exact = sol.exact(*points.T, t)
#     err = np.abs(u - my_exact) / my_exact
#     max_err = max(max_err, np.max(err))
#     tqdm_obj_inner.set_description(f"err={max_err:.3E}")
#
# approx = LocalInterpolator(
#     points=points,
#     fs=u - my_exact,
#     rbf=rbf,
#     poly_deg=poly_deg,
#     stencil_size=stencil_size,
# )
X, Y = np.meshgrid(*2 * (np.linspace(-np.pi, np.pi, 401),))
# eval_pnts = np.c_[X.ravel(), Y.ravel()]

# zs = approx(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)


#############
#
# Figure
#
#############

print("Generating Figure.")
figsize = (8, 10)
fig = plt.figure("flat convergence", figsize=figsize)

grid = gs.GridSpec(2, 2)

#############
#
# quad convergence
#
#############
ax_quad = fig.add_subplot(grid[0, 0])
data_file = "flat_convergence/data/flat_quad_convergence_figure.json"

with open(data_file, "rb") as f:
    results = [SimpleNamespace(**d) for d in json.load(f)]

poly_degs = list(set([result.poly_deg for result in results]))

for poly_deg in poly_degs:
    color = colors[poly_deg]
    my_res = [result for result in results if result.poly_deg == poly_deg]
    ns = [result.N for result in my_res]
    hs = [result.h for result in my_res]
    errs = [result.error for result in my_res]
    fit = linregress(np.log(hs), np.log(errs))
    ax_quad.loglog(hs, errs, ".", color=color)
    ax_quad.loglog(
        hs,
        [np.exp(fit.intercept + np.log(h) * fit.slope) for h in hs],
        "-",
        color=color,
        label=f"deg{poly_deg}~$\\mathcal{{O}}({fit.slope:.2f})$",
    )
ax_quad.legend()
ax_quad.set_title("Quadrature convergence")
ax_quad.set_ylabel("Relative Error")
ax_quad.set_xlabel("$h$")
ax_quad.set_ylim(1e-9, 1e-5)


#############
#
# nf convergence
#
#############
ax_nf = fig.add_subplot(grid[0, 1])
data_file = "flat_convergence/data/flat_nf_convergence.json"
with open(data_file, "rb") as f:
    results = [SimpleNamespace(**d) for d in json.load(f)]

poly_degs = list(set([result.poly_deg for result in results]))

for poly_deg in poly_degs:
    color = colors[poly_deg]
    my_res = [result for result in results if result.poly_deg == poly_deg]
    ns = [result.N for result in my_res]
    hs = [res.h for res in my_res]
    errs = [result.max_relative_error for result in my_res]
    fit = linregress(np.log(hs), np.log(errs))
    ax_nf.loglog(hs, errs, ".", color=color)
    ax_nf.loglog(
        hs,
        [np.exp(fit.intercept + np.log(h) * fit.slope) for h in hs],
        "-",
        color=color,
        label=f"deg{poly_deg}~$\\mathcal{{O}}({fit.slope:.2f})$",
    )
ax_nf.legend()
ax_nf.set_title("Neural Field convergence")
ax_nf.set_ylabel("Relative Error")
ax_nf.set_xlabel("$h$")

#############
#
# QF test function
#
#############

ax_func = fig.add_subplot(grid[1, 0])
x, y = sym.symbols("x y", real=True)


def cheb4(x):
    return 8 * x**4 - 8 * x**2 + 1


def cheb5(x):
    return 16 * x**5 - 20 * x**3 + 5 * x


sym_func = cheb5(2 * x - 1) * cheb4(2 * y - 1) + 1
func = sym.lambdify((x, y), sym_func)
func_X, func_Y = np.meshgrid(*2*(np.linspace(0, 1, 201), ))
ax_func.pcolormesh(func_X, func_Y, func(func_X, func_Y), cmap="jet")
ax_func.set_title("Quadrature Test function")

#############
#
# nf solution
#
#############

ax_sol = fig.add_subplot(grid[1, 1])
ax_sol.pcolormesh(X, Y, sol.exact(X, Y, tf), cmap="jet")
my_ts = np.linspace(0, 2*np.pi, 201)
ax_sol.plot(path_radius*np.cos(my_ts), path_radius*np.sin(my_ts), "w-")
ax_sol.set_title("Neural Field Solution")


#############
#
# Panel labels
#
#############
subplot_label_x = -0.05
subplot_label_y = 1.05
subplot_label_font = {
    "size": "x-large",
    "weight": "bold",
    "family": "sans",
    "usetex": False,
}
for ax, label in zip(
    [
        ax_quad,
        ax_nf,
        ax_func,
        ax_sol,
    ],
    "ABCDEFGH",
):
    ax.text(
        subplot_label_x,
        subplot_label_y,
        label,
        transform=ax.transAxes,
        **subplot_label_font,
    )

plt.suptitle("RBF-QF on the Unit Square")

grid.tight_layout(fig)
plt.show()

# pdfs do not look right with pcolormaps
# plt.savefig(FILE + ".png", dpi=300, bbox_inches="tight")
