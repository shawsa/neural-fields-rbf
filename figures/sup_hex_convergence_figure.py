import json
import matplotlib.pyplot as plt
from matplotlib.colors import (
    TABLEAU_COLORS,
    Normalize,
)
import matplotlib.gridspec as gs
import numpy as np
from scipy.spatial import Delaunay
from types import SimpleNamespace

from neural_fields_rbf.rbf.quadrature import LocalQuad
from neural_fields_rbf.rbf.rbf import PHS
from utils import hex_stencil_min, hex_grid

plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": True,
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
    }
)

#############
#
# Load and generate data
#
#############
DATA_FILE = "data/flat_hex_quad_convergence.json"

with open(DATA_FILE, "r") as f:
    results = [SimpleNamespace(**d) for d in json.load(f)]

poly_degs = sorted(set([res.poly_deg for res in results]))
Ns = sorted(set([res.N for res in results]))

N = 400
points = hex_grid(N)
rbf = PHS(3)
poly_deg = 4
stencil_size = hex_stencil_min(30)
mesh = Delaunay(points)
quad = LocalQuad(
    points=points,
    rbf=rbf,
    poly_deg=poly_deg,
    stencil_size=stencil_size,
    verbose=True,
    tqdm_kwargs={
        "position": 3,
        "leave": False,
        "desc": "Calculating weights",
    },
)
negative_mask = quad.weights < 0
print(f"{np.sum(negative_mask)} negative weights out of {len(mesh.points)}")
print(f"{stencil_size=}")

#############
#
# Generate Figure
#
#############

figsize = (8, 8)
fig = plt.figure("Sphere Quadrature Convergence", figsize=figsize)

grid = gs.GridSpec(12, 12)
axes = []

#############
#
# example grid
#
#############

ax_nodes = fig.add_subplot(grid[:5, :5])
ax_nodes.scatter(*quad.points.T, c=quad.weights, s=5, cmap="viridis")
ax_nodes.scatter(*quad.points[negative_mask].T, c="black", s=20, cmap="viridis")
ax_nodes.axis("equal")
ax_nodes.axis("off")

#############
#
# histogram
#
#############

ax_hist = fig.add_subplot(grid[:5, 7:])
_, bins, patches = ax_hist.hist(quad.weights, bins=25, orientation="horizontal")
cnorm = Normalize(np.min(quad.weights), np.max(quad.weights))
for val, patch in zip(bins, patches):
    patch.set_facecolor(plt.cm.viridis(cnorm(val)))

ax_hist.set_ylabel("Weight")
ax_hist.set_xlabel("Count")

#############
#
# convergence plots
#
#############

poly = "x**3 - y**4"
gauss = "Gaussian"

func_str_dict = {
    poly: "$x^3 - y^4$",
    gauss: "Gausssian",
}

colors = {deg: color for deg, color in zip(range(10), TABLEAU_COLORS.keys())}

for func_str in [poly, gauss]:
    if func_str == poly:
        filtered_results = [res for res in results if res.func == poly]
        ax_poly = fig.add_subplot(grid[6:, :5])
        ax = ax_poly
    if func_str == gauss:
        filtered_results = [res for res in results if res.func == gauss]
        ax_gauss = fig.add_subplot(grid[6:, 5:10], sharey=ax_poly)
        ax = ax_gauss
    for poly_deg in poly_degs:
        color = colors[poly_deg]
        my_res = [result for result in filtered_results if result.poly_deg == poly_deg]
        if len(my_res) == 0:
            continue
        my_res.sort(key=lambda res: res.N)
        ns = [result.N for result in my_res]
        hs = [1 / np.sqrt(result.N) for result in my_res]
        # hs = [res.h for res in my_res]
        errs = [np.abs(result.relative_error) for result in my_res]
        ax.loglog(hs, errs, ".-", color=color, label=f"deg={poly_deg}")
        h_bounds = [hs[0], hs[-1]]
        max_err = errs[0]
        if func_str == poly and poly_deg != 4:
            ax.loglog(
                h_bounds,
                [
                    max_err,
                    max_err * np.exp(np.log(h_bounds[1] / h_bounds[0]) * poly_deg),
                ],
                linestyle=":",
                color=color,
                label=f"$\\mathcal{{O}}({poly_deg})$",
            )
    if func_str == gauss:
        for poly_deg in poly_degs:
            if poly_deg == 4:
                continue
            ax.loglog(
                [],
                [],
                linestyle=":",
                color=colors[poly_deg],
                label=f"$\\mathcal{{O}}({poly_deg})$",
            )
        order = 12
        ax.loglog(
            h_bounds,
            [max_err, max_err * np.exp(np.log(h_bounds[1] / h_bounds[0]) * order)],
            linestyle=":",
            color="black",
            label=f"$\\mathcal{{O}}({order})$",
        )
    ax.set_title(func_str_dict[func_str])

ax_poly.set_ylabel("Relative Error")
for ax in [ax_poly, ax_gauss]:
    N_bounds = [max(ns), min(ns)]
    ax.set_xticks(
        [1 / np.sqrt(my_N) for my_N in ns],
    )
    ax.set_xticklabels(
        [f"${my_N}^{{-1/2}}$" for my_N in ns],
    )
    ax.set_xlabel("${N}^{-1/2}$")
    ax.minorticks_off()
    ax.tick_params(axis="x", labelrotation=45)

ax_gauss.legend(loc="center right", bbox_to_anchor=(1.5, 0.5, 0, 0))
ax_gauss.tick_params(axis="y", labelleft=False)

#############
#
# Panel labels
#
#############
subplot_label_x = -0.05
subplot_label_y = 1.07
subplot_label_font = {
    "size": "x-large",
    "weight": "bold",
    "family": "stix",
    "usetex": True,
}
for ax, label in zip(
    [
        ax_nodes,
        ax_hist,
        ax_poly,
        ax_gauss,
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

# plt.suptitle("")

grid.tight_layout(fig)
plt.show()
plt.savefig("media/hex_quad_convergence.pdf")
