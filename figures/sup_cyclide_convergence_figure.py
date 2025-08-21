import json
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
import matplotlib.gridspec as gs
import numpy as np
from types import SimpleNamespace

CONVERGENCE_DATA_FILE = (
    "convergence_on_multiple_geometries/data/cyclide_convergence.json"
)
NODES_IMAGE_FILE = "convergence_on_multiple_geometries/media/cyclide_nodes.png"
plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": True,
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
    }
)

with open(CONVERGENCE_DATA_FILE, "r") as f:
    results = [SimpleNamespace(**d) for d in json.load(f)]

poly_degs = sorted(set([res.poly_deg for res in results]))
funcs = list(set([res.func for res in results]))

func_str_dict = {
    "1": "1",
    "poly": "$x^3y^2z^4 + 5$",
    "trig": "$\\sin(x)\\cos(2y)\\cos(3z)$",
}
funcs.sort(key=lambda func: list(func_str_dict.keys()).index(func))

#############
#
# Generate Figure
#
#############

figsize = (8, 8)
fig = plt.figure("Sphere Quadrature Convergence", figsize=figsize)

grid = gs.GridSpec(12, 14)
axes = []

#############
#
# sphere nodes
#
#############

with open(NODES_IMAGE_FILE, "rb") as f:
    image = plt.imread(f)

ax_nodes = fig.add_subplot(grid[:6, :])
ax_nodes.imshow(image[160:650, 200:800])
ax_nodes.axis("off")
axes.append(ax_nodes)

#############
#
# convergence plots
#
#############

width = grid.ncols // len(funcs)
my_axes = [
    fig.add_subplot(grid[6:, col * width : (col + 1) * width])
    for col, _ in enumerate(funcs)
]
for ax in my_axes[1:]:
    ax.sharey(my_axes[0])
axes = axes + my_axes
for ax, func_str in zip(my_axes, funcs):
    colors = {deg: color for deg, color in zip(range(10), TABLEAU_COLORS.keys())}
    for poly_deg in poly_degs:
        color = colors[poly_deg]
        my_res = [
            result
            for result in results
            if result.poly_deg == poly_deg and result.func == func_str
        ]
        ns = [result.N for result in my_res]
        hs = [1 / np.sqrt(result.N) for result in my_res]
        # hs = [res.h for res in my_res]
        errs = [np.abs(result.relative_error) for result in my_res]
        ax.loglog(hs, errs, ".-", color=color, label=f"deg={poly_deg}")
        h_bounds = [max(hs), min(hs)]
        max_err = max(errs)
        ax.loglog(
            h_bounds,
            [max_err, max_err * np.exp(np.log(h_bounds[1] / h_bounds[0]) * poly_deg)],
            linestyle=":",
            color=color,
            label=f"$\\mathcal{{O}}({poly_deg})$",
        )
    # ax.legend()
    ax.set_title(func_str_dict[func_str])

my_axes[-1].legend(loc="upper right", bbox_to_anchor=(1.7, 1, 0, 0))
my_axes[0].set_ylabel("Relative Error")
for ax in my_axes:
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

for ax in my_axes[1:]:
    ax.tick_params(axis="y", labelleft=False)

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
    axes,
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
plt.savefig("media/cyclide_quad_convergence.pdf")
