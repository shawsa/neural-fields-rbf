import json
from matplotlib.colors import TABLEAU_COLORS
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gs
from matplotlib.colors import Normalize

# import matplotlib.patches as patches
import numpy as np
import pickle
from scipy.stats import linregress
from types import SimpleNamespace

plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": True,
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
        "axes.spines.right": False,
        "axes.spines.top": False,
    }
)

FILE = "media/torus_convergence"
colors = {deg: color for deg, color in zip(range(10), TABLEAU_COLORS.keys())}

#############
#
# Figure
#
#############

figsize = (8, 7)
fig = plt.figure("flat convergence", figsize=figsize)

num_gs_rows = 12
num_gs_rows_top = 6
num_gs_rows_bot = 5
grid = gs.GridSpec(num_gs_rows, 2)
top_row = slice(0, num_gs_rows_top)
bot_row = slice(num_gs_rows - num_gs_rows_bot, num_gs_rows)
bot_row_small = slice(num_gs_rows - num_gs_rows_bot, num_gs_rows-1)

#############
#
# quad convergence
#
#############
ax_quad = fig.add_subplot(grid[top_row, 0])
data_file = "torus_convergence/data/torus_quad.json"

with open(data_file, "rb") as f:
    results = [SimpleNamespace(**d) for d in json.load(f)]
    results = [res for res in results if res.poly_deg > 0]

poly_degs = list(set([result.poly_deg for result in results]))

for poly_deg in poly_degs:
    color = colors[poly_deg]
    my_res = [result for result in results if result.poly_deg == poly_deg]
    ns = [result.N for result in my_res]
    hs = [1 / np.sqrt(result.N) for result in my_res]
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
ax_quad.set_title("Quadrature Convergence")
ax_quad.set_ylabel("Relative Error")
ax_quad.set_xlabel("${N}^{-1/2}$")
# ax_quad.set_ylim(1e-9, 1e-5)
ax_quad.xaxis.set_major_formatter(ScalarFormatter())
ax_quad.minorticks_off()
N_ticks = [32_000, 45_000, 64_000]
tic_locs = [1 / np.sqrt(tick) for tick in N_ticks]
ax_quad.set_xticks(tic_locs, [f"${tick}^{{-1/2}}$" for tick in N_ticks])


#############
#
# nf convergence
#
#############
ax_nf = fig.add_subplot(grid[top_row, 1])
data_file = "torus_convergence/data/torus_nf.json"
with open(data_file, "rb") as f:
    results = [SimpleNamespace(**d) for d in json.load(f)]

poly_degs = list(set([result.poly_deg for result in results]))

for poly_deg in poly_degs:
    color = colors[poly_deg]
    my_res = [result for result in results if result.poly_deg == poly_deg]
    ns = [result.N for result in my_res]
    hs = [1 / np.sqrt(res.N) for res in my_res]
    errs = [result.max_err for result in my_res]
    fit = linregress(np.log(hs), np.log(errs))
    ax_nf.loglog(hs, errs, ".-", color=color, label=f"deg={poly_deg}")
    # ax_nf.loglog(
    #     hs,
    #     [np.exp(fit.intercept + np.log(h) * fit.slope) for h in hs],
    #     "-",
    #     color=color,
    #     label=f"deg{poly_deg}~$\\mathcal{{O}}({fit.slope:.2f})$",
    # )
upper_order = 4
lower_order = 10
result_upper = results[
    np.argmax(
        [np.log(res.max_err) - lower_order * np.log(res.N**-0.5) for res in results]
    )
]
result_lower = results[
    np.argmin(
        [np.log(res.max_err) - upper_order * np.log(res.N**-0.5) for res in results]
    )
]
for (o, res), linestyle in zip(
    [
        (upper_order, result_upper),
        (lower_order, result_lower),
    ],
    ["--", ":", "-.", "-"],
):
    ax_nf.loglog(
        hs,
        [res.max_err * np.exp(np.log(h / res.N**-0.5) * o) for h in hs],
        linestyle=linestyle,
        color="black",
        label=f"$\\mathcal{{O}}({o})$",
    )

ax_nf.legend()
ax_nf.set_title("Neural Field Convergence")
ax_nf.set_ylabel("Relative Error")
ax_nf.set_xlabel("$N^{-1}$")
ax_nf.xaxis.set_major_formatter(ScalarFormatter())
ax_nf.minorticks_off()
N_ticks = [32_000, 45_000, 64_000]
tic_locs = [1 / np.sqrt(tick) for tick in N_ticks]
ax_nf.set_xticks(tic_locs, [f"${tick}^{{-1/2}}$" for tick in N_ticks])

#############
#
# weights panel
#
#############

ax_weights = fig.add_subplot(grid[bot_row, 0])
with open("torus_convergence/media/torus_weights.png", "rb") as f:
    image = plt.imread(f)
im = ax_weights.imshow(image[200:600, 200:800])
ax_weights.axis("off")


#############
#
# histogram
#
#############

with open("torus_convergence/data/torus_weights.pickle", "rb") as f:
    weights = pickle.load(f)

rounded_weights = np.round(weights, 10)
unique_weights = sorted(list(set(rounded_weights)))
weight_counts = [
    sum(1 for w in rounded_weights if w == weight) for weight in unique_weights
]

ax_hist = fig.add_subplot(grid[bot_row_small, 1])
cnorm = Normalize(np.min(weights), np.max(weights))
for w, c in zip(unique_weights, weight_counts):
    color = plt.cm.viridis(cnorm(w))
    ax_hist.plot([w, w], [0, c], "-", color=color, markersize=10, linewidth=0.5)
    ax_hist.plot([w], [c], ".", color=color, markersize=10, linewidth=0.5)

ax_hist.set_xlabel("weights")
ax_hist.set_ylabel("counts")

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
    "family": "stix",
    "usetex": True,
}
for ax, label in zip(
    [
        ax_quad,
        ax_nf,
        ax_weights,
        ax_hist,
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

plt.suptitle("RBF-QF on the Torus")

grid.tight_layout(fig)
plt.show()

# pdfs do not look right with pcolormaps
plt.savefig("media/torus_convergence.png", dpi=300, bbox_inches="tight")
