from dataclasses import dataclass, asdict
import json
import numpy as np
from matplotlib.colors import TABLEAU_COLORS
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
from scipy.stats import linregress
from types import SimpleNamespace

DATA_FILE = "data/torus_quad.json"

with open(DATA_FILE, "rb") as f:
    results = [SimpleNamespace(**d) for d in json.load(f)]

poly_degs = list(set(result.poly_deg for result in results))
for poly_deg, color in zip(poly_degs, TABLEAU_COLORS):
    my_res = [result for result in results if result.poly_deg == poly_deg]
    # hs = [result.h for result in my_res]
    ns = [result.N for result in my_res]
    hs = [1 / np.sqrt(result.N) for result in my_res]
    errs = [result.error for result in my_res]
    fit = linregress(np.log(hs), np.log(errs))
    plt.loglog(hs, errs, ".", color=color)
    plt.loglog(
        hs,
        [np.exp(fit.intercept + np.log(h) * fit.slope) for h in hs],
        "-",
        color=color,
        label=f"deg{poly_deg}~$\\mathcal{{O}}({fit.slope:.2f})$",
    )
    plt.legend()

plt.ylabel("Relative Error")
plt.xlabel("$\sqrt{N}^{-1}$")
# N_markers = [128_000, 90_000, 64_000]
N_markers = list(ns[::5])
plt.xticks(
    [1 / np.sqrt(N) for N in N_markers], ["$\\sqrt{%d}^{-1}$" % N for N in N_markers]
)
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.gca().xaxis.set_minor_formatter(NullFormatter())
plt.title("Quadrature Convergence")

plt.savefig("media/torus_quad_convergence.png")
