from matplotlib.colors import TABLEAU_COLORS
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from scipy.stats import linregress

from .convergence import Result

colors = {deg: color for deg, color in zip(range(10), TABLEAU_COLORS.keys())}
plt.rcParams.update(
    {
        "font.size": 16,
        "text.usetex": True,
    }
)


DATA_DIR = "data"
MEDIA_DIR = "media"
file_prefix = "convergence_data"
results = []
for filename in os.listdir(DATA_DIR):
    full_path = os.path.join(DATA_DIR, filename)
    if not os.path.isfile(full_path):
        continue
    if not filename[:len(file_prefix)] == file_prefix:
        continue
    with open(full_path, "rb") as f:
        results += pickle.load(f)


time_step_sizes = set(res.delta_t for res in results)
for delta_t in time_step_sizes:
    plt.figure(f"Convergence {delta_t=:.2E}", figsize=(7, 5))
    for poly_deg in set([res.poly_deg for res in results]):
        color = colors[poly_deg]
        my_res = [
            res
            for res in results
            if res.poly_deg == poly_deg and res.delta_t == delta_t
        ]
        my_hs = [res.h for res in my_res]
        my_errs = [res.max_relative_error for res in my_res]
        fit = linregress(np.log(my_hs), np.log(my_errs))
        plt.loglog(my_hs, my_errs, ".", color=color, label=f"{poly_deg=}")
        plt.loglog(
            my_hs,
            [np.exp(fit.intercept + np.log(h) * fit.slope) for h in my_hs],
            "-",
            color=color,
            label=f"$\\mathcal{{O}}({fit.slope:.2f})$",
        )
    plt.xlabel("$h$")
    plt.ylabel("Relative Error")
    plt.grid()
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    filepath = os.path.join(MEDIA_DIR, f"convergence{delta_t=:.2E}.png")
    plt.savefig(filepath)
