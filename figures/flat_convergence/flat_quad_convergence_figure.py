import json
from matplotlib.colors import TABLEAU_COLORS
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from types import SimpleNamespace

DATA_FILE = "data/flat_quad_convergence.json"
MEDIA_FILE_NAME = "media/flat_quad_convergence"

with open(DATA_FILE, "rb") as f:
    results = [SimpleNamespace(**d) for d in json.load(f)]

colors = {deg: color for deg, color in zip(range(10), TABLEAU_COLORS.keys())}
plt.rcParams.update(
    {
        "font.size": 16,
        "text.usetex": True,
    }
)


poly_degs = list(set([result.poly_deg for result in results]))

plt.figure()
for poly_deg in poly_degs:
    color = colors[poly_deg]
    my_res = [result for result in results if result.poly_deg == poly_deg]
    ns = [result.N for result in my_res]
    hs = [result.h for result in my_res]
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
plt.title("Test Function $T_5(2x-1)T_4(2y-1) + 1$")
plt.ylabel("Relative Error")
plt.xlabel("$h$")
plt.ylim(1e-9, 1e-5)
plt.savefig(MEDIA_FILE_NAME)
