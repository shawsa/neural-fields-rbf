from dataclasses import dataclass
import pickle
import numpy as np
from matplotlib.colors import TABLEAU_COLORS
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
from scipy.stats import linregress

from torus_convergence import Result

DATA_FILE = "data/torrus1.pickle"

with open(DATA_FILE, "rb") as f:
    results = pickle.load(f)

poly_degs = list(set(result.poly_deg for result in results))
for poly_deg, color in zip(poly_degs, TABLEAU_COLORS):
    my_res = [
        result
        for result in results
        if result.poly_deg == poly_deg
    ]
    hs = [result.h for result in my_res]
    ns = [result.N for result in my_res]
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
plt.xlabel("$h$")
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.gca().xaxis.set_minor_formatter(NullFormatter())

plt.savefig("media/torus_convergence.png")
