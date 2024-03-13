"""
Test convergence using pregenerated quadrature formulae.
"""

from dataclasses import dataclass
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.ticker import NullFormatter, ScalarFormatter
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import sympy as sym
from pre_gen_points import CachedQF, pregens

colors = {deg: color for deg, color in zip(range(10), TABLEAU_COLORS.keys())}
plt.rcParams.update(
    {
        "font.size": 16,
        "text.usetex": True,
    }
)


x, y = sym.symbols("x y", real=True)


def cheb7(x):
    return 64*x**7 - 112*x**5 + 56*x**3 - 7*x


def cheb4(x):
    return 8*x**4 - 8*x**2 + 1


def cheb5(x):
    return 16*x**5 - 20*x**3 + 5*x


# sym_func = cheb7(2*x - 1) * cheb7(2*y - 1) + 1,
sym_func = cheb5(2*x - 1) * cheb4(2*y - 1) + 1
exact = float(sym.integrate(sym.integrate(sym_func, (x, 0, 1)), (y, 0, 1)))


@dataclass
class Result:
    n: int
    error: float
    poly_deg: int


func = sym.lambdify((x, y), sym_func)
results = []
for qf in pregens():
    fs = func(*qf.points.T)
    approx = fs @ qf.weights
    error = abs((approx - exact) / exact)
    results.append(Result(n=len(qf.points), error=error, poly_deg=qf.poly_deg))

poly_degs = sorted(list(set([result.poly_deg for result in results])))

plt.figure()
for poly_deg in poly_degs:
    color = colors[poly_deg]
    my_res = [result for result in results if result.poly_deg == poly_deg]
    hs = [result.n**-0.5 for result in my_res]
    ns = [result.n for result in my_res]
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
plt.xlabel("$N^{-1/2}$")
n_max = max(ns)
n_min = min(ns)
n_med = int(np.median(ns))
ns = [n_min, n_med, n_max]
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.gca().xaxis.set_minor_formatter(NullFormatter())
plt.xticks()
plt.xticks([n**-.5 for n in ns], [f"${n}^{{-1/2}}$" for n in ns])
plt.tight_layout()
plt.savefig("../presentation_images/quad_convergence.png", dpi=300)
