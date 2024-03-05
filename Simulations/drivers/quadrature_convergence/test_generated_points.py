"""
Run convergence Tests for number of points
"""
from dataclasses import dataclass
from matplotlib.colors import TABLEAU_COLORS
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
import sympy as sym
from pre_gen_points import CachedQF, pregens

colors = {deg: color for deg, color in zip(range(10), TABLEAU_COLORS.keys())}
plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": True,
    }
)


x, y = sym.symbols("x y", real=True)

test_functions = [
    sym.sin(10 * x) * sym.sin(10 * y) + 1,
]
exact_values = [
    float(sym.integrate(sym.integrate(sym_func, (x, 0, 1)), (y, 0, 1)))
    for sym_func in test_functions
]


@dataclass
class Result:
    n: int
    error: float
    poly_deg: int


for sym_func, exact in zip(test_functions, exact_values):
    func = sym.lambdify((x, y), sym_func)
    results = []
    for qf in pregens():
        fs = func(*qf.points.T)
        approx = fs @ qf.weights
        error = abs((approx - exact)/exact)
        results.append(Result(n=len(qf.points), error=error, poly_deg=qf.poly_deg))

    poly_degs = sorted(list(set([result.poly_deg for result in results])))

    plt.figure(str(sym_func))
    for poly_deg in poly_degs:
        color = colors[poly_deg]
        my_res = [result for result in results if result.poly_deg == poly_deg]
        hs = [result.n**-0.5 for result in my_res]
        errs = [result.error for result in my_res]
        fit = linregress(np.log(hs), np.log(errs))
        plt.loglog(hs, errs, ".", color=color)
        plt.loglog(
            hs,
            [np.exp(fit.intercept + np.log(h) * fit.slope) for h in hs],
            "-",
            color=color,
            label=f"deg={poly_deg}~$\\mathcal{{O}}({fit.slope:.2f})$",
        )
    plt.legend()
    plt.ylabel("Relative Error")
    plt.xlabel("$N^{-1/2}$")
