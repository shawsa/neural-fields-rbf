"""
I have concluded that the approach in Sommariva and Vianello (2006) is
too difficult to generalize even with Sympy.

Look into Reeger and Fornberg (2016) instead. They seem to have
addressed the issue.
"""

from itertools import pairwise
import numpy as np
from sympy import integrate, lambdify, log, pi, sqrt, symbols
from sympy.printing.pycode import pycode

x, y, r = symbols("x y r")  # , real=True)
dist = sqrt(x**2 + y**2)

phi = r**2 * log(r)

phi_q = integrate(phi.subs(r, dist), x)

x1, y1, x2, y2 = symbols("x1 y1 x2 y2", real=True)

y_from_x = y1 + (x2 - x1)/(y2 - y1) * (x - x1)

term1 = (y2-y1)/(x2-x1)*integrate(phi_q.subs(y, y_from_x).expand(), (x, x1, x2))
term1_eval = lambdify((x1, y1, x2, y2), term1)

term2 = integrate(phi_q.subs(x1, y).expand(), (y, y1, y2))
term2_eval = lambdify((x1, y1, x2, y2), term2)


def segment_integrate(P1, P2):
    x1, y1 = P1
    x2, y2 = P2

    if abs(y2 - y1) > 1e-12:
        ret = term1_eval(x1, y1, x2, y2)
    else:
        ret = term2_eval(x1, y1, x2, y2)

    return ret


R = 1
exact = (integrate(phi * r, (r, 0, R)) * pi).evalf()


thetas = np.linspace(0, 2 * np.pi, 20)
xs = R * np.cos(thetas)
ys = R * np.sin(thetas)

approx = sum(segment_integrate(*pair) for pair in pairwise(zip(xs, ys)))

print(f"{exact=}")
print(f"{approx=}")


points = [
        (1, -1),
        (1, 1),
        (-1, 1),
        (-1, -1),
        (1, -1),
]

sum(segment_integrate(*pair) for pair in pairwise(points))
