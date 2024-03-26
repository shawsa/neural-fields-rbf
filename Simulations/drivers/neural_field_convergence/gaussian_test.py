import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from typing import Callable
from odeiter import TimeDomain_Start_Stop_MaxSpacing, RK4, TqdmWrapper
import sympy as sym
from rbf.quadrature import LocalQuad
from rbf.rbf import PHS
from rbf.points import UnitSquare

from neural_fields.scattered import NeuralField, Gaussian, Sigmoid

x, y, t = sym.symbols("x y t", real=True)

firing_rate = Sigmoid(threshold=0.2, gain=20)

############################
# Error
############################
"""
Need to change inverse firing rate.
Need f[u] -> 0 as u -> 0
Currently, f[u] -> 0 as u -> -oo
Maybe try piecewise polynomial.
"""


g1 = Gaussian(sigma=1.0)


center_x_sym = sym.cos(t)
center_y_sym = sym.sin(t)


def inv_firing(f):
    return firing_rate.threshold - 1 / firing_rate.gain * sym.log(1 / f - 1)


r = sym.sqrt((x - center_x_sym) ** 2 + (y - center_y_sym) ** 2)
exact_sym = inv_firing(
    1 / (2 * sym.pi * g1.sigma**2) * sym.exp(-(r**2) / (2 * g1.sigma**2))
)

f1_sym = exact_sym + exact_sym.diff(t)
f1 = sym.lambdify((t, x, y), f1_sym)

center_x = sym.lambdify(t, center_x_sym)
center_y = sym.lambdify(t, center_y_sym)
exact = sym.lambdify((t, x, y), exact_sym)


g2 = Gaussian(sigma=1.0)


def weight_kernel(r):
    return g2.radial(r)


g3 = Gaussian(sigma=np.sqrt(g1.sigma**2 + g2.sigma**2))


t0, tf = 0, 10
delta_t = 5e-2
width = 100

N = 8_00
rbf = PHS(7)
poly_deg = 3
stencil_size = 41

points = UnitSquare(N, verbose=True).points * width - width / 2

xs, ys = points.T


def forcing(t):
    return f1(t, xs, ys) - g3.radial(
        np.sqrt((xs - center_x(t)) ** 2 + (ys - center_y(t)) ** 2)
    )


qf = LocalQuad(
    points=points,
    rbf=rbf,
    poly_deg=poly_deg,
    stencil_size=stencil_size,
    verbose=True,
)

plt.figure("weights")
neg_mask = qf.weights < 0
plt.plot(*points[neg_mask].T, "k*")
plt.scatter(*points.T, c=qf.weights, s=1.0, cmap="jet")
plt.axis("equal")
plt.colorbar()

nf = NeuralField(
    qf=qf,
    firing_rate=Sigmoid(threshold=0.2, gain=20),
    weight_kernel=weight_kernel,
)


def rhs(t, u):
    return nf.rhs(t, u) + forcing(t)


time = TimeDomain_Start_Stop_MaxSpacing(t0, tf, delta_t)
solver = TqdmWrapper(RK4())

u0 = exact(0, xs, ys)

plt.figure("Solution")
scatter = plt.scatter(*points.T, c=u0, s=1, cmap="jet", vmin=-0.5, vmax=2.0)
plt.axis("equal")
plt.colorbar()

for u in solver.solution_generator(u0, rhs, time):
    scatter.set_array(u)
    plt.draw()
    plt.pause(1e-3)
