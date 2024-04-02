from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from numpy.polynomial import Polynomial as nppoly
from typing import Callable
from odeiter import TimeDomain_Start_Stop_MaxSpacing, RK4, TqdmWrapper
from rbf.quadrature import LocalQuad
from rbf.rbf import PHS
from rbf.points import UnitSquare
import sympy as sym

import imageio.v2 as imageio
import os


FILE_NAME = "scattered.gif"
SAVE_ANIMATION = True


class NeuralField:
    def __init__(
        self,
        qf: LocalQuad,
        firing_rate: Callable[[np.ndarray], np.ndarray],
        weight_kernel: Callable[[np.ndarray], np.ndarray],
        dist=lambda x, y: la.norm(x - y, axis=1),
    ):
        self.qf = qf
        self.points = qf.points
        self.firing_rate = firing_rate
        self.weight_kernel = weight_kernel
        self.dist = dist
        self.initialize_convolution()

    def initialize_convolution(self):
        conv_mat = np.zeros((len(self.points), len(self.points)))
        for index, point in enumerate(self.points):
            conv_mat[index] = self.qf.weights * np.array(
                [self.weight_kernel(self.dist(self.points, point))]
            )
        self.conv_mat = conv_mat

    def conv(self, arr: np.ndarray):
        return self.conv_mat @ arr

    def rhs(self, t, u):
        return -u + self.conv(self.firing_rate(u))


class Sigmoid:
    def __init__(self, threshold=0.2, gain=20):
        self.threshold = threshold
        self.gain = gain

    def __call__(self, u):
        return 1 / (1 + np.exp(-self.gain * (u - self.threshold)))

    def inv(self, f):
        return self.threshold - 1 / self.gain * np.log(1 / f - 1)


class Gaussian:
    def __init__(self, sigma=1.0):
        self.sigma = sigma

    def radial(self, r):
        return (
            1
            / (2 * np.pi * self.sigma**2)
            * np.exp(-(r**2) / (2 * self.sigma**2))
        )

    def __call__(self, x):
        return self.radial(la.norm(x, axis=1))


class HermiteBump:
    def __init__(
        self,
        *,
        threshold: float,
        radius: float,
        order: int,
        r: sym.Symbol = sym.symbols("r"),
    ):
        self.r = r
        self.radius = radius
        self.threshold = threshold
        self.order = order
        self.init_poly()

    def init_poly(self):
        # lower triangle edge
        coeffs = [1] + [0] * self.order
        # fill lower left block
        for i in range(self.order):
            for j in range(self.order, 0, -1):
                coeffs[j] = coeffs[j] - coeffs[j - 1]
        for i in range(self.order):
            for j in range(self.order, i, -1):
                coeffs[j] = coeffs[j] - coeffs[j - 1]
        poly = 0
        for p, c in enumerate(coeffs):
            poly += c * self.r ** (self.order + 1) * (self.r - 1) ** p
        self.poly_sym = poly.subs(self.r, self.r)
        self.poly_coeffs = [float(c) for c in sym.Poly(self.poly_sym).all_coeffs()]
        self.poly_full = nppoly(self.poly_coeffs[::-1])

    def profile(self, rs: np.ndarray[float]):
        ret = np.zeros_like(rs)
        ret[rs >= 1] = 1
        poly_mask = np.logical_and(rs > 0, rs < 1)
        ret[poly_mask] = self.poly_full(rs[poly_mask])
        return ret

    def __call__(self, r):
        return self.profile(((r - self.threshold)/(2*self.radius) + 0.5))


if __name__ == "__main__":
    t0, tf = 0, 50
    delta_t = 1e-1
    width = 30

    N = 16_000
    rbf = PHS(7)
    poly_deg = 2
    stencil_size = 21

    points = UnitSquare(N, verbose=True).points * width - width / 2
    qf = LocalQuad(
        points=points,
        rbf=rbf,
        poly_deg=poly_deg,
        stencil_size=stencil_size,
        verbose=True,
    )

    # plt.figure("weights")
    # neg_mask = qf.weights < 0
    # plt.plot(*points[neg_mask].T, "k*")
    # plt.scatter(*points.T, c=qf.weights, s=1.0, cmap="jet")
    # plt.axis("equal")
    # plt.colorbar()

    def periodic_dist(points, x):
        x_offsets = [i * width for i in [-1, 0, 1]]
        y_offsets = [i * width for i in [-1, 0, 1]]
        dist_tensor = np.zeros((len(x_offsets) * len(y_offsets), len(points)))
        for index, (x_offset, y_offset) in enumerate(product(x_offsets, y_offsets)):
            x2 = x + np.array([x_offset, y_offset])
            dist_tensor[index] = la.norm(points - x2, axis=1)
        return np.min(dist_tensor, axis=0)

    def weight_kernel(r):
        return np.exp(-r) * (2 - r)
        # return np.exp(-r)

    # firing_rate = Sigmoid(threshold=0.2, gain=20)
    firing_rate = HermiteBump(threshold=0.1, radius=0.1, order=3)
    nf = NeuralField(
        qf=qf,
        firing_rate=firing_rate,
        weight_kernel=weight_kernel,
        dist=periodic_dist,
    )

    time = TimeDomain_Start_Stop_MaxSpacing(t0, tf, delta_t)
    solver = TqdmWrapper(RK4())

    u0 = Gaussian(sigma=1.0)(points)

    plt.figure("Solution")
    scatter = plt.scatter(*points.T, c=u0, s=4, cmap="jet", vmin=-0.5, vmax=2.0)
    plt.axis("equal")
    plt.colorbar()

    class NullContext:
        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

        def append_data(self, *args):
            pass

    if SAVE_ANIMATION:
        writer = imageio.get_writer(FILE_NAME, mode="I")
    else:
        writer = NullContext()

    with writer:
        for u in solver.solution_generator(u0, nf.rhs, time):
            scatter.set_array(u)
            plt.draw()
            plt.pause(1e-3)
            plt.savefig(FILE_NAME + ".png")
            image = imageio.imread(FILE_NAME + ".png")
            os.remove(FILE_NAME + ".png")
            writer.append_data(image)
