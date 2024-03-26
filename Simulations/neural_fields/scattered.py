import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from typing import Callable
from odeiter import TimeDomain_Start_Stop_MaxSpacing, RK4, TqdmWrapper
from rbf.quadrature import LocalQuad
from rbf.rbf import PHS
from rbf.points import UnitSquare


class NeuralField:
    def __init__(
        self,
        qf: LocalQuad,
        firing_rate: Callable[[np.ndarray], np.ndarray],
        weight_kernel: Callable[[np.ndarray], np.ndarray],
    ):
        self.qf = qf
        self.points = qf.points
        self.firing_rate = firing_rate
        self.weight_kernel = weight_kernel
        self.initialize_convolution()

    def initialize_convolution(self):
        conv_mat = np.zeros((len(self.points), len(self.points)))
        for index, point in enumerate(self.points):
            conv_mat[index] = self.qf.weights * np.array(
                [self.weight_kernel(la.norm(self.points - point, axis=1))]
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


if __name__ == "__main__":
    t0, tf = 0, 10
    delta_t = 5e-2
    width = 100

    N = 8_000
    rbf = PHS(7)
    poly_deg = 3
    stencil_size = 41

    points = UnitSquare(N, verbose=True).points * width - width / 2
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

    def weight_kernel(r):
        return np.exp(-r) * (2 - r)

    nf = NeuralField(
        qf=qf,
        firing_rate=Sigmoid(threshold=0.2, gain=20),
        weight_kernel=weight_kernel,
    )

    time = TimeDomain_Start_Stop_MaxSpacing(t0, tf, delta_t)
    solver = TqdmWrapper(RK4())

    u0 = Gaussian(sigma=1.0)(points)

    plt.figure("Solution")
    scatter = plt.scatter(*points.T, c=u0, s=1, cmap="jet", vmin=-.5, vmax=2.0)
    plt.axis("equal")
    plt.colorbar()

    for u in solver.solution_generator(u0, nf.rhs, time):
        scatter.set_array(u)
        plt.draw()
        plt.pause(1e-3)
