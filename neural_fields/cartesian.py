from itertools import product
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
from numpy.fft import fft2, ifft2
from typing import Callable
from odeiter import TimeDomain_Start_Stop_MaxSpacing, RK4, TqdmWrapper

from scipy.special import k0


class SpaceDomain:
    def __init__(
        self,
        x_min: float,
        x_max: float,
        x_num: int,
        y_min: float,
        y_max: float,
        y_num: int,
    ):
        self.x_min = x_min
        self.x_max = x_max
        self.x_num = x_num
        self.x_width = x_max - x_min
        self.x_step = self.x_width / x_num

        self.y_min = y_min
        self.y_max = y_max
        self.y_num = y_num
        self.y_width = y_max - y_min
        self.y_step = self.y_width / y_num

        self.X, self.Y = np.meshgrid(
            np.linspace(self.x_min, self.x_max, self.x_num, endpoint=False),
            np.linspace(self.y_min, self.y_max, self.y_num, endpoint=False),
        )

    def __repr__(self):
        ret = f"Domain [{self.x_min}, {self.x_max}]*[{self.y_min}, {self.y_max}]"
        ret += " " * 4 + f"Cell size: {(self.x_step, self.y_step)}"
        return ret

    def dist(self, x, y):
        """An array of distances from the point (x, y) to the points in (X2, Y2).
        Return value has the shape of X2 (which must match Y2)"""
        x_offsets = [i * self.x_width for i in [-1, 0, 1]]
        y_offsets = [i * self.y_width for i in [-1, 0, 1]]
        dist_tensor = np.zeros((len(x_offsets) * len(y_offsets),) + self.X.shape)
        for index, (x_offset, y_offset) in enumerate(product(x_offsets, y_offsets)):
            dist_tensor[index] = np.sqrt(
                (x + x_offset - self.X) ** 2 + (y + y_offset - self.Y) ** 2
            )
        return np.min(dist_tensor, axis=0)


class NeuralField:
    def __init__(
        self,
        space: SpaceDomain,
        firing_rate: Callable[[np.ndarray], np.ndarray],
        weight_kernel: Callable[[np.ndarray], np.ndarray],
    ):
        self.space = space
        self.firing_rate = firing_rate
        self.weight_kernel = weight_kernel
        self.initialize_convolution()

    def initialize_convolution(self):
        r = self.space.dist(self.space.x_min, self.space.y_min)
        factor = self.space.x_step * np.sqrt(self.space.x_num)
        factor *= self.space.y_step * np.sqrt(self.space.y_num)
        kernel = self.weight_kernel(r)
        self.kernel_fft = fft2(kernel, norm="ortho") * factor

    def conv(self, arr: np.ndarray):
        arr_fft = fft2(arr, norm="ortho")
        return ifft2(arr_fft * self.kernel_fft, norm="ortho").real
        # return fftconvolve(arr, self.kernel, mode="same")

    def conv_mat(self, arr: np.ndarray):
        """Same as conv. Intended for testing."""
        ret = np.zeros_like(arr, dtype=float).flatten()
        for index, (x, y) in enumerate(zip(self.space.X.ravel(), self.space.Y.ravel())):
            ret[index] = np.sum(self.weight_kernel(self.space.dist(x, y)) * arr)
        return ret.reshape(self.space.X.shape) * self.space.x_step * self.space.y_step

    def rhs(self, t, u):
        return -u + self.conv(self.firing_rate(u))


class CoombesSchmidtBojakBessel:
    """The Bessel function based kernel used in
    Coombes Schmidt Bojak 2012.
    """

    def __init__(self, beta, gamma):
        self.beta = beta
        self.gamma = gamma

    def __call__(self, r):
        r = r.copy()
        zero_mask = (r <= 0)
        r[zero_mask] = 1e-100
        return (
            2 / (3 * np.pi)
            * (
                k0(r) - k0(2 * r)
                - 1 / self.gamma * (k0(self.beta * r) - k0(2 * self.beta * r))
            )
        )


if __name__ == "__main__":
    from kernels import Gaussian, laterally_inhibitory_weight_kernel
    from firing_rate import Sigmoid, HermiteBump, Heaviside

    width = 60
    # width = 140
    x_linspace_params = (-width / 2, width / 2, 601)
    y_linspace_params = x_linspace_params

    t0, tf = 0, 100
    # t0, tf = 0, 80
    delta_t = 1e-1

    space = SpaceDomain(*x_linspace_params, *y_linspace_params)

    threshold = 0.1
    # threshold = 0.115

    nf = NeuralField(
        space=space,
        # firing_rate=Heaviside(threshold=threshold),
        firing_rate=HermiteBump(threshold=threshold, radius=0.1, order=3),
        # firing_rate=Sigmoid(threshold=0.2, gain=20),
        weight_kernel=laterally_inhibitory_weight_kernel,
        # weight_kernel=CoombesSchmidtBojakBessel(beta=0.5, gamma=4)
    )

    time = TimeDomain_Start_Stop_MaxSpacing(t0, tf, delta_t)
    solver = TqdmWrapper(RK4())

    u0 = Gaussian(sigma=1.0).radial(np.sqrt(space.X**2 + space.Y**2) + .5*np.cos(4*np.arctan2(space.Y, space.X)))
    # u0 = np.zeros_like(space.X).flatten()
    # for index, (x, y) in enumerate(zip(space.X.ravel(), space.Y.ravel())):
    #     r = np.sqrt(x**2 + y**2)
    #     theta = np.arctan2(y, x)
    #     if r < 12 + 0.5*np.cos(4*theta):
    #         u0[index] = 1.0
    # u0 = u0.reshape(space.X.shape)

    mesh = plt.pcolormesh(space.X, space.Y, u0, cmap="jet", vmin=-1, vmax=2)
    # mesh = plt.pcolormesh(space.X, space.Y, u0, cmap="jet", vmin=-.14, vmax=.4)
    plt.axis("equal")
    plt.colorbar()

    for u in solver.solution_generator(u0, nf.rhs, time):
        mesh.set_array(u)
        plt.draw()
        plt.pause(1e-3)
