from dataclasses import dataclass
from itertools import product
import numpy as np
import numpy.linalg as la
from numpy.fft import fft2, ifft2
from typing import Callable


class SpaceDomain2D:
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


@dataclass
class Parameters:
    mu: float
    alpha: float
    gamma: float

    @property
    def beta(self):
        return 1 / self.gamma - 1


class NeuralField2D:
    def __init__(
        self,
        space: SpaceDomain2D,
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
    x_linspace_params = (-1, 1, 201)
    y_linspace_params = (-1, 1, 201)
    space = SpaceDomain2D(*x_linspace_params, *y_linspace_params)

    def firing_rate(u):
        return 1 / (1 + np.exp(u))

    def weight_kernel(r):
        return np.exp(-r)

    nf = NeuralField2D(
        space=space,
        firing_rate=firing_rate,
        weight_kernel=weight_kernel,
        params=Parameters(mu=1.0, alpha=20.0, gamma=0.2),
    )
