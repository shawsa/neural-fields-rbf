from itertools import product
import numpy as np
import sympy as sym
from numpy.fft import fft2, ifft2


class ManufacturedSigmoidFiring:
    def __init__(
        self,
        *,
        threshold: float,
        gain: float,
    ):
        self.threshold = threshold
        self.gain = gain

    def __call__(self, u: np.ndarray[float]) -> np.ndarray[float]:
        return 1 / (1 + np.exp(-self.gain * (u - self.threshold)))

    def inv(self, f: np.ndarray[float], log=np.log) -> np.ndarray[float]:
        return -log(1 / f - 1) / self.gain + self.threshold


def gauss(X, Y, x0, y0, sigma, pi=np.pi, exp=np.exp):
    return (
        1
        / (2 * pi * sigma**2)
        * exp(-((X - x0) ** 2 + (Y - y0) ** 2) / (2 * sigma**2))
    )


def gauss_conv(X, Y, x0, y0, sigma0, x1, y1, sigma1):
    return gauss(X, Y, x0 + x1, y0 + y1, np.sqrt(sigma0**2 + sigma1**2))


class ManufacturedSolution:
    """Solution is f^{-1}[Gaussian(x, y, t) + epsilon]."""

    def __init__(
        self,
        *,
        weight_kernel_sd: float,
        threshold: float,
        gain: float,
        epsilon: float,
        solution_sd: float,
        path_radius: float,
    ):
        self.weight_kernel_sd = weight_kernel_sd
        self.firing = ManufacturedSigmoidFiring(threshold=threshold, gain=gain)
        self.solution_sd = solution_sd
        self.path_radius = path_radius
        self.epsilon = epsilon

        x, y, t = sym.symbols("x y t")
        center_x = self.path_radius * sym.cos(t)
        center_y = self.path_radius * sym.sin(t)

        self.center_x_num = sym.lambdify(t, center_x)
        self.center_y_num = sym.lambdify(t, center_y)

        self.sol = self.firing.inv(
            gauss(x, y, center_x, center_y, solution_sd, pi=sym.pi, exp=sym.exp)
            + self.epsilon,
            log=sym.log,
        )

        self.sol_dt = self.sol.diff(t)

        self.sol_num = sym.lambdify((x, y, t), self.sol)
        self.sol_dt_num = sym.lambdify((x, y, t), self.sol_dt)

    def exact(self, X: np.ndarray[float], Y: np.ndarray[float], t: float):
        return self.sol_num(X, Y, t)

    def dt(self, X: np.ndarray[float], Y: np.ndarray[float], t: float):
        return self.sol_dt_num(X, Y, t)

    def rhs(self, X, Y, t):
        return (
            self.dt(X, Y, t)
            + self.exact(X, Y, t)
            - gauss_conv(
                X,
                Y,
                0,
                0,
                self.weight_kernel_sd,
                self.center_x_num(t),
                self.center_y_num(t),
                self.solution_sd,
            )
            - self.epsilon
        )


class ManufacturedSolutionPeriodic:
    """Solution is f^{-1}[Gaussian(x, y, t) + epsilon]."""

    def __init__(
        self,
        *,
        weight_kernel_sd: float,
        threshold: float,
        gain: float,
        epsilon: float,
        solution_sd: float,
        path_radius: float,
        period: float,
        num_tiles: int = 1,
    ):
        self.weight_kernel_sd = weight_kernel_sd
        self.firing = ManufacturedSigmoidFiring(threshold=threshold, gain=gain)
        self.solution_sd = solution_sd
        self.path_radius = path_radius
        self.epsilon = epsilon
        self.period = period
        self.num_tiles = num_tiles

        x, y, t = sym.symbols("x y t")
        center_x = self.path_radius * sym.cos(t)
        center_y = self.path_radius * sym.sin(t)

        self.center_x_num = sym.lambdify(t, center_x)
        self.center_y_num = sym.lambdify(t, center_y)

        self.sol = self.firing.inv(
            sum(
                gauss(
                    x,
                    y,
                    center_x + self.period * x_tile_index,
                    center_y + self.period * y_tile_index,
                    solution_sd,
                    pi=sym.pi,
                    exp=sym.exp,
                )
                for x_tile_index, y_tile_index in product(
                    *2 * (range(-self.num_tiles, self.num_tiles + 1),)
                )
            )
            + self.epsilon,
            log=sym.log,
        )

        self.sol_dt = self.sol.diff(t)

        self.sol_num = sym.lambdify((x, y, t), self.sol)
        self.sol_dt_num = sym.lambdify((x, y, t), self.sol_dt)

    def exact(self, X: np.ndarray[float], Y: np.ndarray[float], t: float):
        return self.sol_num(X, Y, t)

    def dt(self, X: np.ndarray[float], Y: np.ndarray[float], t: float):
        return self.sol_dt_num(X, Y, t)

    def rhs(self, X, Y, t):
        return (
            self.dt(X, Y, t)
            + self.exact(X, Y, t)
            - sum(
                gauss_conv(
                    X,
                    Y,
                    x_tile_index * self.period,
                    y_tile_index * self.period,
                    self.weight_kernel_sd,
                    self.center_x_num(t),
                    self.center_y_num(t),
                    self.solution_sd,
                )
                for x_tile_index, y_tile_index in product(
                    *2 * (range(-self.num_tiles, self.num_tiles + 1),)
                )
            )
            - self.epsilon
        )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # test Gaussian Convolve

    def roll(F, x_shift, y_shift):
        return np.roll(np.roll(F, x_shift, axis=0), y_shift, axis=1)

    def conv(F1, F2, x_step, y_step):
        x_shift = len(F2[0]) // 2 + 1
        y_shift = len(F2) // 2 + 1
        F2 = roll(F2, x_shift, y_shift)
        F1_hat = fft2(F1, norm="ortho")
        F2_hat = fft2(F2, norm="ortho")
        factor = x_step * y_step * np.sqrt(len(F1)) * np.sqrt(len(F1[:, 0]))
        return ifft2(F1_hat * F2_hat, norm="ortho").real * factor

    x0, y0, sigma0 = 0, 0, 1
    x1, y1, sigma1 = 0.5, 0.1, 0.2

    xs = np.linspace(-20, 20, 401)
    ys = np.linspace(-20, 20, 401)
    X, Y = np.meshgrid(xs, ys)

    g0 = gauss(X, Y, x0, y0, sigma0)
    g1 = gauss(X, Y, x1, y1, sigma1)
    gconv = gauss_conv(X, Y, x0, y0, sigma0, x1, y1, sigma1)

    x_step = (xs[-1] - xs[0]) / (len(xs) - 1)
    y_step = (ys[-1] - ys[0]) / (len(ys) - 1)
    gconv_test = conv(g0, g1, x_step, y_step)

    plt.figure("Gaussian Convolve test.")
    err_max = np.max(np.abs(gconv - gconv_test))
    plt.pcolormesh(
        X, Y, gconv - gconv_test, cmap="seismic", vmin=-err_max, vmax=err_max
    )
    plt.axis("equal")
    plt.colorbar()

    # test manufactured solution

    mf = ManufacturedSolution(
        weight_kernel_sd=1,
        firing_rate_threshold=0.2,
        firing_rate_radius=0.2,
        solution_sd=1,
        path_radius=5,
    )

    ts = [0, 1, 2]
    ts_path = np.linspace(0, 2 * np.pi, 201)

    fig, axes = plt.subplots(
        len(ts), 3, figsize=(10, 10), label="manufactured solution"
    )
    for col, label in enumerate(["u", "u_t", "F(t)"]):
        axes[0][col].set_title(label)
    for row, t in enumerate(ts):
        axes[row][0].pcolormesh(
            X, Y, mf.exact(X, Y, t), cmap="seismic", vmin=-0.3, vmax=0.3
        )
        axes[row][1].pcolormesh(
            X, Y, mf.dt(X, Y, t), cmap="seismic", vmin=-2.7, vmax=2.7
        )
        axes[row][2].pcolormesh(X, Y, mf.rhs(X, Y, t), cmap="seismic", vmin=-3, vmax=3)
        for col in range(3):
            axes[row][col].plot(
                mf.center_x_num(ts_path), mf.center_y_num(ts_path), "g-"
            )
