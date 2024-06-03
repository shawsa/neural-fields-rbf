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


class ManufacturedHermiteFiring:
    """
    Smooth at threshold +/- radius for 3 derivatives.
    The same as HermiteBump with order = 3.
    Efficient and accurate computation of inverse (unlike HermiteBump).

    This didn't work. It should be removed from here, but I'm not sure where to put it.
    """

    def __init__(
        self,
        *,
        threshold: float,
        radius: float,
    ):
        self.radius = radius
        self.threshold = threshold

        self.lower = self.threshold - self.radius
        self.upper = self.threshold + self.radius

    def affine(self, r):
        return (r - self.lower) / (2 * self.radius)

    def affine_inv(self, r):
        return r * (2 * self.radius) + self.lower

    def foo(self, r):
        return r**3 * (6 * (r - 5 / 4) ** 2 + 5 / 8)

    def dfoo(self, r):
        return 3 * r**2 * (6 * (r - 5 / 4) ** 2 + 5 / 8) + r**3 * (12 * (r - 5 / 4))

    def newton_find(self, f):
        x = (f - 0.5) / self.dfoo(0.5) + 0.5
        for _ in range(10):
            x -= (self.foo(x) - f) / self.dfoo(x)
        return x

    def root_find(self, f):
        x = 1e-1
        for _ in range(10):
            x = (f / (6 * (x - 5 / 4) ** 2 + 5 / 8)) ** (1 / 3)
        return x

    def __call__(self, rs: np.ndarray[float]):
        ret = np.zeros_like(rs)
        ret[rs >= self.upper] = 1
        poly_mask = np.logical_and(rs > self.lower, rs < self.upper)
        ret[poly_mask] = self.foo(self.affine(rs[poly_mask]))
        return ret

    def foo_inv(self, ys: np.ndarray[float]):
        newton_cutoff = 7e-5  # Hand tuned - lower than this Newton becomes inaccurate.
        xs = np.empty_like(ys)
        # upper saturation
        mask = ys >= 1
        xs[mask] = 1
        # lower saturation
        mask = ys <= 0
        xs[mask] = 0
        # middle use newton
        mask = np.logical_and(ys >= newton_cutoff, ys <= 1 - newton_cutoff)
        xs[mask] = self.newton_find(ys[mask])
        # close to 0, use fixed point
        mask = np.logical_and(ys > 0.0, ys < newton_cutoff)
        xs[mask] = self.root_find(ys[mask])
        # close to 1, use fixed point
        mask = np.logical_and(ys < 1.0, ys > 1 - newton_cutoff)
        xs[mask] = 1 - self.root_find(1 - ys[mask])
        return xs

    def inv(self, ys: np.ndarray[float]):
        return self.affine_inv(self.foo_inv(ys))


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


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # test firing rate

    f = ManufacturedHermiteFiring(threshold=0.3, radius=0.1)

    xs = np.linspace(0, 1, 201)
    ys = f(xs)
    plt.figure("Firing rate inverse test.")
    plt.plot(xs, ys)
    plt.plot(f.inv(ys), ys)

    zs = np.linspace(f.lower, f.upper, 2001)
    err = zs - f.inv(f(zs))
    plt.figure("Firing rate inverse error.")
    plt.semilogy(zs, np.abs(err))

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
