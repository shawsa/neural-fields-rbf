import numpy as np
from numpy.polynomial import Polynomial as nppoly
import sympy as sym


class Heaviside:
    def __init__(self, threshold):
        self.threshold = threshold

    def __call__(self, u):
        return np.heaviside(u - self.threshold, 0.5)


class Sigmoid:
    def __init__(self, threshold=0.2, gain=20):
        self.threshold = threshold
        self.gain = gain

    def __call__(self, u):
        return 1 / (1 + np.exp(-self.gain * (u - self.threshold)))

    def inv(self, f):
        return self.threshold - 1 / self.gain * np.log(1 / f - 1)


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

        self.lower = self.threshold - self.radius
        self.upper = self.threshold + self.radius
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
        self.poly_shift = poly.copy()
        affine = (self.r - self.threshold) / (2 * self.radius) + 0.5
        self.poly_sym = poly.subs(self.r, affine).expand()
        self.poly_coeffs = [float(c) for c in sym.Poly(self.poly_sym).all_coeffs()]
        self.poly_full = nppoly(self.poly_coeffs[::-1])
        self.poly = sym.lambdify(self.r, self.poly_sym)
        self.poly_diff = sym.lambdify(self.r, self.poly_sym.diff(self.r))

    def __call__(self, rs: np.ndarray[float]):
        ret = np.zeros_like(rs)
        ret[rs >= self.upper] = 1
        poly_mask = np.logical_and(rs > self.lower, rs < self.upper)
        ret[poly_mask] = self.poly_full(rs[poly_mask])
        return ret

    def inv(self, fs: np.ndarray, iters=10):
        ret = np.zeros_like(fs)
        ret[fs <= 0] = self.lower
        ret[fs >= 1] = self.upper
        mask = np.logical_and(fs > 0, fs < 1)
        ys = fs[mask]

        xs = (ys - 0.5) / self.poly_diff(self.threshold) + self.threshold
        for _ in range(iters):
            xs -= (self.poly(xs) - ys) / self.poly_diff(xs)

        ret[mask] = xs

        return ret


class ManufacturedInvertable:
    """
    Smooth at threshold +/- radius for 3 derivatives.
    The same as HermiteBump with order = 3.
    Efficient and accurate computation of inverse (unlike HermiteBump).
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
            x -= (self.foo(x)-f) / self.dfoo(x)
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
        newton_cutoff = 0.02
        xs = np.empty_like(ys)
        # upper saturation
        mask = ys >= 1
        xs[mask] = 1
        # lower saturation
        mask = ys <= 0
        xs[mask] = 0
        # middle use newton
        mask = np.logical_and(ys >= newton_cutoff, ys <= 1-newton_cutoff)
        xs[mask] = self.newton_find(ys[mask])
        # close to 0, use fixed point
        mask = np.logical_and(ys > 0.0, ys < newton_cutoff)
        xs[mask] = self.root_find(ys[mask])
        # close to 1, use fixed point
        mask = np.logical_and(ys < 1.0, ys > 1-newton_cutoff)
        xs[mask] = 1 - self.root_find(1 - ys[mask])
        return xs

    def inv(self, ys: np.ndarray[float]):
        return self.affine_inv(self.foo_inv(ys))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # f = HermiteBump(threshold=0.3, radius=0.1, order=2)
    # f = ManufacturedInvertable(threshold=0.3, radius=0.1)
    f = ManufacturedInvertable(threshold=0.5, radius=0.5)

    xs = np.linspace(0, 1, 201)
    ys = f(xs)
    plt.plot(xs, ys)
    plt.plot(f.inv(ys), ys)

    zs = np.linspace(f.lower, f.upper, 2001)
    err = zs - f.inv(f(zs))
    plt.figure()
    plt.semilogy(zs, np.abs(err))
