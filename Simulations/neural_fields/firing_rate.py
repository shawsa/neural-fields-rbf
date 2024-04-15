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
