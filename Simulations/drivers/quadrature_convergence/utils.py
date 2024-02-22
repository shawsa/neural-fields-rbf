import numpy as np
from numpy.polynomial import Polynomial as nppoly
import pickle
import sympy as sym
from rbf.points import UnitSquare

CART_STENCILS = "data/cartesian_stencil_sizes.pickle"
with open(CART_STENCILS, "rb") as f:
    cart_sizes = pickle.load(f)
HEX_STENCILS = "data/hex_stencil_sizes.pickle"
with open(HEX_STENCILS, "rb") as f:
    hex_sizes = pickle.load(f)


class HermiteBump:
    def __init__(
        self,
        *,
        order: int,
        radius: (float | sym.Rational),
        r: sym.Symbol = sym.symbols("r"),
    ):
        self.r = r
        self.radius = radius
        self.radius_float = float(radius)
        self.order = order
        self.init_poly()

    def init_poly(self):
        coeffs = [-1] + [0] * self.order
        for i in range(self.order):
            for j in range(self.order, 0, -1):
                coeffs[j] = coeffs[j] - coeffs[j - 1]
        for i in range(self.order):
            for j in range(self.order, i, -1):
                coeffs[j] = coeffs[j] - coeffs[j - 1]
        poly = 1
        for p, c in enumerate(coeffs):
            poly += c * self.r ** (self.order + 1) * (self.r - 1) ** p
        self.poly_sym = poly.subs(self.r, self.r / self.radius)
        self.poly_coeffs = [
            float(c) for c in sym.Poly(self.poly_sym).all_coeffs()
        ]
        self.poly_full = nppoly(self.poly_coeffs[::-1])

    def profile(self, rs: np.ndarray[float]):
        return np.heaviside(self.radius_float - rs, 0) * self.poly_full(rs)

    @property
    def deg(self):
        return 2 * order + 1

    def __call__(self, x: np.ndarray[float], y: np.ndarray[float]) -> np.ndarray[float]:
        rs = np.sqrt(x**2 + y**2)
        ret = 0
        for c in self.poly_coeffs:
            ret *= rs
            ret += c
        ret *= np.heaviside(self.radius_float - rs, 0)
        return ret

    def integrate(self):
        """Returns the integral of the Bump."""
        theta = sym.symbols("\\theta")
        return 4 * float(
            sym.integrate(
                sym.integrate(self.r * self.poly_sym, (self.r, 0, self.radius)),
                (theta, 0, sym.pi / 2),
            )
        )


class Gaussian:
    def __init__(
        self,
        standard_deviation: float,
    ):
        self.sd = standard_deviation
        x, y = sym.symbols("x y")
        self.vars = (x, y)
        self.sym = sym.exp(-(x**2 + y**2) / self.sd**2)
        self.numeric = sym.lambdify((x, y), self.sym)

    def __call__(self, x: np.ndarray[float], y: np.ndarray[float]) -> np.ndarray[float]:
        return self.numeric(x, y)

    def integrate(self):
        """Returns the integral of the Bump."""
        r, theta = sym.symbols("r \\theta")
        x, y = self.vars
        return 4 * float(sym.integrate(sym.integrate(self.sym, (x, 0, 1)), (y, 0, 1)))


def cartesian_grid(n: int):
    """Return a cartesian grid with approximately n points."""
    n_sqrt = int(np.ceil(np.sqrt(n)))
    X, Y = np.meshgrid(np.linspace(0, 1, n_sqrt), np.linspace(0, 1, n_sqrt))
    points = np.array([X.flatten(), Y.flatten()]).T
    return points


def hex_grid(n: int):
    """Return a hexagonal grid with approximately n points."""
    n_x = int(np.ceil(np.sqrt(n / np.sqrt(3))))
    n_y = int(np.ceil(n_x * np.sqrt(3) / 2))
    h_x = 1 / (n_x - 1) / 2
    h_y = 1 / (n_y - 1) / 2
    X, Y = np.meshgrid(np.linspace(0, 1, n_x), np.linspace(0, 1, n_y))
    X_inner, Y_inner = np.meshgrid(
        np.linspace(h_x, 1 - h_x, n_x - 1), np.linspace(h_y, 1 - h_y, n_y - 1)
    )
    X = np.append(X, X_inner)
    Y = np.append(Y, Y_inner)
    points = np.array([X.flatten(), Y.flatten()]).T
    return points


def random_points(n: int, verbose=False):
    """
    Return an set of roughly equally distributed points in [0, 1]^2.
    Uses rbf.points.unit_square.UnitSquare.
    """
    return UnitSquare(n, verbose=verbose).points


def _smallest_greater_than(k_min, sizes):
    for k in sizes:
        if k >= k_min:
            break
    return k


def cart_stencil_min(k_min: int) -> int:
    """
    Returns the smallest stencil size bigger than k_min
    for which a Cartesian grid has unambiguous stencils.
    """
    return _smallest_greater_than(k_min, cart_sizes)


def hex_stencil_min(k_min: int) -> int:
    """
    Returns the smallest stencil size bigger than k_min
    for which a hex grid has unambiguous stencils.
    """
    return _smallest_greater_than(k_min, hex_sizes)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test = HermiteBump(order=5, radius=sym.Rational(1, 5))

    def deriv(order, x):
        return test.poly_sym.diff(test.r, order).subs(test.r, x)

    for order in range(test.order + 2):
        print(f"{order=}")
        print(f"\t{deriv(order, 0)}\t{deriv(order, test.radius)}")

    plt.figure("Order 5 bump")
    side = np.linspace(0, 1, 2001)
    X, Y = np.meshgrid(side, side)
    plt.pcolormesh(X, Y, test(X - 0.5, Y - 0.5))
    plt.colorbar()
    plt.axis("equal")

    radius = 0.3
    rs = np.linspace(0, 1.2 * radius, 401)

    plt.figure("Some Profiles")
    for order in range(10):
        foo = HermiteBump(order=order, radius=radius)
        plt.plot(rs, foo.profile(rs), label=f"deg={foo.deg}")
    plt.legend()
    plt.ylim(-0.1, 1.1)
