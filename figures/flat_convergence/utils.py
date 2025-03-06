from abc import ABC, abstractmethod
from math import ceil, floor, sqrt
import numpy as np
from numpy.polynomial import Polynomial as nppoly
import pickle
from rbf.points import UnitSquare
from rbf.quadrature import LocalQuad
import sympy as sym
from tqdm import tqdm


try:
    CART_STENCILS = "data/cartesian_stencil_sizes.pickle"
    with open(CART_STENCILS, "rb") as f:
        cart_sizes = pickle.load(f)
    HEX_STENCILS = "data/hex_stencil_sizes.pickle"
    with open(HEX_STENCILS, "rb") as f:
        hex_sizes = pickle.load(f)
except FileNotFoundError:
    print("Failed to load stencil sizes dict.")


class TestFunction(ABC):
    @abstractmethod
    def __call__(self, x: np.ndarray[float], y: np.ndarray[float]) -> np.ndarray[float]:
        raise NotImplementedError

    @abstractmethod
    def integrate(self):
        raise NotImplementedError


class HermiteBump(TestFunction):
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
        self.poly_coeffs = [float(c) for c in sym.Poly(self.poly_sym).all_coeffs()]
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


class Gaussian(TestFunction):
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


class PeriodicTile(TestFunction):
    def __init__(self, func: TestFunction):
        assert func(0.5, 0.5) < 1e-17
        self.func = func

    def __call__(self, x: np.ndarray[float], y: np.ndarray[float]) -> np.ndarray[float]:
        ret = np.zeros_like(x)
        for x_0 in [-1, 0, 1, 2]:
            for y_0 in [-1, 0, 1, 2]:
                ret += self.func(x - x_0, y - y_0)
        return ret

    def integrate(self):
        return self.func.integrate()


def quad_test(
    qf: LocalQuad,
    func: PeriodicTile,
    X: np.ndarray[float],
    Y: np.ndarray[float],
    verbose=False,
    tqdm_kwargs={},
) -> np.ndarray[float]:
    exact = func.integrate()
    Z = np.zeros_like(X).flatten()
    if verbose:

        def wrapper(x, **kwargs):
            return tqdm(x, **kwargs, **tqdm_kwargs)

    else:

        def wrapper(x, **_):
            return x

    for index, (x0, y0) in wrapper(enumerate(zip(X.ravel(), Y.ravel())), total=len(Z)):
        fs = func(qf.points[:, 0] - x0, qf.points[:, 1] - y0)
        approx = qf.weights @ fs
        error = (approx - exact) / exact
        Z[index] = error
    Z = Z.reshape(X.shape)
    return Z


def covering_dist(
    qf: LocalQuad,
    X: np.ndarray[float],
    Y: np.ndarray[float],
    workers=1,
) -> np.ndarray[float]:
    points = np.array([X.ravel(), Y.ravel()]).T
    return qf.kdt.query(points, workers=workers)[0].reshape(X.shape)


def cartesian_grid(n: int):
    """Return a cartesian grid with approximately n points."""
    n_sqrt = int(np.ceil(np.sqrt(n)))
    X, Y = np.meshgrid(np.linspace(0, 1, n_sqrt), np.linspace(0, 1, n_sqrt))
    points = np.array([X.flatten(), Y.flatten()]).T
    return points


def hex_grid(N: int):
    N_inner_approx = N - ceil(sqrt(N) * (1 + sqrt(3)) / 2) * 4
    A, B, C = 2 * sqrt(3), -(1 + sqrt(3)), 1 - N_inner_approx
    ny_approx = (-B + sqrt(B**2 - 4 * A * C)) / (2 * A)
    ny = int(ceil(ny_approx))
    nx = int(ceil(sqrt(3) * ny_approx))
    X, Y = np.meshgrid(np.arange(0, nx, 1), np.arange(0, ny) * np.sqrt(3))
    h = 1 / max(np.max(X), np.max(Y))
    X2 = X[:-1, :-1] + 0.5
    Y2 = Y[:-1, :-1] + np.sqrt(3) / 2
    inner_points = np.array(
        [[*X.flatten(), *X2.flatten()], [*Y.flatten(), *Y2.flatten()]]
    ).T
    inner_points *= h * (1 - h)
    inner_points[:, 0] += (1 - np.max(inner_points[:, 0])) / 2
    inner_points[:, 1] += (1 - np.max(inner_points[:, 1])) / 2
    N_inner = len(inner_points)
    n_side = ceil((N - N_inner) / 4)
    N = 4 * n_side + N_inner

    points = np.zeros((N, 2))
    side = np.linspace(0, 1, n_side, endpoint=False)
    # bottom
    points[:n_side, 0] = side
    # right
    points[n_side : 2 * n_side, 0] = 1
    points[n_side : 2 * n_side, 1] = side
    # top
    points[2 * n_side : 3 * n_side, 0] = 1 - side
    points[2 * n_side : 3 * n_side, 1] = 1
    # left
    points[3 * n_side : 4 * n_side, 1] = 1 - side

    points[4 * n_side :] = inner_points

    return points


def random_points(n: int, verbose=False, tqdm_kwargs={}):
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


def poly_stencil_min(deg: int) -> int:
    """
    Returns the number of polynomial basis terms in 2D.
    The stencil size must be at least this big for the interpolation matrix
    to be unisolvent in theory.
    """
    return ((deg + 1) * (deg + 2)) // 2


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

    points = hex_grid(4_000)
    plt.plot(*points.T, "k.")
    plt.axis("equal")
    print(np.max(points, axis=0))
