from collections import namedtuple
import matplotlib.pyplot as plt
from min_energy_points import UnitSquare
import numpy as np
import numpy.linalg as la
from rbf.interpolate import Interpolator, Stencil
from rbf.rbf import RBF
from typing import Union


class UnscaledStencil(Stencil):
    def __init__(
        self,
        points,
        center=None,
    ):
        super().__init__(points, center=center)
        self.scale_factor = 1.0
        self.scaled_points = points.copy() - self.center


class PHSEvenShape(RBF):
    def __init__(self, order: int, shape: float, tol=1e-14):
        assert isinstance(order, int)
        assert order > 0
        assert order % 2 == 0
        self.order = order
        assert shape > 0
        self.shape = shape
        self.tol = 1e-14

    def __repr__(self):
        return f"$(cr)^{{{self.order}}}\\log(cr)$"

    def __call__(self, r):
        if isinstance(r, Union[float, int]):
            return self._scalar_call(r)
        if isinstance(r, np.ndarray):
            return self._array_call(r)
        raise TypeError()

    def _array_call(self, r: np.ndarray[float]):
        zero_mask = np.abs(self.shape * r) < self.tol
        ret = np.empty_like(r)
        ret[zero_mask] = 0
        ret[~zero_mask] = (self.shape * r[~zero_mask]) ** self.order * np.log(
            self.shape * r[~zero_mask]
        )
        return ret

    def _scalar_call(self, r: float):
        if abs(r) < 1e-14:
            return 0
        return (self.shape * r) ** self.order * np.log(self.shape * r)

    def dr(self, r):
        raise NotImplementedError

    def d2r(self, r):
        raise NotImplementedError

    def dr_div_r(self, r):
        raise NotImplementedError


class PHSOddShape(RBF):
    def __init__(self, order: int, shape: float, tol=1e-14):
        assert isinstance(order, int)
        assert order > 0
        assert order % 2 == 1
        self.order = order
        assert shape > 0
        self.shape = shape
        self.tol = 1e-14

    def __repr__(self):
        return f"$(cr)^{{{self.order}}}$"

    def __call__(self, r):
        return (self.shape * r) ** self.order

    def dr(self, r):
        raise NotImplementedError

    def d2r(self, r):
        raise NotImplementedError

    def dr_div_r(self, r):
        raise NotImplementedError


def PHSShape(order: int, shape: float):
    assert isinstance(order, int)
    assert order > 0
    if order % 2 == 0:
        return PHSEvenShape(order, shape)
    if order % 2 == 1:
        return PHSOddShape(order, shape)
    raise ValueError()


def test_func(x, y):
    x0, y0 = 0.5, 0.5
    # x0, y0 = 0, 0
    return np.exp(-((x - x0) ** 2) - (y - y0) ** 2)
    # return np.sin(x) * np.cos(y)
    # return x**2 * y
    # return (x**2 + y**2)**3


phs_order = 4
poly_deg = 2
N = 50
np.random.seed(0)
points = UnitSquare(N, verbose=True).points
stencil = UnscaledStencil(points, center=np.r_[0, 0])
# stencil = Stencil(points, center=np.r_[0, 0])

test_rbf = PHSEvenShape(phs_order, shape=2)


def phs_test(X, Y):
    Z = np.c_[X.ravel(), Y.ravel()]
    return test_rbf(la.norm(Z - points[0], axis=1)).reshape(X.shape)


def poly_test(X, Y):
    x, y = points[0]
    return ((X - x) ** 2 + (Y - y) ** 2) ** (phs_order // 2)


X, Y = np.meshgrid(*2 * (np.linspace(0, 1, 200),))
Z = np.c_[X.ravel(), Y.ravel()]

foo = test_func

Fs = foo(X, Y)

c1, c2 = 1, 2

approx1 = Interpolator(
    stencil=stencil,
    fs=foo(*points.T),
    rbf=PHSShape(phs_order, c1),
    poly_deg=poly_deg,
)

plt.figure("interpolant")
plt.pcolormesh(X, Y, approx1(Z).reshape(X.shape), cmap="jet")
plt.plot(*points.T, "k.")
plt.colorbar()

plt.figure("Error")
plt.pcolormesh(X, Y, approx1(Z).reshape(X.shape) - Fs, cmap="jet")
plt.plot(*points.T, "k.")
plt.colorbar()
plt.title("$s_1 - f$")

plt.figure("interp diff")
approx2 = Interpolator(
    stencil=stencil,
    fs=foo(*points.T),
    rbf=PHSShape(phs_order, c2),
    poly_deg=poly_deg,
)
plt.pcolormesh(X, Y, (approx1(Z) - approx2(Z)).reshape(X.shape), cmap="jet")
plt.plot(*points.T, "k.")
plt.colorbar()
plt.title("$s_1 - s_2$")

print(approx1.rbf_weights / approx2.rbf_weights)
print(approx1.poly_weights / approx2.poly_weights)
