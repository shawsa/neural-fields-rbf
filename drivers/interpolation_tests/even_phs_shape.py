from collections import namedtuple
from itertools import product
import matplotlib.pyplot as plt
from min_energy_points import UnitSquare
import numpy as np
from rbf.interpolate import Interpolator, Stencil
from rbf.rbf import RBF
from tqdm import tqdm
from typing import Union


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


def foo(x, y):
    # x0, y0 = 0.5, 0.5
    # return np.exp(-((x - x0) ** 2) - (y - y0) ** 2)
    # return np.sin(x) * np.cos(y)
    # return x**2 * y
    return (x**2 + y**2)**3


Result = namedtuple(
    "Result",
    (
        "phs_order",
        "shape",
        "poly_deg",
        "Loo_error",
    ),
)

phs_order = 6
poly_degs = list(range(-1, 5))
N = 200
np.random.seed(0)
points = UnitSquare(N, verbose=True).points
X, Y = np.meshgrid(*2 * (np.linspace(0, 1, 100),))
Fs = foo(X, Y)

shapes = np.linspace(0, 1, 41)[1:]
results = []
for c, poly_deg in tqdm(list(product(shapes, poly_degs))):
    rbf = PHSShape(phs_order, c)
    # rbf = PHS3Shape(c)

    approx = Interpolator(
        stencil=Stencil(points),
        fs=foo(*points.T),
        rbf=rbf,
        poly_deg=poly_deg,
    )

    my_err = Fs - approx(np.c_[X.ravel(), Y.ravel()]).reshape(X.shape)
    max_err = np.max(np.abs(my_err))
    results.append(
        Result(
            phs_order=rbf.order,
            shape=rbf.shape,
            poly_deg=poly_deg,
            Loo_error=max_err,
        )
    )


plt.figure()
for poly_deg in poly_degs:
    my_res = [
        res
        for res in results
        if res.phs_order == phs_order and res.poly_deg == poly_deg
    ]
    cs = [res.shape for res in my_res]
    errs = [res.Loo_error for res in my_res]
    plt.semilogy(cs, errs, ".-", label=f"{poly_deg=}")
plt.legend()
plt.xlabel("c")
plt.ylabel("max error")
plt.title(str(rbf))
plt.savefig("media/even_phs_shape.png")
