import numpy as np
from .rbf import RBF

right_triangle_integrate_dict = {}


def test_dict():
    for expr_str, func in right_triangle_integrate_dict.items():
        assert isinstance(func(1.0, 2.0), float)


def validate(func):
    def right_triangle_integrate(a: np.ndarray[float], b: np.ndarray[float]):
        ret = np.zeros_like(a)
        mask = a > 1e-10
        if b.ndim != 0:
            b = b[mask]
        ret[mask] = func(a[mask], b)
        return ret
        # if abs(a) < 1e-10:
        #     return 0
        # return func(a, b)

    return right_triangle_integrate


def get_right_triangle_integral_function(rbf: RBF):
    expr_str = str(rbf)
    if expr_str in right_triangle_integrate_dict.keys():
        return validate(right_triangle_integrate_dict[expr_str])
    raise ValueError(f"Error: rbf {expr_str} not implemented in quadrature library")


# GENERATED CODE

# r**3
right_triangle_integrate_dict["r**3"] = lambda a, b:(1/40)*a**2*(3*a**3*np.arcsinh(b/a) + 5*a**2*b*np.sqrt(1 + b**2/a**2) + 2*b**3*np.sqrt(1 + b**2/a**2))

# r**5
right_triangle_integrate_dict["r**5"] = lambda a, b:(1/336)*a**2*(15*a**5*np.arcsinh(b/a) + 33*a**4*b*np.sqrt(1 + b**2/a**2) + 26*a**2*b**3*np.sqrt(1 + b**2/a**2) + 8*b**5*np.sqrt(1 + b**2/a**2))

# r**7
right_triangle_integrate_dict["r**7"] = lambda a, b:(1/3456)*a**2*(105*a**7*np.arcsinh(b/a) + 279*a**6*b*np.sqrt(1 + b**2/a**2) + 326*a**4*b**3*np.sqrt(1 + b**2/a**2) + 200*a**2*b**5*np.sqrt(1 + b**2/a**2) + 48*b**7*np.sqrt(1 + b**2/a**2))

# r**9
right_triangle_integrate_dict["r**9"] = lambda a, b:(1/14080)*a**2*(315*a**9*np.arcsinh(b/a) + 965*a**8*b*np.sqrt(1 + b**2/a**2) + 1490*a**6*b**3*np.sqrt(1 + b**2/a**2) + 1368*a**4*b**5*np.sqrt(1 + b**2/a**2) + 656*a**2*b**7*np.sqrt(1 + b**2/a**2) + 128*b**9*np.sqrt(1 + b**2/a**2))

# r**11
right_triangle_integrate_dict["r**11"] = lambda a, b:(1/199680)*a**2*(3465*a**11*np.arcsinh(b/a) + 11895*a**10*b*np.sqrt(1 + b**2/a**2) + 22790*a**8*b**3*np.sqrt(1 + b**2/a**2) + 27848*a**6*b**5*np.sqrt(1 + b**2/a**2) + 20016*a**4*b**7*np.sqrt(1 + b**2/a**2) + 7808*a**2*b**9*np.sqrt(1 + b**2/a**2) + 1280*b**11*np.sqrt(1 + b**2/a**2))

# r**2*log(r)
right_triangle_integrate_dict["r**2*log(r)"] = lambda a, b:(1/144)*a*(12*a**3*(-1j*np.log((-a - 1j*b)/a) + 1j*np.log((a - 1j*b)/a) - np.pi) + 3*a**2*b*(6*np.log((-a**2 - b**2)/a**2) - 11) - b**3*(7 - 6*np.log((-a**2 - b**2)/a**2)) + 12*b*(3*a**2 + b**2)*np.log(1j*a))

# r**4*log(r)
right_triangle_integrate_dict["r**4*log(r)"] = lambda a, b:(1/900)*a*(40*a**5*(-1j*np.log((-a - 1j*b)/a) + 1j*np.log((a - 1j*b)/a) - np.pi) + 15*a**4*b*(5*np.log((-a**2 - b**2)/a**2) - 7) + 10*a**2*b**3*(5*np.log((-a**2 - b**2)/a**2) - 4) + b**5*(15*np.log((-a**2 - b**2)/a**2) - 11) + 10*b*(15*a**4 + 10*a**2*b**2 + 3*b**4)*np.log(1j*a))
