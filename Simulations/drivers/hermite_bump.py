import numpy as np
import sympy as sym


def sym_hermite_bump_poly(r, radius, smoothness: int):
    coeffs = [-1] + [0] * (smoothness - 1)
    for i in range(smoothness - 1):
        for j in range(smoothness - 1, 0, -1):
            coeffs[j] = coeffs[j] - coeffs[j - 1]
    for i in range(smoothness - 1):
        for j in range(smoothness - 1, i, -1):
            coeffs[j] = coeffs[j] - coeffs[j - 1]
    ret = 1
    for p, c in enumerate(coeffs):
        ret += c * r**smoothness * (r - 1) ** p
    return ret.subs(r, r / radius)


def hermite_bump(r, radius, smoothness: int):
    foo_r = sym_hermite_bump_poly(r, radius, smoothness)
    foo = sym.lambdify((r,), foo_r)
    return lambda x: foo(x) * np.heaviside(float(radius) - x, 0.5)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    rs = np.linspace(0, 1, 2001)
    radius = 0.7
    r = sym.symbols("r")

    plt.figure()
    for p in range(1, 10):
        foo = hermite_bump(r, radius, p)
        plt.plot(rs, foo(rs))
