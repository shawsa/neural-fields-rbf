from sympy import symbols, integrate, expand_func, pycode
from itertools import product

LIB_FILE = "poly_triangle_integrate_lib.py"

MAX_DEG = 3

x, y, Lx, Ly, Mx, My, Rx, Ry = symbols("x y Lx Ly Mx My Rx Ry", real=True)
L = (Lx, Ly)
M = (Mx, My)
R = (Rx, Ry)


def seg(x, A, B):
    Ax, Ay = A
    Bx, By = B
    return Ay + (x - Ax) * (By - Ay) / (Bx - Ax)


with open(LIB_FILE, "a") as f:
    for n, m in product(range(MAX_DEG), range(MAX_DEG)):
        print(f"Generating {(n, m)=}...", end="")
        p = x**n * y**m
        arg1 = integrate(p, (y, seg(x, L, M), seg(x, L, R)))
        summand1 = integrate(arg1, (x, Lx, Mx))
        arg2 = integrate(p, (y, seg(x, M, R), seg(x, L, R)))
        summand2 = integrate(arg2, (x, Mx, Rx))
        result = summand1 + summand2
        print("Success")
        code = (
            f"integral_dict[({n}, {m})] = lambda Lx, Ly, Mx, My, Rx, Ry: "
            + pycode(result)
            + "\n"
        )
        f.write(code)
