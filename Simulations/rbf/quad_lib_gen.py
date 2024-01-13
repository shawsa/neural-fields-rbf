"""
Generate functions for exactly integrating an rbf over a right triangle.

Symbolic expressions are converted to python code and appeneded to the
quadrature library file. It is recommended to delete the generated code section
before running this script.
"""
from os import system
from sympy import symbols, sqrt, integrate, pycode
from sympy.parsing import sympy_parser as parser

x, y, r, a, b = symbols("x y r a b")  # , real=True)
r_subs = sqrt(x**2 + y**2)


DICT_NAME = "right_triangle_integrate_dict"
LIB_FILE = "quad_lib.py"
GEN_CODE_BANNER = "# GENERATED CODE"


def generate(expr_str: str):
    phi = parser.parse_expr(expr_str)
    triangle_integral = integrate(
        integrate(phi.subs(r, r_subs), (y, 0, b / a * x)), (x, 0, a)
    ).simplify()
    assert a in triangle_integral.free_symbols
    assert b in triangle_integral.free_symbols
    assert len(triangle_integral.free_symbols) == 2
    return DICT_NAME + f'["{expr_str}"] = lambda a, b:' + pycode(triangle_integral)


def add_to_library(expr_str: str):
    print(f"Adding {expr_str} to library...")
    line = generate(expr_str)
    with open(LIB_FILE, "a") as f:
        f.write(f"\n# {expr_str}\n")
        f.write(line + "\n")


def clear_library():
    print("Clearing library...")
    with open(LIB_FILE, 'r') as f:
        lines = f.readlines()
    with open(LIB_FILE, 'w') as f:
        for line in lines:
            f.write(line)
            if GEN_CODE_BANNER in line:
                break


def append_test():
    print("Appending dunder main test code...")
    with open(LIB_FILE, "a") as f:
        f.write('\n\nif __name__ == "__main__":\n')
        f.write(" "*4 + "test_dict()")


def make_black():
    print("Attempting to code format using black...")
    cmd = f"black {LIB_FILE}"
    system(cmd)


if __name__ == "__main__":
    clear_library()
    for n in range(3, 12, 2):
        expr_str = f"r**{n}"
        add_to_library(expr_str)

    for n in range(2, 11, 2):
        expr_str = f"r**{n}*log(r)"
        add_to_library(expr_str)

    append_test()
    make_black()
