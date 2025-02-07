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

NP_SUB = {
    "math." + func: "np." + func
    for func in [
        "log",
        "pi",
        "sqrt",
    ]
}

# careful of the order of replacement e.g. arcsinh before arcsin
NP_SUB_SPECIAL = {
    "math.asinh": "np.arcsinh",
    # "math.log": "np.log",
    "math.log": "complexlog",
    "math.sqrt": "np.sqrt",
}


def numpy_subs(code: str) -> str:
    for key, val in {**NP_SUB, **NP_SUB_SPECIAL}.items():
        code = code.replace(key, val)
    return code


def generate(phi):
    triangle_integral = integrate(
        integrate(phi.subs(r, r_subs), (y, 0, b / a * x)), (x, 0, a)
    ).simplify()
    assert a in triangle_integral.free_symbols
    assert b in triangle_integral.free_symbols
    assert len(triangle_integral.free_symbols) == 2
    return (
        DICT_NAME
        + f'["{str(phi)}"] = lambda a, b:('
        + numpy_subs(pycode(triangle_integral))
        + ").real"
    )


def add_to_library(phi):
    print(f"Adding {str(phi)} to library...")
    line = generate(phi)
    with open(LIB_FILE, "a") as f:
        f.write(f"\n# {str(phi)}\n")
        f.write(line + "\n")


def clear_library():
    print("Clearing library...")
    with open(LIB_FILE, "r") as f:
        lines = f.readlines()
    with open(LIB_FILE, "w") as f:
        for line in lines:
            f.write(line)
            if GEN_CODE_BANNER in line:
                break


def append_test():
    print("Appending dunder main test code...")
    with open(LIB_FILE, "a") as f:
        f.write('\n\nif __name__ == "__main__":\n')
        f.write(" " * 4 + "test_dict()")


def make_black():
    print("Attempting to code format using black...")
    cmd = f"black {LIB_FILE}"
    system(cmd)


if __name__ == "__main__":
    clear_library()
    for n in range(3, 22, 2):
        phi = parser.parse_expr(f"r**{n}")
        add_to_library(phi)

    for n in range(2, 21, 2):
        phi = parser.parse_expr(f"r**{n}*log(r)")
        add_to_library(phi)

    append_test()
    make_black()
    print("done")
