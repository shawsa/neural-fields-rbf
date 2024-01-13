import math
from .rbf import RBF

right_triangle_integrate_dict = {}


def test_dict():
    for expr_str, func in right_triangle_integrate_dict.items():
        assert isinstance(func(1.0, 2.0), float)


def get_right_triangle_integral_function(rbf: RBF):
    expr_str = str(rbf)
    if expr_str in right_triangle_integrate_dict.keys():
        return right_triangle_integrate_dict[expr_str]
    raise ValueError(f"Error: rbf {expr_str} not implemented in quadrature library")


# GENERATED CODE

# r**3
right_triangle_integrate_dict["r**3"] = (
    lambda a, b: (1 / 40)
    * a ** 2
    * (
        3 * a ** 3 * math.asinh(b / a)
        + 5 * a ** 2 * b * math.sqrt(1 + b ** 2 / a ** 2)
        + 2 * b ** 3 * math.sqrt(1 + b ** 2 / a ** 2)
    )
)

# r**5
right_triangle_integrate_dict["r**5"] = (
    lambda a, b: (1 / 336)
    * a ** 2
    * (
        15 * a ** 5 * math.asinh(b / a)
        + 33 * a ** 4 * b * math.sqrt(1 + b ** 2 / a ** 2)
        + 26 * a ** 2 * b ** 3 * math.sqrt(1 + b ** 2 / a ** 2)
        + 8 * b ** 5 * math.sqrt(1 + b ** 2 / a ** 2)
    )
)

# r**7
right_triangle_integrate_dict["r**7"] = (
    lambda a, b: (1 / 3456)
    * a ** 2
    * (
        105 * a ** 7 * math.asinh(b / a)
        + 279 * a ** 6 * b * math.sqrt(1 + b ** 2 / a ** 2)
        + 326 * a ** 4 * b ** 3 * math.sqrt(1 + b ** 2 / a ** 2)
        + 200 * a ** 2 * b ** 5 * math.sqrt(1 + b ** 2 / a ** 2)
        + 48 * b ** 7 * math.sqrt(1 + b ** 2 / a ** 2)
    )
)

# r**9
right_triangle_integrate_dict["r**9"] = (
    lambda a, b: (1 / 14080)
    * a ** 2
    * (
        315 * a ** 9 * math.asinh(b / a)
        + 965 * a ** 8 * b * math.sqrt(1 + b ** 2 / a ** 2)
        + 1490 * a ** 6 * b ** 3 * math.sqrt(1 + b ** 2 / a ** 2)
        + 1368 * a ** 4 * b ** 5 * math.sqrt(1 + b ** 2 / a ** 2)
        + 656 * a ** 2 * b ** 7 * math.sqrt(1 + b ** 2 / a ** 2)
        + 128 * b ** 9 * math.sqrt(1 + b ** 2 / a ** 2)
    )
)

# r**11
right_triangle_integrate_dict["r**11"] = (
    lambda a, b: (1 / 199680)
    * a ** 2
    * (
        3465 * a ** 11 * math.asinh(b / a)
        + 11895 * a ** 10 * b * math.sqrt(1 + b ** 2 / a ** 2)
        + 22790 * a ** 8 * b ** 3 * math.sqrt(1 + b ** 2 / a ** 2)
        + 27848 * a ** 6 * b ** 5 * math.sqrt(1 + b ** 2 / a ** 2)
        + 20016 * a ** 4 * b ** 7 * math.sqrt(1 + b ** 2 / a ** 2)
        + 7808 * a ** 2 * b ** 9 * math.sqrt(1 + b ** 2 / a ** 2)
        + 1280 * b ** 11 * math.sqrt(1 + b ** 2 / a ** 2)
    )
)

# r**2*log(r)
right_triangle_integrate_dict["r**2*log(r)"] = (
    lambda a, b: (1 / 144)
    * a
    * (
        12
        * a ** 3
        * (
            -1j * math.log((-a - 1j * b) / a)
            + 1j * math.log((a - 1j * b) / a)
            - math.pi
        )
        + 3 * a ** 2 * b * (6 * math.log((-(a ** 2) - b ** 2) / a ** 2) - 11)
        - b ** 3 * (7 - 6 * math.log((-(a ** 2) - b ** 2) / a ** 2))
        + 12 * b * (3 * a ** 2 + b ** 2) * math.log(1j * a)
    )
)

# r**4*log(r)
right_triangle_integrate_dict["r**4*log(r)"] = (
    lambda a, b: (1 / 900)
    * a
    * (
        40
        * a ** 5
        * (
            -1j * math.log((-a - 1j * b) / a)
            + 1j * math.log((a - 1j * b) / a)
            - math.pi
        )
        + 15 * a ** 4 * b * (5 * math.log((-(a ** 2) - b ** 2) / a ** 2) - 7)
        + 10 * a ** 2 * b ** 3 * (5 * math.log((-(a ** 2) - b ** 2) / a ** 2) - 4)
        + b ** 5 * (15 * math.log((-(a ** 2) - b ** 2) / a ** 2) - 11)
        + 10 * b * (15 * a ** 4 + 10 * a ** 2 * b ** 2 + 3 * b ** 4) * math.log(1j * a)
    )
)

# r**6*log(r)
right_triangle_integrate_dict["r**6*log(r)"] = (
    lambda a, b: (1 / 235200)
    * a
    * (
        a ** 7
        * (
            -6720 * 1j * math.log((-a - 1j * b) / a)
            + 6720 * 1j * math.log((a - 1j * b) / a)
            - 6720 * math.pi
        )
        + 105 * a ** 6 * b * (140 * math.log((-(a ** 2) - b ** 2) / a ** 2) - 163)
        + a ** 4 * b ** 3 * (14700 * math.log((-(a ** 2) - b ** 2) / a ** 2) - 8995)
        + a ** 2 * b ** 5 * (8820 * math.log((-(a ** 2) - b ** 2) / a ** 2) - 4893)
        + b ** 7 * (2100 * math.log((-(a ** 2) - b ** 2) / a ** 2) - 1125)
        + 840
        * b
        * (35 * a ** 6 + 35 * a ** 4 * b ** 2 + 21 * a ** 2 * b ** 4 + 5 * b ** 6)
        * math.log(1j * a)
    )
)

# r**8*log(r)
right_triangle_integrate_dict["r**8*log(r)"] = (
    lambda a, b: (1 / 1984500)
    * a
    * (
        a ** 9
        * (
            -40320 * 1j * math.log((-a - 1j * b) / a)
            + 40320 * 1j * math.log((a - 1j * b) / a)
            - 40320 * math.pi
        )
        + 315 * a ** 8 * b * (315 * math.log((-(a ** 2) - b ** 2) / a ** 2) - 319)
        + a ** 6 * b ** 3 * (132300 * math.log((-(a ** 2) - b ** 2) / a ** 2) - 65730)
        + a ** 4 * b ** 5 * (119070 * math.log((-(a ** 2) - b ** 2) / a ** 2) - 53172)
        + a ** 2 * b ** 7 * (56700 * math.log((-(a ** 2) - b ** 2) / a ** 2) - 24390)
        + b ** 9 * (11025 * math.log((-(a ** 2) - b ** 2) / a ** 2) - 4655)
        + 630
        * b
        * (
            315 * a ** 8
            + 420 * a ** 6 * b ** 2
            + 378 * a ** 4 * b ** 4
            + 180 * a ** 2 * b ** 6
            + 35 * b ** 8
        )
        * math.log(1j * a)
    )
)

# r**10*log(r)
right_triangle_integrate_dict["r**10*log(r)"] = (
    lambda a, b: (1 / 115259760)
    * a
    * (
        a ** 11
        * (
            -1774080 * 1j * math.log((-a - 1j * b) / a)
            + 1774080 * 1j * math.log((a - 1j * b) / a)
            - 1774080 * math.pi
        )
        + 3465 * a ** 10 * b * (1386 * math.log((-(a ** 2) - b ** 2) / a ** 2) - 1255)
        + a ** 8
        * b ** 3
        * (8004150 * math.log((-(a ** 2) - b ** 2) / a ** 2) - 3352965)
        + a ** 6
        * b ** 5
        * (9604980 * math.log((-(a ** 2) - b ** 2) / a ** 2) - 3591126)
        + a ** 4
        * b ** 7
        * (6860700 * math.log((-(a ** 2) - b ** 2) / a ** 2) - 2466090)
        + a ** 2 * b ** 9 * (2668050 * math.log((-(a ** 2) - b ** 2) / a ** 2) - 940555)
        + b ** 11 * (436590 * math.log((-(a ** 2) - b ** 2) / a ** 2) - 152145)
        + 13860
        * b
        * (
            693 * a ** 10
            + 1155 * a ** 8 * b ** 2
            + 1386 * a ** 6 * b ** 4
            + 990 * a ** 4 * b ** 6
            + 385 * a ** 2 * b ** 8
            + 63 * b ** 10
        )
        * math.log(1j * a)
    )
)


if __name__ == "__main__":
    test_dict()
