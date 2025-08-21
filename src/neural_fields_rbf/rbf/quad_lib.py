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


def complexlog(a):
    return np.log(a, dtype="complex")


# GENERATED CODE

# r**3
right_triangle_integrate_dict["r**3"] = lambda a, b: (
    (1 / 40)
    * a**2
    * (
        3 * a**3 * np.arcsinh(b / a)
        + 5 * a**2 * b * np.sqrt(1 + b**2 / a**2)
        + 2 * b**3 * np.sqrt(1 + b**2 / a**2)
    )
).real

# r**5
right_triangle_integrate_dict["r**5"] = lambda a, b: (
    (1 / 336)
    * a**2
    * (
        15 * a**5 * np.arcsinh(b / a)
        + 33 * a**4 * b * np.sqrt(1 + b**2 / a**2)
        + 26 * a**2 * b**3 * np.sqrt(1 + b**2 / a**2)
        + 8 * b**5 * np.sqrt(1 + b**2 / a**2)
    )
).real

# r**7
right_triangle_integrate_dict["r**7"] = lambda a, b: (
    (1 / 3456)
    * a**2
    * (
        105 * a**7 * np.arcsinh(b / a)
        + 279 * a**6 * b * np.sqrt(1 + b**2 / a**2)
        + 326 * a**4 * b**3 * np.sqrt(1 + b**2 / a**2)
        + 200 * a**2 * b**5 * np.sqrt(1 + b**2 / a**2)
        + 48 * b**7 * np.sqrt(1 + b**2 / a**2)
    )
).real

# r**9
right_triangle_integrate_dict["r**9"] = lambda a, b: (
    (1 / 14080)
    * a**2
    * (
        315 * a**9 * np.arcsinh(b / a)
        + 965 * a**8 * b * np.sqrt(1 + b**2 / a**2)
        + 1490 * a**6 * b**3 * np.sqrt(1 + b**2 / a**2)
        + 1368 * a**4 * b**5 * np.sqrt(1 + b**2 / a**2)
        + 656 * a**2 * b**7 * np.sqrt(1 + b**2 / a**2)
        + 128 * b**9 * np.sqrt(1 + b**2 / a**2)
    )
).real

# r**11
right_triangle_integrate_dict["r**11"] = lambda a, b: (
    (1 / 199680)
    * a**2
    * (
        3465 * a**11 * np.arcsinh(b / a)
        + 11895 * a**10 * b * np.sqrt(1 + b**2 / a**2)
        + 22790 * a**8 * b**3 * np.sqrt(1 + b**2 / a**2)
        + 27848 * a**6 * b**5 * np.sqrt(1 + b**2 / a**2)
        + 20016 * a**4 * b**7 * np.sqrt(1 + b**2 / a**2)
        + 7808 * a**2 * b**9 * np.sqrt(1 + b**2 / a**2)
        + 1280 * b**11 * np.sqrt(1 + b**2 / a**2)
    )
).real

# r**13
right_triangle_integrate_dict["r**13"] = lambda a, b: (
    (1 / 3225600)
    * a**2
    * (
        45045 * a**13 * np.arcsinh(b / a)
        + 169995 * a**12 * b * np.sqrt(1 + b**2 / a**2)
        + 388430 * a**10 * b**3 * np.sqrt(1 + b**2 / a**2)
        + 592424 * a**8 * b**5 * np.sqrt(1 + b**2 / a**2)
        + 567408 * a**6 * b**7 * np.sqrt(1 + b**2 / a**2)
        + 331904 * a**4 * b**9 * np.sqrt(1 + b**2 / a**2)
        + 108800 * a**2 * b**11 * np.sqrt(1 + b**2 / a**2)
        + 15360 * b**13 * np.sqrt(1 + b**2 / a**2)
    )
).real

# r**15
right_triangle_integrate_dict["r**15"] = lambda a, b: (
    (1 / 3899392)
    * a**2
    * (
        45045 * a**15 * np.arcsinh(b / a)
        + 184331 * a**14 * b * np.sqrt(1 + b**2 / a**2)
        + 488782 * a**12 * b**3 * np.sqrt(1 + b**2 / a**2)
        + 893480 * a**10 * b**5 * np.sqrt(1 + b**2 / a**2)
        + 1069168 * a**8 * b**7 * np.sqrt(1 + b**2 / a**2)
        + 833664 * a**6 * b**9 * np.sqrt(1 + b**2 / a**2)
        + 409856 * a**4 * b**11 * np.sqrt(1 + b**2 / a**2)
        + 115712 * a**2 * b**13 * np.sqrt(1 + b**2 / a**2)
        + 14336 * b**15 * np.sqrt(1 + b**2 / a**2)
    )
).real

# r**17
right_triangle_integrate_dict["r**17"] = lambda a, b: (
    (1 / 78446592)
    * a**2
    * (
        765765 * a**17 * np.arcsinh(b / a)
        + 3363003 * a**16 * b * np.sqrt(1 + b**2 / a**2)
        + 10144302 * a**14 * b**3 * np.sqrt(1 + b**2 / a**2)
        + 21611688 * a**12 * b**5 * np.sqrt(1 + b**2 / a**2)
        + 31020912 * a**10 * b**7 * np.sqrt(1 + b**2 / a**2)
        + 30228608 * a**8 * b**9 * np.sqrt(1 + b**2 / a**2)
        + 19812608 * a**6 * b**11 * np.sqrt(1 + b**2 / a**2)
        + 8389632 * a**4 * b**13 * np.sqrt(1 + b**2 / a**2)
        + 2078720 * a**2 * b**15 * np.sqrt(1 + b**2 / a**2)
        + 229376 * b**17 * np.sqrt(1 + b**2 / a**2)
    )
).real

# r**19
right_triangle_integrate_dict["r**19"] = lambda a, b: (
    (1 / 1734082560)
    * a**2
    * (
        14549535 * a**19 * np.arcsinh(b / a)
        + 68025825 * a**18 * b * np.sqrt(1 + b**2 / a**2)
        + 229900650 * a**16 * b**3 * np.sqrt(1 + b**2 / a**2)
        + 559257720 * a**14 * b**5 * np.sqrt(1 + b**2 / a**2)
        + 936213840 * a**12 * b**7 * np.sqrt(1 + b**2 / a**2)
        + 1094568320 * a**10 * b**9 * np.sqrt(1 + b**2 / a**2)
        + 896664320 * a**8 * b**11 * np.sqrt(1 + b**2 / a**2)
        + 506219520 * a**6 * b**13 * np.sqrt(1 + b**2 / a**2)
        + 188131328 * a**4 * b**15 * np.sqrt(1 + b**2 / a**2)
        + 41517056 * a**2 * b**17 * np.sqrt(1 + b**2 / a**2)
        + 4128768 * b**19 * np.sqrt(1 + b**2 / a**2)
    )
).real

# r**21
right_triangle_integrate_dict["r**21"] = lambda a, b: (
    (1 / 1989672960)
    * a**2
    * (
        14549535 * a**21 * np.arcsinh(b / a)
        + 71957985 * a**20 * b * np.sqrt(1 + b**2 / a**2)
        + 269222250 * a**18 * b**3 * np.sqrt(1 + b**2 / a**2)
        + 736204920 * a**16 * b**5 * np.sqrt(1 + b**2 / a**2)
        + 1408073040 * a**14 * b**7 * np.sqrt(1 + b**2 / a**2)
        + 1920321920 * a**12 * b**9 * np.sqrt(1 + b**2 / a**2)
        + 1887568640 * a**10 * b**11 * np.sqrt(1 + b**2 / a**2)
        + 1331973120 * a**8 * b**13 * np.sqrt(1 + b**2 / a**2)
        + 659990528 * a**6 * b**15 * np.sqrt(1 + b**2 / a**2)
        + 218464256 * a**4 * b**17 * np.sqrt(1 + b**2 / a**2)
        + 43450368 * a**2 * b**19 * np.sqrt(1 + b**2 / a**2)
        + 3932160 * b**21 * np.sqrt(1 + b**2 / a**2)
    )
).real

# r**2*log(r)
right_triangle_integrate_dict["r**2*log(r)"] = lambda a, b: (
    (1 / 144)
    * a
    * (
        12
        * a**3
        * (
            -1j * complexlog((-a - 1j * b) / a)
            + 1j * complexlog((a - 1j * b) / a)
            - np.pi
        )
        + 3 * a**2 * b * (6 * complexlog((-(a**2) - b**2) / a**2) - 11)
        - b**3 * (7 - 6 * complexlog((-(a**2) - b**2) / a**2))
        + 12 * b * (3 * a**2 + b**2) * complexlog(1j * a)
    )
).real

# r**4*log(r)
right_triangle_integrate_dict["r**4*log(r)"] = lambda a, b: (
    (1 / 900)
    * a
    * (
        40
        * a**5
        * (
            -1j * complexlog((-a - 1j * b) / a)
            + 1j * complexlog((a - 1j * b) / a)
            - np.pi
        )
        + 15 * a**4 * b * (5 * complexlog((-(a**2) - b**2) / a**2) - 7)
        + 10 * a**2 * b**3 * (5 * complexlog((-(a**2) - b**2) / a**2) - 4)
        + b**5 * (15 * complexlog((-(a**2) - b**2) / a**2) - 11)
        + 10
        * b
        * (15 * a**4 + 10 * a**2 * b**2 + 3 * b**4)
        * complexlog(1j * a)
    )
).real

# r**6*log(r)
right_triangle_integrate_dict["r**6*log(r)"] = lambda a, b: (
    (1 / 235200)
    * a
    * (
        a**7
        * (
            -6720 * 1j * complexlog((-a - 1j * b) / a)
            + 6720 * 1j * complexlog((a - 1j * b) / a)
            - 6720 * np.pi
        )
        + 105 * a**6 * b * (140 * complexlog((-(a**2) - b**2) / a**2) - 163)
        + a**4 * b**3 * (14700 * complexlog((-(a**2) - b**2) / a**2) - 8995)
        + a**2 * b**5 * (8820 * complexlog((-(a**2) - b**2) / a**2) - 4893)
        + b**7 * (2100 * complexlog((-(a**2) - b**2) / a**2) - 1125)
        + 840
        * b
        * (35 * a**6 + 35 * a**4 * b**2 + 21 * a**2 * b**4 + 5 * b**6)
        * complexlog(1j * a)
    )
).real

# r**8*log(r)
right_triangle_integrate_dict["r**8*log(r)"] = lambda a, b: (
    (1 / 1984500)
    * a
    * (
        a**9
        * (
            -40320 * 1j * complexlog((-a - 1j * b) / a)
            + 40320 * 1j * complexlog((a - 1j * b) / a)
            - 40320 * np.pi
        )
        + 315 * a**8 * b * (315 * complexlog((-(a**2) - b**2) / a**2) - 319)
        + a**6 * b**3 * (132300 * complexlog((-(a**2) - b**2) / a**2) - 65730)
        + a**4 * b**5 * (119070 * complexlog((-(a**2) - b**2) / a**2) - 53172)
        + a**2 * b**7 * (56700 * complexlog((-(a**2) - b**2) / a**2) - 24390)
        + b**9 * (11025 * complexlog((-(a**2) - b**2) / a**2) - 4655)
        + 630
        * b
        * (
            315 * a**8
            + 420 * a**6 * b**2
            + 378 * a**4 * b**4
            + 180 * a**2 * b**6
            + 35 * b**8
        )
        * complexlog(1j * a)
    )
).real

# r**10*log(r)
right_triangle_integrate_dict["r**10*log(r)"] = lambda a, b: (
    (1 / 115259760)
    * a
    * (
        a**11
        * (
            -1774080 * 1j * complexlog((-a - 1j * b) / a)
            + 1774080 * 1j * complexlog((a - 1j * b) / a)
            - 1774080 * np.pi
        )
        + 3465 * a**10 * b * (1386 * complexlog((-(a**2) - b**2) / a**2) - 1255)
        + a**8
        * b**3
        * (8004150 * complexlog((-(a**2) - b**2) / a**2) - 3352965)
        + a**6
        * b**5
        * (9604980 * complexlog((-(a**2) - b**2) / a**2) - 3591126)
        + a**4
        * b**7
        * (6860700 * complexlog((-(a**2) - b**2) / a**2) - 2466090)
        + a**2
        * b**9
        * (2668050 * complexlog((-(a**2) - b**2) / a**2) - 940555)
        + b**11 * (436590 * complexlog((-(a**2) - b**2) / a**2) - 152145)
        + 13860
        * b
        * (
            693 * a**10
            + 1155 * a**8 * b**2
            + 1386 * a**6 * b**4
            + 990 * a**4 * b**6
            + 385 * a**2 * b**8
            + 63 * b**10
        )
        * complexlog(1j * a)
    )
).real

# r**12*log(r)
right_triangle_integrate_dict["r**12*log(r)"] = lambda a, b: (
    (1 / 3787563780)
    * a
    * (
        a**13
        * (
            -46126080 * 1j * complexlog((-a - 1j * b) / a)
            + 46126080 * 1j * complexlog((a - 1j * b) / a)
            - 46126080 * np.pi
        )
        + 45045
        * a**12
        * b
        * (3003 * complexlog((-(a**2) - b**2) / a**2) - 2477)
        + a**10
        * b**3
        * (270540270 * complexlog((-(a**2) - b**2) / a**2) - 98077980)
        + a**8
        * b**5
        * (405810405 * complexlog((-(a**2) - b**2) / a**2) - 130531401)
        + a**6
        * b**7
        * (386486100 * complexlog((-(a**2) - b**2) / a**2) - 119330640)
        + a**4
        * b**9
        * (225450225 * complexlog((-(a**2) - b**2) / a**2) - 68223155)
        + a**2
        * b**11
        * (73783710 * complexlog((-(a**2) - b**2) / a**2) - 22063860)
        + b**13 * (10405395 * complexlog((-(a**2) - b**2) / a**2) - 3087315)
        + 90090
        * b
        * (
            3003 * a**12
            + 6006 * a**10 * b**2
            + 9009 * a**8 * b**4
            + 8580 * a**6 * b**6
            + 5005 * a**4 * b**8
            + 1638 * a**2 * b**10
            + 231 * b**12
        )
        * complexlog(1j * a)
    )
).real

# r**14*log(r)
right_triangle_integrate_dict["r**14*log(r)"] = lambda a, b: (
    (1 / 74205331200)
    * a
    * (
        a**15
        * (
            -738017280 * 1j * complexlog((-a - 1j * b) / a)
            + 738017280 * 1j * complexlog((a - 1j * b) / a)
            - 738017280 * np.pi
        )
        + 45045
        * a**14
        * b
        * (51480 * complexlog((-(a**2) - b**2) / a**2) - 39203)
        + a**12
        * b**3
        * (5410805400 * complexlog((-(a**2) - b**2) / a**2) - 1730283555)
        + a**10
        * b**5
        * (9739449720 * complexlog((-(a**2) - b**2) / a**2) - 2749393647)
        + a**8
        * b**7
        * (11594583000 * complexlog((-(a**2) - b**2) / a**2) - 3137763915)
        + a**6
        * b**9
        * (9018009000 * complexlog((-(a**2) - b**2) / a**2) - 2390593205)
        + a**4
        * b**11
        * (4427022600 * complexlog((-(a**2) - b**2) / a**2) - 1159372305)
        + a**2
        * b**13
        * (1248647400 * complexlog((-(a**2) - b**2) / a**2) - 324396765)
        + b**15 * (154594440 * complexlog((-(a**2) - b**2) / a**2) - 39936897)
        + 720720
        * b
        * (
            6435 * a**14
            + 15015 * a**12 * b**2
            + 27027 * a**10 * b**4
            + 32175 * a**8 * b**6
            + 25025 * a**6 * b**8
            + 12285 * a**4 * b**10
            + 3465 * a**2 * b**12
            + 429 * b**14
        )
        * complexlog(1j * a)
    )
).real

# r**16*log(r)
right_triangle_integrate_dict["r**16*log(r)"] = lambda a, b: (
    (1 / 1005250346100)
    * a
    * (
        a**17
        * (
            -8364195840 * 1j * complexlog((-a - 1j * b) / a)
            + 8364195840 * 1j * complexlog((a - 1j * b) / a)
            - 8364195840 * np.pi
        )
        + 765765
        * a**16
        * b
        * (36465 * complexlog((-(a**2) - b**2) / a**2) - 25897)
        + a**14
        * b**3
        * (74462988600 * complexlog((-(a**2) - b**2) / a**2) - 21313281990)
        + a**12
        * b**5
        * (156372276060 * complexlog((-(a**2) - b**2) / a**2) - 39336122826)
        + a**10
        * b**7
        * (223388965800 * complexlog((-(a**2) - b**2) / a**2) - 53812056870)
        + a**8
        * b**9
        * (217183716750 * complexlog((-(a**2) - b**2) / a**2) - 51224913740)
        + a**6
        * b**11
        * (142156614600 * complexlog((-(a**2) - b**2) / a**2) - 33115809090)
        + a**4
        * b**13
        * (60143183100 * complexlog((-(a**2) - b**2) / a**2) - 13896906870)
        + a**2
        * b**15
        * (14892597720 * complexlog((-(a**2) - b**2) / a**2) - 3421403986)
        + b**17 * (1642565925 * complexlog((-(a**2) - b**2) / a**2) - 375750375)
        + 510510
        * b
        * (
            109395 * a**16
            + 291720 * a**14 * b**2
            + 612612 * a**12 * b**4
            + 875160 * a**10 * b**6
            + 850850 * a**8 * b**8
            + 556920 * a**6 * b**10
            + 235620 * a**4 * b**12
            + 58344 * a**2 * b**14
            + 6435 * b**16
        )
        * complexlog(1j * a)
    )
).real

# r**18*log(r)
right_triangle_integrate_dict["r**18*log(r)"] = lambda a, b: (
    (1 / 268811388846000)
    * a
    * (
        a**19
        * (
            -1907036651520 * 1j * complexlog((-a - 1j * b) / a)
            + 1907036651520 * 1j * complexlog((a - 1j * b) / a)
            - 1907036651520 * np.pi
        )
        + 14549535
        * a**18
        * b
        * (461890 * complexlog((-(a**2) - b**2) / a**2) - 308333)
        + a**16
        * b**3
        * (20160854163450 * complexlog((-(a**2) - b**2) / a**2) - 5224917462765)
        + a**14
        * b**5
        * (48386049992280 * complexlog((-(a**2) - b**2) / a**2) - 10977647436756)
        + a**12
        * b**7
        * (80643416653800 * complexlog((-(a**2) - b**2) / a**2) - 17503897064940)
        + a**10
        * b**9
        * (94083986096100 * complexlog((-(a**2) - b**2) / a**2) - 19987281444130)
        + a**8
        * b**11
        * (76977806805900 * complexlog((-(a**2) - b**2) / a**2) - 16148510378910)
        + a**6
        * b**13
        * (43423378198200 * complexlog((-(a**2) - b**2) / a**2) - 9034459891380)
        + a**4
        * b**15
        * (16128683330760 * complexlog((-(a**2) - b**2) / a**2) - 3336146297484)
        + a**2
        * b**17
        * (3557797793550 * complexlog((-(a**2) - b**2) / a**2) - 732732555555)
        + b**19
        * (353699195850 * complexlog((-(a**2) - b**2) / a**2) - 72601413885)
        + 58198140
        * b
        * (
            230945 * a**18
            + 692835 * a**16 * b**2
            + 1662804 * a**14 * b**4
            + 2771340 * a**12 * b**6
            + 3233230 * a**10 * b**8
            + 2645370 * a**8 * b**10
            + 1492260 * a**6 * b**12
            + 554268 * a**4 * b**14
            + 122265 * a**2 * b**16
            + 12155 * b**18
        )
        * complexlog(1j * a)
    )
).real

# r**20*log(r)
right_triangle_integrate_dict["r**20*log(r)"] = lambda a, b: (
    (1 / 620954308234260)
    * a
    * (
        a**21
        * (
            -3814073303040 * 1j * complexlog((-a - 1j * b) / a)
            + 3814073303040 * 1j * complexlog((a - 1j * b) / a)
            - 3814073303040 * np.pi
        )
        + 14549535
        * a**20
        * b
        * (969969 * complexlog((-(a**2) - b**2) / a**2) - 612467)
        + a**18
        * b**3
        * (47041993048050 * complexlog((-(a**2) - b**2) / a**2) - 11142227896800)
        + a**16
        * b**5
        * (127013381229735 * complexlog((-(a**2) - b**2) / a**2) - 26244058395555)
        + a**14
        * b**7
        * (241930249961400 * complexlog((-(a**2) - b**2) / a**2) - 47785062742560)
        + a**12
        * b**9
        * (329293951336350 * complexlog((-(a**2) - b**2) / a**2) - 63638110906370)
        + a**10
        * b**11
        * (323306788584780 * complexlog((-(a**2) - b**2) / a**2) - 61688546992800)
        + a**8
        * b**13
        * (227972735540550 * complexlog((-(a**2) - b**2) / a**2) - 43136051742630)
        + a**6
        * b**15
        * (112900783315320 * complexlog((-(a**2) - b**2) / a**2) - 21236982390624)
        + a**4
        * b**17
        * (37356876832275 * complexlog((-(a**2) - b**2) / a**2) - 6996223499265)
        + a**2
        * b**19
        * (7427683112850 * complexlog((-(a**2) - b**2) / a**2) - 1386365460480)
        + b**21
        * (672028472115 * complexlog((-(a**2) - b**2) / a**2) - 125096209095)
        + 29099070
        * b
        * (
            969969 * a**20
            + 3233230 * a**18 * b**2
            + 8729721 * a**16 * b**4
            + 16628040 * a**14 * b**6
            + 22632610 * a**12 * b**8
            + 22221108 * a**10 * b**10
            + 15668730 * a**8 * b**12
            + 7759752 * a**6 * b**14
            + 2567565 * a**4 * b**16
            + 510510 * a**2 * b**18
            + 46189 * b**20
        )
        * complexlog(1j * a)
    )
).real


if __name__ == "__main__":
    test_dict()
