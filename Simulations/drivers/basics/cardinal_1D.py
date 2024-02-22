import matplotlib.pyplot as plt
import numpy as np
from rbf.interpolate import interpolate, LocalInterpolator1D
from rbf.rbf import PHS

interval = (-1, 1)
N = 21

rbf = PHS(9)
poly_deg = 3
stencil_size = 5

xs = np.linspace(*interval, N)
zs = np.linspace(*interval, 2001)


def cardinal_vec(index, N):
    ret = np.zeros(N, dtype=float)
    ret[index] = 1.0
    return ret


approx = interpolate(xs, cardinal_vec(4, N), rbf=rbf, poly_deg=poly_deg)

cardinal_basis = [
    LocalInterpolator1D(
        points=xs,
        fs=cardinal_vec(index, N),
        rbf=rbf,
        poly_deg=poly_deg,
        stencil_size=stencil_size,
    )
    for index, _ in enumerate(xs)
]

sample = [(index, cardinal_basis[index]) for index in [0, 1, 2, 11]]

fig, axes = plt.subplots(len(sample), 1, sharex=True, figsize=(2 * len(sample), 5))
for ax, (index, phi) in zip(axes, sample):
    ax.plot(zs, [phi(z) for z in zs], "b.")
    ax.plot(xs, cardinal_vec(index, N), "go")
axes[0].set_title(f"{rbf=}, {poly_deg=}, {stencil_size=}")
