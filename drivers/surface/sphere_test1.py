from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, ScalarFormatter
from scipy.spatial import ConvexHull
from scipy.stats import linregress
from tqdm import tqdm

from rbf.points.sphere import SpherePoints
from rbf.rbf import RBF, PHS
from rbf.surface import TriMesh, SurfaceQuad


@dataclass
class Result:
    N: int
    rbf: RBF
    poly_deg: int
    stencil_size: int
    approx: float
    error: float


quad_tqdm_args = {
    "position": 2,
    "leave": False,
}

rbf = PHS(3)
stencil_size = 41
poly_deg = 3

# Ns = np.logspace(1.7, 3.5, 20, dtype=int)
Ns = np.logspace(2, 2.5, 5, dtype=int)
repeats = 5

exact = 4 * np.pi * 1**2
results = []
for trial in (tqdm_obj := tqdm(range(repeats), position=0, leave=True)):
    for N in tqdm(Ns[::-1], position=1, leave=False):
        points = SpherePoints(N=N).points
        hull = ConvexHull(points)
        trimesh = TriMesh(points, hull.simplices, normals=points)

        quad = SurfaceQuad(
            trimesh=trimesh,
            rbf=rbf,
            poly_deg=poly_deg,
            stencil_size=stencil_size,
            verbose=True,
            tqdm_kwargs=quad_tqdm_args,
        )
        approx = np.sum(quad.weights)
        error = abs(approx - exact) / exact

        result = Result(
            N=N,
            rbf=rbf,
            poly_deg=poly_deg,
            stencil_size=stencil_size,
            approx=approx,
            error=error,
        )
        results.append(result)
        tqdm_obj.set_description(f"{trial=}, {N=}, {error=:.3E}")


hs = [result.N**-0.5 for result in results]
ns = [result.N for result in results]
errs = [result.error for result in results]
fit = linregress(np.log(hs), np.log(errs))
plt.loglog(hs, errs, "k.")
plt.loglog(
    hs,
    [np.exp(fit.intercept + np.log(h) * fit.slope) for h in hs],
    "k-",
    label=f"deg{poly_deg}~$\\mathcal{{O}}({fit.slope:.2f})$",
)
plt.legend()
plt.loglog([res.N**-.5 for res in results], [res.error for res in results], "k.")

plt.title("Area of Sphere")
plt.ylabel("Relative Error")
plt.xlabel("$N^{-1/2}$")
plt.gca().xaxis.set_major_formatter(ScalarFormatter())
plt.gca().xaxis.set_minor_formatter(NullFormatter())
plt.xticks()
plt.xticks([n**-.5 for n in Ns], [f"${n}^{{-1/2}}$" for n in Ns])

plt.savefig("sphere_convergence.png")
