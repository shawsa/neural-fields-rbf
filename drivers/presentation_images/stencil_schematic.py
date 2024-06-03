import matplotlib.pyplot as plt
import numpy as np
from rbf.quadrature import LocalQuad
from rbf.points import UnitSquare
from rbf.rbf import PHS

plt.rcParams.update(
    {
        "font.size": 20,
        "text.usetex": True,
    }
)

N = 200
stencil_size = 24
rbf = PHS(3)
poly_deg = 1

np.random.seed(0)
points = UnitSquare(N).points
qf = LocalQuad(points, rbf, poly_deg, stencil_size)
for stencil in qf.stencils:
    x, y = np.average(stencil.element.points, axis=0)
    if (0.6 <= x <= 0.7) and (0.4 <= y <= 0.6):
        break

mask = np.ones(len(points), dtype=bool)
mask[stencil.mesh_indices] = False


plt.figure(figsize=(8, 8))
plt.gca().add_patch(
        plt.Polygon(stencil.element.points, color="gray"))
plt.triplot(*points.T, qf.mesh.simplices, "k.-", markersize=0, linewidth=0.5)
plt.plot(*points[mask].T, "k.", markersize=10, label="nodes")
plt.plot([], [], "^", color="gray", label="element")
plt.plot(*stencil.points.T, "gs", markersize=10, label="stencil")
legend = plt.legend(loc="lower left", framealpha=1)
plt.axis("equal")
plt.axis("off")
plt.tight_layout()
plt.savefig("triangulation.eps")
plt.savefig("triangulation.jpeg")


plt.figure(figsize=(8, 8))
weight_color = plt.scatter(
    *points.T, c=qf.weights, cmap="jet", s=200, zorder=20
)
weight_color_bar = plt.colorbar(ax=plt.gca(), shrink=0.75)
plt.text(.3, 1.05, "Quadrature Weights")
plt.axis("equal")
plt.axis("off")
plt.tight_layout()
plt.savefig("weights.eps")
plt.savefig("weights.jpeg")

print("negative weights")
print(qf.weights[qf.weights < 0])
