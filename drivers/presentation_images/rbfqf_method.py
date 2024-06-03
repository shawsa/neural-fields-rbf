import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from rbf.quadrature import LocalQuad
from rbf.points import UnitSquare
from rbf.rbf import PHS

import os.path

FIG_DIR = "rbfqf_method"
FIG_SIZE = (4, 4)


def plt_fix():
    plt.axis("equal")
    plt.axis("off")
    plt.tight_layout()


plt.rcParams.update(
    {
        "font.size": 20,
        "text.usetex": True,
    }
)

# step 0 - domain
plt.figure(figsize=FIG_SIZE)
plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], "k-")
plt_fix()
plt.savefig(os.path.join(FIG_DIR, "step0_domain.png"))

# step 1 - points
N = 200
stencil_size = 24
rbf = PHS(3)
poly_deg = 1

np.random.seed(0)
points = UnitSquare(N).points
plt.plot(*points.T, "k.")
plt_fix()
plt.savefig(os.path.join(FIG_DIR, "step1_points.png"))

# step 2 - partition
qf = LocalQuad(points, rbf, poly_deg, stencil_size)
plt.triplot(*points.T, qf.mesh.simplices, "k.-", markersize=0, linewidth=0.5)
plt_fix()
plt.savefig(os.path.join(FIG_DIR, "step2_mesh.png"))

# step 3 - stencil
for stencil in qf.stencils:
    x, y = np.average(stencil.element.points, axis=0)
    if (0.6 <= x <= 0.7) and (0.4 <= y <= 0.6):
        break

mask = np.ones(len(points), dtype=bool)
mask[stencil.mesh_indices] = False


plt.gca().add_patch(plt.Polygon(stencil.element.points, color="gray"))
plt.plot(*stencil.points.T, "gs", markersize=5, label="stencil")
plt_fix()
plt.savefig(os.path.join(FIG_DIR, "step3_stencil.png"))

# step 4 - interp
triang = tri.Triangulation(*points.T, triangles=qf.mesh.simplices)
triang.set_mask(
    [
        sum(i in stencil.mesh_indices for i in triangle) != 3
        for triangle in qf.mesh.simplices
    ]
)
CARDINAL_INDEX = 5
cardinal = np.zeros(N)
cardinal[stencil.mesh_indices[CARDINAL_INDEX]] = 1

plt.gca().tripcolor(triang, cardinal, shading="gouraud")
plt.gca().add_patch(plt.Polygon(stencil.element.points, color="gray"))
plt_fix()
plt.savefig(os.path.join(FIG_DIR, "step4_interp.png"))

# step 5 - weights
plt.figure(figsize=FIG_SIZE)
plt.plot([0, 1, 1, 0, 0], [0, 0, 1, 1, 0], "k-")
weight_color = plt.scatter(*points.T, c=qf.weights, cmap="jet", s=50)
# weight_color_bar = plt.colorbar(ax=plt.gca(), shrink=0.75)
plt_fix()
plt.savefig(os.path.join(FIG_DIR, "step5_weights.png"))
