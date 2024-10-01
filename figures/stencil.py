import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la

from rbf.points.unit_square import UnitSquare
from scipy.spatial import Delaunay, KDTree, Voronoi, voronoi_plot_2d

plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": True,
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
    }
)


N = 200
k = 12

np.random.seed(0)
points = UnitSquare(N=N).points
tree = KDTree(points)

figsize = (8.5, 4)
fig = plt.figure("stencils", figsize=figsize)
grid = gs.GridSpec(1, 2)

# Delaunay
ax_tri = fig.add_subplot(grid[0, 0])
tri = Delaunay(points)
simplex_index = 20

vertex_indices = tri.simplices[simplex_index]
center = np.sum(points[vertex_indices], axis=0)/3
r = la.norm(center - points[vertex_indices[0]])
ts = np.linspace(0, 2*np.pi, 201)

simplex_points = [points[index] for index in vertex_indices]
simplex_points.append(simplex_points[0])
simplex_points = np.array(simplex_points)

ax_tri.plot(*points.T, "ko")
ax_tri.triplot(*points.T, tri.simplices, color="k")
ax_tri.fill(*points[vertex_indices].T)
ax_tri.plot(r*np.cos(ts) + center[0], r*np.sin(ts) + center[1], "k--")

ax_tri.axis("off")
# ax_tri.axis("equal")
ax_half_width = 0.25
ax_tri.set_xlim(center[0]-ax_half_width, center[0] + ax_half_width)
ax_tri.set_ylim(center[1]-ax_half_width, center[1] + ax_half_width)

_, stencil = tree.query(center, k=k)

ax_tri.plot(*points[stencil].T, "g^")
ax_tri.set_title("Delaunay Triangulation")

# Voronoi
vor = Voronoi(points)
center = points[simplex_index]
patch_index = vor.point_region[simplex_index]
vertex = vor.vertices[vor.regions[patch_index][0]]
r = la.norm(center - vertex)

ax_vor = fig.add_subplot(grid[0, 1])
voronoi_plot_2d(
    vor,
    ax=ax_vor,
    show_points=False,
    show_vertices=False,
)
ax_vor.plot(*points.T, "ko")
ax_vor.fill(*vor.vertices[vor.regions[patch_index]].T)
ax_vor.plot(r*np.cos(ts) + vertex[0], r*np.sin(ts) + vertex[1], "k--")

_, stencil = tree.query(center, k=k)

ax_vor.plot(*points[stencil].T, "g^")

ax_vor.axis("off")
ax_vor.set_xlim(center[0]-ax_half_width, center[0] + ax_half_width)
ax_vor.set_ylim(center[1]-ax_half_width, center[1] + ax_half_width)
ax_vor.set_title("Voronoi Partition")


# Panel labels
subplot_label_x = -0.1
subplot_label_y = 1.1
subplot_label_font = {
    "size": "x-large",
    "weight": "bold",
    "family": "sans",
    "usetex": False,
}
for ax, label in zip(
    [
        ax_tri,
        ax_vor,
    ],
    "ABCDEFGH",
):
    ax.text(
        subplot_label_x,
        subplot_label_y,
        label,
        transform=ax.transAxes,
        **subplot_label_font,
    )

grid.tight_layout(fig)
plt.savefig("media/stencil.pdf")
