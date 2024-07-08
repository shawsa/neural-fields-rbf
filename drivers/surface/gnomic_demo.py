import numpy as np
import numpy.linalg as la
from scipy.spatial import ConvexHull, KDTree
import pyvista as pv

from rbf.surface import TriMesh, SurfaceStencil

n_theta = 20
n_phi = 10
x_wavieness = 2

thetas = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
phis = np.linspace(0, np.pi, n_phi + 2)[1:-1]

points = np.zeros((n_theta * n_phi + 2, 3))
points[-1] = (0, 0, -1)
points[-2] = (0, 0, 1)
for batch, theta in enumerate(thetas):
    rows = slice(n_phi * batch, n_phi * (batch + 1))
    points[rows, 0] = np.cos(theta + np.cos(2 * x_wavieness * phis) / n_theta) * np.sin(
        phis
    )
    points[rows, 1] = np.sin(theta) * np.sin(phis)
    points[rows, 2] = np.cos(phis)
points /= la.norm(points, axis=1)[:, np.newaxis]

hull = ConvexHull(points)
trimesh = TriMesh(points, hull.simplices, normals=points)

FACE_INDEX = 33
face = trimesh.face(FACE_INDEX)
neighbors = face.neighbors

color_arr = np.zeros(len(trimesh.simplices))
color_arr[face.index] = 1
for neighbor in neighbors:
    color_arr[neighbor.index] = 2

surf = pv.PolyData(points, [(3, *f) for f in trimesh.simplices])
normals = pv.PolyData([f.center for f in trimesh.faces])
normals["vectors"] = 0.1 * np.array([f.normal for f in trimesh.faces])
normals.set_active_vectors("vectors", preference="point")

edge_normals = pv.PolyData(
    [
        (face.a + face.b) / 2,
        (face.b + face.c) / 2,
        (face.c + face.a) / 2,
    ]
)
edge_normals["vectors"] = 0.2 * np.array(face.edge_normals)
edge_normals.set_active_vectors("vectors")

kdt = KDTree(points)


_, stencil = kdt.query(face.center, k=18)
surf_stencil = SurfaceStencil(face, points[stencil], points[stencil])
stencil_points = pv.PolyData(points[stencil])
stencil_map = {value: index for index, value in enumerate(stencil)}
stencil_mesh = []
for f in trimesh.faces:
    if sum(i in stencil for i in f.vert_indices) == 3:
        stencil_mesh.append([stencil_map[i] for i in f.vert_indices])


proj = face.projection_point
planar_stencil_points = (
    surf_stencil.planar_points
    @ np.eye(3)[:2]  # cast to 3d
    @ surf_stencil.rotation_matrix  # rotation matrix is orthogonal
) + face.center

planar_stencil_mesh = pv.PolyData(
    planar_stencil_points, [(3, *f) for f in stencil_mesh]
)

projection_points = np.array([proj, *planar_stencil_mesh.points])
proj_mesh = pv.PolyData(
    projection_points, [(2, 0, i) for i in range(1, len(projection_points))]
)

plotter = pv.Plotter()
plotter.add_mesh(
    surf,
    scalars=color_arr,
    show_edges=True,
    cmap=["#AAAAAA", "#005500", "#774444"],
    show_scalar_bar=False,
)
plotter.add_mesh(normals.arrows, show_scalar_bar=False)
plotter.add_mesh(edge_normals.arrows, color="red", show_scalar_bar=False)
plotter.add_mesh(stencil_points, color="red", show_scalar_bar=False)
plotter.add_mesh(
    planar_stencil_mesh,
    color="green",
    show_scalar_bar=False,
    style="wireframe",
    show_edges=True,
    line_width=3,
)
plotter.add_mesh(
    proj_mesh,
    color="blue",
    show_scalar_bar=False,
    style="wireframe",
    show_edges=True,
    line_width=3,
)
plotter.show()
