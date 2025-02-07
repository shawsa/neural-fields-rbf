import numpy as np
from scipy.spatial import Delaunay, KDTree
import pyvista as pv

from min_energy_points import LocalSurfaceVoronoi
from min_energy_points.torus import SpiralTorus
from rbf.surface import TriMesh, SurfaceStencil

# torus
N = 1200
R = 3
r = 1

face_index = 100

torus = SpiralTorus(N, R=R, r=r)
N = torus.N
points = torus.points

# flat triangles
vor = LocalSurfaceVoronoi(
    torus.points,
    torus.normals,
    torus.implicit_surf,
)
trimesh = TriMesh(points, vor.triangles, normals=vor.normals)
flat_triangles = pv.PolyData(points, [(3, *f) for f in trimesh.simplices])

face = list(trimesh.faces)[face_index]

# smooth torus
smooth_N = 20_000
smooth_torus = SpiralTorus(smooth_N, R=R, r=r)
smooth_points = smooth_torus.points
smooth_vor = LocalSurfaceVoronoi(
    smooth_torus.points,
    smooth_torus.normals,
    smooth_torus.implicit_surf,
)
smooth_trimesh = TriMesh(
    smooth_points, smooth_vor.triangles, normals=smooth_vor.normals
)
smooth_flat_triangles = pv.PolyData(
    smooth_points, [(3, *f) for f in smooth_trimesh.simplices]
)

# default triangle mesh
tri_int = 100
tri_verts = np.array(
    [[j, tri_int - i] for i in range(100) for j in range(i)], dtype=float
)
tri_verts *= 1 / (tri_int - 1)
tri_delaunay = Delaunay(tri_verts @ np.array([[1, 0.5], [0, 1]]).T)

# surface triangle
edges = []
proj = face.projection_point
ts = np.linspace(0, 1, 11)
for e1, e2 in ((face.a, face.b), (face.b, face.c), (face.c, face.a)):
    xs = np.outer(ts, e1) + np.outer(1 - ts, e2)
    dirs = xs - proj

    def f(d, xs, dirs):
        return torus.implicit_surf(xs + dirs * d[:, np.newaxis])

    d1 = ts * 0
    d2 = ts * 0 + 1e-5
    f1 = f(d1, xs, dirs)
    f2 = f(d2, xs, dirs)

    for _ in range(100):
        mask = np.abs(f2 - f1) > 1e-10
        d2[mask], d1[mask] = (
            d2[mask] - (d2[mask] - d1[mask]) / (f2[mask] - f1[mask]) * f2[mask],
            d2[mask],
        )
        f2[mask], f1[mask] = f(d2[mask], xs[mask], dirs[mask]), f2[mask]

    pnts = xs + dirs * d2[:, np.newaxis]
    # edge = pv.PolyData(pnts, [(2, *pair) for pair in pairwise(range(len(pnts)))])
    edge = pv.Spline(pnts, 101)
    edges.append(edge)

##########################
#
# projection
#
##########################

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
surf_stencil = SurfaceStencil(face, points[stencil], points[stencil], stencil)
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

##########################
#
# make plots
#
##########################
SAVE = False
point_size = 10
camera_zoom = 5

# torus with flat triangles
plotter = pv.Plotter(off_screen=SAVE)
plotter.add_mesh(
    smooth_flat_triangles,
    show_edges=False,
    color="#00AAAA",
    opacity=0.3,
)
# plotter.add_points(
#     torus.points[np.array([i for i in range(torus.N) if i not in stencil])],
#     color="black",
#     render_points_as_spheres=True,
#     point_size=point_size,
# )
plotter.add_mesh(
    pv.PolyData(face.verts, (3, 0, 1, 2)),
    show_edges=True,
    show_vertices=True,
    color="white",
    show_scalar_bar=False,
)
for edge in edges:
    plotter.add_mesh(
        edge,
        color="#AAAA00",
        line_width=5,
    )

plotter.add_mesh(edge_normals.arrows, color="red", show_scalar_bar=False)
plotter.add_points(
    stencil_points,
    color="red",
    render_points_as_spheres=True,
    point_size=point_size,
)
plotter.add_points(
    planar_stencil_points,
    color="blue",
    render_points_as_spheres=True,
    point_size=point_size,
)
plotter.add_mesh(
    planar_stencil_mesh,
    color="green",
    show_scalar_bar=False,
    # style="wireframe",
    opacity=0.7,
    show_edges=True,
    # line_width=3,
)
plotter.add_mesh(
    proj_mesh,
    color="blue",
    show_scalar_bar=False,
    style="wireframe",
    show_edges=True,
    line_width=3,
)

plotter.camera.focal_point = face.center
plotter.camera.zoom(camera_zoom)
if SAVE:
    plotter.save_graphic("media/projection_stencil.pdf")
else:
    plotter.show()
