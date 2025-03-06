import numpy as np
from scipy.spatial import Delaunay
import pyvista as pv

from min_energy_points import LocalSurfaceVoronoi
from min_energy_points.torus import SpiralTorus
from rbf.surface import TriMesh

# torus
N = 100
R = 3
r = 1
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

# correct projection
edges = []
for face in list(trimesh.faces):
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


tri_faces = []
for face in list(trimesh.faces):
    proj = face.projection_point
    xs = (
        np.outer(tri_verts[:, 0], face.a)
        + np.outer(tri_verts[:, 1], face.b)
        + np.outer(1 - tri_verts[:, 0] - tri_verts[:, 1], face.c)
    )
    dirs = xs - proj

    def f(d, xs, dirs):
        return torus.implicit_surf(xs + dirs * d[:, np.newaxis])

    d1 = np.zeros(len(xs))
    d2 = d1 + 1e-5
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
    # pnts = xs
    tri_faces.append(pv.PolyData(pnts, [(3, *f) for f in tri_delaunay.simplices]))

# incorrect projection
proj_point_factor = 10

bad_edges = []
for face in list(trimesh.faces):
    proj = face.projection_point - proj_point_factor * face.normal
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
        bad_edges.append(edge)


bad_tri_faces = []
for face in list(trimesh.faces):
    proj = face.projection_point - proj_point_factor * face.normal
    xs = (
        np.outer(tri_verts[:, 0], face.a)
        + np.outer(tri_verts[:, 1], face.b)
        + np.outer(1 - tri_verts[:, 0] - tri_verts[:, 1], face.c)
    )
    dirs = xs - proj

    def f(d, xs, dirs):
        return torus.implicit_surf(xs + dirs * d[:, np.newaxis])

    d1 = np.zeros(len(xs))
    d2 = d1 + 1e-5
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
    # pnts = xs
    bad_tri_faces.append(pv.PolyData(pnts, [(3, *f) for f in tri_delaunay.simplices]))


# make plots
SAVE = True

# torus
plotter = pv.Plotter(off_screen=SAVE)
plotter.add_mesh(smooth_flat_triangles, show_edges=False, color="#00AAAA")
plotter.add_points(
    torus.points,
    color="black",
    render_points_as_spheres=True,
)
if SAVE:
    plotter.save_graphic("media/torus.pdf")
else:
    plotter.show()

# flat_triangles
plotter = pv.Plotter(off_screen=SAVE)
plotter.add_mesh(
    flat_triangles,
    show_edges=True,
    show_vertices=True,
    color="white",
    show_scalar_bar=False,
)
if SAVE:
    plotter.save_graphic("media/torus_flat_triangles.pdf")
else:
    plotter.show()

# torus with flat triangles
plotter = pv.Plotter(off_screen=SAVE)
plotter.add_mesh(smooth_flat_triangles, show_edges=False, color="#00AAAA", opacity=0.7)
plotter.add_mesh(
    flat_triangles,
    show_edges=True,
    show_vertices=True,
    color="white",
    show_scalar_bar=False,
)
if SAVE:
    plotter.save_graphic("media/torus_over_flat_triangles.pdf")
else:
    plotter.show()

# correct projection
plotter = pv.Plotter(off_screen=SAVE)
plotter.add_mesh(
    flat_triangles,
    show_edges=True,
    show_vertices=True,
    color="white",
    show_scalar_bar=False,
)
for tri_face in tri_faces:
    plotter.add_mesh(tri_face, show_edges=False, color="#00AAAA", opacity=0.7)
for edge in edges:
    plotter.add_mesh(
        edge,
        color="#AAAA00",
        line_width=5,
    )
plotter.add_points(
    points,
    color="black",
    point_size=10,
    render_points_as_spheres=True,
)
plotter.camera.zoom(1.7)
if SAVE:
    plotter.save_graphic("media/torus_partition.pdf")
else:
    plotter.show()

# bad projection
plotter = pv.Plotter(off_screen=SAVE)
plotter.add_mesh(
    flat_triangles,
    show_edges=True,
    show_vertices=True,
    color="white",
    show_scalar_bar=False,
)
for tri_face in bad_tri_faces:
    plotter.add_mesh(tri_face, show_edges=False, color="#00AAAA", opacity=0.7)
for edge in bad_edges:
    plotter.add_mesh(
        edge,
        color="#AAAA00",
        line_width=5,
    )
if SAVE:
    plotter.save_graphic("media/torus_bad_partition.pdf")
else:
    plotter.show()
