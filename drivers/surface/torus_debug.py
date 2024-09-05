import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
import sympy as sym
from sympy.abc import x, y, z

from min_energy_points import TorusPoints, LocalSurfaceVoronoi
from rbf.rbf import PHS
from rbf.surface import TriMesh, SurfaceQuad

from rbf.geometry import triangle
from rbf.quadrature import QuadStencil


class TestFunc:
    def __init__(self, expr):
        self.expr = expr
        self.foo = sym.lambdify((x, y, z), expr)

    def __call__(self, points: np.ndarray[float]) -> np.ndarray[float]:
        if type(self.expr) in [int, float]:
            return np.ones(len(points))
        return self.foo(*points.T)

    def __repr__(self) -> str:
        return str(self.expr)


R, r = 3, 1
exact = 4 * np.pi**2 * R * r
test_functions = list(
    map(
        TestFunc,
        [
            1,
            1 + y,
            1 + z,
            1 + x * y,
        ],
    )
)

quad_tqdm_args = {
    "position": 3,
    "leave": False,
}

rbf = PHS(3)
stencil_size = 12
poly_deg = 1

N = 3_000

np.random.seed(0)
torus = TorusPoints(N, R=R, r=r)
points = torus.points
vor = LocalSurfaceVoronoi(torus.points, torus.normals, torus.implicit_surf)
trimesh = TriMesh(points, vor.triangles, normals=vor.normals)

quad = SurfaceQuad(
    trimesh=trimesh,
    rbf=rbf,
    poly_deg=poly_deg,
    stencil_size=stencil_size,
    verbose=True,
    tqdm_kwargs=quad_tqdm_args,
)

for test_func in test_functions:
    approx = quad.weights @ test_func(points)
    error = abs(approx - exact) / exact
    print(f"{error=:.4g}")

print(f"max weight = {np.max(quad.weights)}")
index = np.argmax(np.abs(quad.weights))

stencil_indices = [
    face_index
    for face_index, stencil in enumerate(quad.stencils)
    if index in stencil.point_indices
]
stencils = [quad.stencils[i] for i in stencil_indices]

# visualizing
surf = pv.PolyData(points, [(3, *f) for f in vor.triangles])
surf_normals = pv.PolyData(points)
surf_normals["vectors"] = torus.normals * 0.2
surf_normals.set_active_vectors("vectors")
plotter = pv.Plotter(off_screen=False)
plotter.add_mesh(
    surf,
    show_vertices=True,
    scalars=quad.weights,
    show_edges=True,
    cmap="jet",
    show_scalar_bar=False,
)
plotter.add_mesh(surf_normals.arrows, show_scalar_bar=False)
plotter.add_mesh(
    pv.PolyData(points[index]),
    style="points",
    point_size=10,
    render_points_as_spheres=True,
)
plotter.add_mesh(
    pv.PolyData([stencil.face.center for stencil in stencils]),
    style="points",
    point_size=10,
    render_points_as_spheres=True,
)
plotter.set_focus(points[index])
plotter.show()

# find bad stencil
target_index = -1
weight_mag = -float("inf")
for stencil_index, stencil in zip(stencil_indices, stencils):
    weights = sorted(np.abs(stencil.weights(rbf, poly_deg)))
    my_max = np.max(weights)
    if my_max > weight_mag:
        weight_mag = my_max
        target_index = stencil_index
    plt.semilogy(weights, ".", label=stencil_index)
plt.legend()

stencil = quad.stencils[target_index]
print("stencil weights")
for w in stencil.weights(rbf, poly_deg):
    print(w)

print("flat weights")
for w in QuadStencil(
    stencil.planar_points, triangle(stencil.planar_face_verts)
).weights(rbf, poly_deg):
    print(w)

# visualize
plotter = pv.Plotter(off_screen=False)
plotter.add_mesh(
    surf,
    show_vertices=True,
    scalars=quad.weights,
    show_edges=True,
    cmap="jet",
    show_scalar_bar=False,
)
plotter.add_mesh(
    pv.PolyData(stencil.points),
    style="points",
    point_size=30,
    render_points_as_spheres=True,
)
plotter.add_mesh(
    pv.PolyData(stencil.face.center),
    style="points",
    point_size=30,
    render_points_as_spheres=True,
    color="red",
)
plotter.add_mesh(
    pv.PolyData(stencil.face.projection_point),
    style="points",
    point_size=10,
    render_points_as_spheres=True,
    color="magenta",
)
flat_points = np.zeros((stencil_size, 3))
flat_points[:, :2] = stencil.planar_points
flat_points = flat_points @ stencil.rotation_matrix + stencil.face.center
plotter.add_mesh(
    pv.PolyData(flat_points),
    style="points",
    point_size=30,
    render_points_as_spheres=True,
    color="yellow",
)
plotter.set_focus(stencil.face.center)
plotter.show()

# find bad point and examine transformation
pnt_id = np.argmax(stencil.weights(rbf, poly_deg))
pnt = stencil.points[pnt_id]
normal = torus.normals[stencil.point_indices[pnt_id]]
face = stencil.face
proj = stencil.face.projection_point
ref = pnt - proj
num = np.dot(ref, face.normal)
print(num)
print(np.dot(ref, normal))
print(num)
print(np.dot(face.normal, face.a - proj))
print(
    f"factor = {num / np.dot(ref, normal) * (num / np.dot(face.normal, face.a - proj)) ** 2}"
)

plt.hist(quad.weights, bins=100)
