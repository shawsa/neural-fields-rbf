from min_energy_points import LocalSurfaceVoronoi
from min_energy_points.torus import SpiralTorus

from rbf.rbf import PHS
from rbf.surface import TriMesh, SurfaceQuad

import pyvista as pv

import pickle

R, r = 3, 1

N = 32_000
rbf = PHS(3)
poly_deg = 2
stencil_size = 12

torus = SpiralTorus(N, R=R, r=r)
N = torus.N
points = torus.points
valid_surface = False
while not valid_surface:
    vor = LocalSurfaceVoronoi(
        torus.points,
        torus.normals,
        torus.implicit_surf,
    )
    trimesh = TriMesh(points, vor.triangles, normals=vor.normals)
    valid_surface = trimesh.is_valid()

qf = SurfaceQuad(
    trimesh=trimesh,
    rbf=rbf,
    poly_deg=poly_deg,
    stencil_size=stencil_size,
    verbose=True,
    tqdm_kwargs={
        "position": 0,
        "leave": True,
        "desc": "Calculating weights",
    },
)

with open("data/torus_weights.pickle", "wb") as f:
    pickle.dump(qf.weights, f)

# plotter = pv.Plotter()
plotter = pv.Plotter(off_screen=True)
plotter.add_mesh(
    pv.PolyData(
        qf.points,
        [(3, *triangle) for triangle in trimesh.simplices],
    ),
    show_edges=False,
    scalars=qf.weights,
    cmap="viridis",
)
plotter.remove_scalar_bar()
plotter.screenshot("media/torus_weights.png")
plotter.close()
# plotter.show()
