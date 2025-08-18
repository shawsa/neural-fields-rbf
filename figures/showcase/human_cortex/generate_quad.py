import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from rbf.rbf import PHS
from rbf.surface import TriMesh, SurfaceQuad

import pickle

MESH_FILE = "data/left_hemi_mesh_improved.pickle"

with open(MESH_FILE, "rb") as f:
    mesh_dict = pickle.load(f)


trimesh = TriMesh(
    mesh_dict["vertices"], mesh_dict["triangles"], normals=mesh_dict["normals"]
)

quad = SurfaceQuad(
    trimesh=trimesh,
    rbf=PHS(3),
    poly_deg=1,
    stencil_size=6,
    hops_instead_of_distance=True,
    verbose=True,
    tqdm_kwargs={
        "position": 0,
        "leave": False,
        "desc": "Calculating weights",
    },
)

bad_points = quad.weights < 0
print(f"num negative: {np.sum(bad_points)}")
bad_stencils = []
for index, point in enumerate(trimesh.points):
    if not bad_points[index]:
        continue
    face_index = trimesh.vertex_map[index][0]

plt.hist(quad.weights, bins=2000)

plt.hist(np.log10(np.abs(quad.weights)), bins=2000)

with open("data/human_cortex_left_hemi_quad.pickle", "wb") as f:
    pickle.dump(quad, f)

if False:
    with open("data/human_cortex_left_hemi_quad.pickle", "rb") as f:
        quad = pickle.load(f)


# arrows = pv.PolyData([face.center for face in trimesh.faces])
plotter = pv.Plotter()
# arrows["vectors"] = trimesh.normals * 0.5
# arrows.set_active_vectors("vectors")
# plotter.add_mesh(arrows.arrows, color="black")
plotter.add_mesh(
    pv.PolyData(mesh_dict["vertices"], [(3, *f) for f in mesh_dict["triangles"]]),
    cmap="viridis",
    scalars=np.log10(np.abs(quad.weights)),
    # cmap="seismic",
    # scalars=quad.weights,
    # clim=[-1, 1],
    show_edges=True,
)
plotter.show()

stencil_index = list(trimesh.vertex_map[np.argmax(quad.weights)])[0]
stencil = quad.stencils[stencil_index]
plotter = pv.Plotter()
plotter.add_mesh(
    pv.PolyData(mesh_dict["vertices"], [(3, *f) for f in mesh_dict["triangles"]]),
    show_edges=True,
    # style="wireframe",
    # opacity=0.9,
)
stencil_points = pv.PolyData(stencil.points)
stencil_points["normals"] = trimesh.normals[stencil.point_indices]
stencil_points.set_active_vectors("normals")
plotter.add_mesh(stencil_points.arrows, color="black")
plotter.add_points(
    stencil_points,
    color="green",
    point_size=10,
)
plotter.add_points(
    pv.PolyData(quad.stencils[stencil_index].face.center),
    color="Black",
    point_size=10,
)
plotter.set_focus(stencil.face.center)
plotter.show()
