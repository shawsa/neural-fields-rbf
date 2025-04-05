import numpy as np
import pyvista as pv

from rbf.rbf import PHS
from rbf.surface import TriMesh, SurfaceQuad

import pickle

MESH_FILE = "data/left_hemi_mesh.pickle"

with open(MESH_FILE, "rb") as f:
    mesh_dict = pickle.load(f)


trimesh = TriMesh(
    mesh_dict["vertices"], mesh_dict["triangles"], normals=mesh_dict["normals"]
)

quad = SurfaceQuad(
    trimesh=trimesh,
    rbf=PHS(3),
    poly_deg=1,
    stencil_size=12,
    verbose=True,
    tqdm_kwargs={
        "position": 0,
        "leave": False,
        "desc": "Calculating weights",
    },
)

with open("data/human_cortex_left_hemi_quad.pickle", "wb") as f:
    pickle.dump(quad, f)


plotter = pv.Plotter()
plotter.add_mesh(
    pv.PolyData(mesh_dict["vertices"], [(3, *f) for f in mesh_dict["triangles"]]),
    cmap="viridis",
    scalars=np.log10(np.abs(quad.weights)),
    show_edges=False,
)
plotter.show()
