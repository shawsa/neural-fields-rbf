import pickle
import numpy as np
import pyvista as pv


MESH_FILE = "data/left_hemi_mesh.pickle"

with open(MESH_FILE, "rb") as f:
    mesh_dict = pickle.load(f)


points = mesh_dict["vertices"]
triangles = mesh_dict["triangles"]


plotter = pv.Plotter()
plotter.add_mesh(
    pv.PolyData(points, [(3, *t) for t in triangles])
)
plotter.show()
