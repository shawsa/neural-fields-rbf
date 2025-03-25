import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

import numpy as np
import numpy.linalg as la
import pickle

import sys

sys.path.append("showcase/bumpy_sphere/")
from bumpy_sphere_utils import sphere_to_bumpy_sphere, rotation_matrix


plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": True,
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
    }
)

OUTPUT = "media/bumpy_sphere.png"
DATA = "showcase/bumpy_sphere/data/"
MEDIA = "showcase/bumpy_sphere/media/"

snapshot_indices = [40000, 99200]
# snapshot_indices = [112000]

#############
#
# Load Data
#
#############

centers = np.genfromtxt("showcase/bumpy_sphere/bump_centers.csv", delimiter=" ")
with open("showcase/bumpy_sphere/data/bumpy_sphere_64000_quad.pickle", "rb") as f:
    params, _ = pickle.load(f)
    del params["num_bumps"]

phi, theta = np.meshgrid(
    np.linspace(-np.pi / 2, np.pi / 2, 801), np.linspace(-np.pi, np.pi, 801)
)
X = np.cos(theta) * np.cos(phi)
Y = np.sin(theta) * np.cos(phi)
Z = np.sin(phi)


pnts = sphere_to_bumpy_sphere(np.c_[X.ravel(), Y.ravel(), Z.ravel()], centers, **params)
altitudes = la.norm(pnts, axis=1).reshape(X.shape)

with open("showcase/bumpy_sphere/data/bump_path.pickle", "rb") as f:
    _, smooth_peaks = pickle.load(f)
    smooth_peaks = np.array(smooth_peaks)
    smooth_peaks /= la.norm(smooth_peaks, axis=1)[:, np.newaxis]


def to_lat_long(points: np.ndarray[float]) -> np.ndarray[float]:
    if points.ndim == 1:
        points = points[np.newaxis, :]
    lat_long = np.c_[
        np.arctan2(points[:, 1], points[:, 0]),
        np.arctan2(points[:, 2], la.norm(points[:, :2], axis=1)),
    ]
    if len(lat_long) == 1:
        return lat_long[0]
    else:
        return lat_long


def great_circle(
    intercept: np.ndarray[float],
    tangent: np.ndarray[float],
    num_points: int,
) -> np.ndarray[float]:
    """Returns the path as an array of (theta, phi) coordinates of the great circle
    passing through the point with the given tangent vector.
    """
    plane_normal = np.cross(intercept, tangent)
    plane_normal /= la.norm(plane_normal)
    reflect_vec = np.array([0, 0, 1]) - plane_normal
    reflect_vec /= la.norm(reflect_vec)
    A = np.eye(3) - 2 * np.outer(reflect_vec, reflect_vec)
    ts = np.linspace(-np.pi, np.pi, num_points)
    flat_circ = np.c_[np.cos(ts), np.sin(ts), np.zeros_like(ts)]
    cpnts = flat_circ @ A.T
    return to_lat_long(cpnts)


def tangent_vector(index: int, path: np.ndarray[float]) -> np.ndarray[float]:
    neg_offset = 0
    while la.norm(path[index] - path[index + neg_offset]) < 1e-14:
        neg_offset -= 1
    pos_offset = 0
    while la.norm(path[index] - path[index + pos_offset]) < 1e-14:
        pos_offset += 1
    tan = path[index + pos_offset] - path[index + neg_offset]
    tan /= la.norm(tan)
    return tan


def tangent_circle(
    index: int, path: np.ndarray[float], num_points: int = 501
) -> np.ndarray[float]:
    return great_circle(path[index], tangent_vector(index, path), num_points=num_points)


#############
#
# Generate Figure
#
#############

figsize = (8, 2 * len(snapshot_indices))
fig = plt.figure("flat convergence", figsize=figsize)


grid = gs.GridSpec(len(snapshot_indices), 2)

#############
#
# sphere plots
#
#############
ax_snapshots = []
for ax_index, snapshot_index in enumerate(snapshot_indices):
    my_ax = fig.add_subplot(grid[ax_index, 0])
    ax_snapshots.append(my_ax)
    with open(
        MEDIA + f"traveling_bump_frames/traveling_bump_anim{snapshot_index}.png", "rb"
    ) as f:
        image = plt.imread(f)
    im = my_ax.imshow(image[235:530, 195:830])
    my_ax.axis("off")

#############
#
# flat paths
#
#############

ax_flat = fig.add_subplot(grid[:2, 1])
ax_flat.pcolormesh(theta, phi, altitudes, cmap="viridis")
ax_flat.axis("off")
ax_flat.text(-0.05, 0.38, "Latitude", rotation=90, transform=ax_flat.transAxes)
ax_flat.text(0.38, -0.05, "Longitude", transform=ax_flat.transAxes)
for index in snapshot_indices:
    ax_flat.plot(*tangent_circle(index, smooth_peaks, 10001).T, "w.", markersize=1.0)
ax_flat.plot(*to_lat_long(smooth_peaks).T, "r.", markersize=0.5)
for index, label in zip(snapshot_indices, "ABCDEF"):
    ax_flat.plot(*to_lat_long(smooth_peaks[index]), "w*")
    ax_flat.text(
        *(to_lat_long(smooth_peaks[index]) + np.r_[0.1, 0.03]), label, color="w"
    )

#############
#
# Panel labels
#
#############
subplot_label_x = -0.05
subplot_label_y = 1.05
subplot_label_font = {
    "size": "x-large",
    "weight": "bold",
    "family": "stix",
    "usetex": True,
}
for ax, label in zip(
    [
        *ax_snapshots,
        ax_flat,
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

plt.suptitle("A traveling spot on a bumpy sphere")

grid.tight_layout(fig)
grid.tight_layout(fig)
grid.tight_layout(fig)
plt.show()

# pdfs do not look right with pcolormaps
plt.savefig("media/bumpy_sphere_traveling_spot.png", dpi=300, bbox_inches="tight")
