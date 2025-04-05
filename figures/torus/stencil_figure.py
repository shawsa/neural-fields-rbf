import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

plt.rcParams.update(
    {
        "font.size": 12,
        "text.usetex": True,
        "mathtext.fontset": "stix",
        "font.family": "STIXGeneral",
    }
)

FILE = "../media/partition_and_projection"

figsize = (8, 3)
fig = plt.figure("Stencil Projection", figsize=figsize)

grid = gs.GridSpec(1, 3)

#############
#
# partition
#
#############
ax_partition = fig.add_subplot(grid[0, 0])
img_file = "media/torus_partition.png"
with open(img_file, "rb") as file:
    partition_image = plt.imread(file)

ax_partition.imshow(partition_image)
ax_partition.axis("off")

#############
#
# projection
#
#############

ax_projection = fig.add_subplot(grid[0, 1])
img_file = "media/projection_stencil.png"
with open(img_file, "rb") as file:
    projection_image = plt.imread(file)

ax_projection.imshow(projection_image[100:-100, 200:-200])
ax_projection.axis("off")

#############
#
# DA sketch
#
#############

ax_sketch = fig.add_subplot(grid[0, 2])
img_file = "media/da_sketch.jpg"
with open(img_file, "rb") as file:
    sketch = plt.imread(file)

ax_sketch.imshow(sketch[140:900, 200:1300])
ax_sketch.axis("off")

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
        ax_partition,
        ax_projection,
        ax_sketch,
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
plt.show()

plt.savefig(FILE + ".png", dpi=300, bbox_inches="tight")
