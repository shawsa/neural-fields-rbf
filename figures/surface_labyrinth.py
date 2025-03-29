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

FILE = "media/surface_labyrinth"

#############
#
# Figure
#
#############

figsize = (8, 4)
fig = plt.figure("surface labyrinth", figsize=figsize)
grid = gs.GridSpec(1, 3)

ax_sphere = fig.add_subplot(grid[0, 0])
with open("showcase/blood_cell/media/labyrinth_blood_cell_index0.png", "rb") as f:
    image = plt.imread(f)
im = ax_sphere.imshow(image[40:-90, 330:690])

ax_half = fig.add_subplot(grid[0, 1])
with open("showcase/blood_cell/media/labyrinth_blood_cell_index0.4.png", "rb") as f:
    image = plt.imread(f)
im = ax_half.imshow(image[50:-80, 330:690])

ax_cell = fig.add_subplot(grid[0, 2])
with open("showcase/blood_cell/media/labyrinth_blood_cell_index0.8.png", "rb") as f:
    image = plt.imread(f)
im = ax_cell.imshow(image[50:-60, 310:720])

ax_sphere.axis("off")
ax_half.axis("off")
ax_cell.axis("off")

#############
#
# Panel labels
#
#############
subplot_label_x = -0.05
subplot_label_y = 1.0
subplot_label_font = {
    "size": "x-large",
    "weight": "bold",
    "family": "stix",
    "usetex": True,
}
for ax, label in zip(
    [
        ax_sphere,
        ax_half,
        ax_cell,
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

plt.suptitle("Labyrinthine Patterns")

grid.tight_layout(fig)
plt.show()

# pdfs do not look right with pcolormaps
plt.savefig("media/labyrinthine_blood_cell.png", dpi=300, bbox_inches="tight")
