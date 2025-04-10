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

FILE = "media/cortex_figure.png"
MEDIA_DIR = "showcase/human_cortex/media/"

#############
#
# Figure
#
#############

figsize = (8, 4)
fig = plt.figure("cortex labyrinth", figsize=figsize)

snapshot_indices = [i * 1000 for i in range(6)]
grid = gs.GridSpec(3, len(snapshot_indices))

axes = []
for ax_index, snapshot_index in zip(range(len(snapshot_indices)), snapshot_indices):
    ax = fig.add_subplot(grid[0, ax_index])
    axes.append(ax)
    with open(
        MEDIA_DIR
        + f"cortex_labyrinth_frames/cortex_labyrinth_frames_{snapshot_index}.png",
        "rb",
    ) as f:
        image = plt.imread(f)
    ax.imshow(image[200:600, 200:800])
    ax.axis("off")


for panel_index in [1, 2, 0]:
    with open(MEDIA_DIR + f"panel_{panel_index}.png", "rb") as f:
        image = plt.imread(f)
    ax = fig.add_subplot(grid[1:, 2 * panel_index : 2 * panel_index + 2])
    axes.append(ax)
    ax.imshow(image)
    ax.axis("off")


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
        *axes,
    ],
    "ABCDEFGHIJKLMNOP",
):
    ax.text(
        subplot_label_x,
        subplot_label_y,
        label,
        transform=ax.transAxes,
        **subplot_label_font,
    )

plt.suptitle("Labyrinthine patters on a human cortex")

grid.tight_layout(fig)
plt.show()

# pdfs do not look right with pcolormaps
plt.savefig(FILE, dpi=300, bbox_inches="tight")
