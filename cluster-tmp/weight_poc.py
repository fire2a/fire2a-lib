import numpy as np

# %%

# Define map dimensions
# width = 160
# height = 90
width = 6
height = 9
# width = 16
# height = 9

# Create elevation data using a 2D normal distribution to emulate slopes
x, y = np.meshgrid(np.linspace(-1, 1, height), np.linspace(-1, 1, width))

feature1 = np.exp(-(x**2 + y**2))
feature2 = np.exp(-(x)) * np.cos(y)

# mix both maps as a 2 feature obervation
observations = np.c_[feature1.ravel(), feature2.ravel()]


# %%
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

grid_points = np.c_[x.ravel(), y.ravel()]
connectivity = kneighbors_graph(grid_points, n_neighbors=8, include_self=False, n_jobs=-1)


def get_label_map(observations, n_clusters=5):
    cluster = AgglomerativeClustering(n_clusters=n_clusters, connectivity=connectivity)
    labels = cluster.fit_predict(observations)
    label_map = labels.reshape((width, height))
    return label_map


# %%
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Create the figure and the line that we will manipulate
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 9))
plt.subplots_adjust(bottom=0.25)

# set matplotlib rc setting cmap='terrain'
plt.rcParams["image.cmap"] = "terrain"

# Create the plot
im1 = ax1.imshow(feature2, vmin=0, vmax=1)
im2 = ax2.imshow(get_label_map(observations))
im3 = ax3.imshow(feature1, vmin=0, vmax=1)

# Adjust the main plot to make room for the sliders
ax2.margins(x=0)

# Make a horizontal slider to control the frequency.
axfreq = plt.axes([0.25, 0.1, 0.65, 0.03])
freq_slider = Slider(
    ax=axfreq,
    label="peso relativo",
    valmin=0.0,
    valmax=1.0,
    valinit=0.5,
)


# The function to be called anytime a slider's value changes
def update(val):
    val = freq_slider.val
    im1 = ax1.imshow(feature2 * (1 - val), vmin=0, vmax=1)
    im3 = ax3.imshow(feature1 * val, vmin=0, vmax=1)
    observations = np.c_[feature1.ravel() * val, feature2.ravel() * (1 - val)]
    print(observations[0], np.max(observations), np.min(observations))
    im2 = ax2.imshow(get_label_map(observations))
    fig.canvas.draw_idle()


# Register the update function with each slider
freq_slider.on_changed(update)

plt.show()
