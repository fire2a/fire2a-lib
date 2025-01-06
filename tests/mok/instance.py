#!python3
"""
test instance for multi objective knapsack problem
"""
import sys

import numpy as np
from fire2a.raster import write_raster
from matplotlib import pyplot as plt

# arguments
show, save, write = False, False, False
if "show" in sys.argv:
    show = True
if "save" in sys.argv:
    save = True
if "write" in sys.argv:
    write = True

width = 80
height = 60

x, y = np.meshgrid(np.arange(width), np.arange(height))

v = np.zeros((height, width, 4))
v[:, :, 0] = np.sin(x * 2 * np.pi / width)
v[:, :, 1] = -np.sin(x * 2 * np.pi / width)
v[:, :, 2] = np.sin(y * 2 * np.pi / height)
v[:, :, 3] = -np.sin(y * 2 * np.pi / height)

if show or save:
    fig, ax = plt.subplots(2, 2)
    fig.suptitle("values")
    for i, (j, k) in enumerate(np.indices((2, 2)).reshape(2, -1).T):
        ax[j, k].imshow(v[:, :, i])
        ax[j, k].set_title("v" + str(i))
    if show:
        plt.show()
    if save:
        plt.savefig("v.png")
    plt.close()

# all layers nodata
v[1, 1, :] = -9999
# one layer nodata
v[0, 0, 0] = -9999

if write:
    for i in range(4):
        write_raster(v[:, :, i], "v" + str(i) + ".tif", nodata=-9999)

w = np.ones((height, width, 2))
w[:, :, 0] += 1
w[:, :, 1] += np.triu(w[:, :, 1])
w[:, :, 1] += np.triu(w[:, :, 1], width // 3)
w[:, :, 1] += np.triu(w[:, :, 1], width * 2 // 3)

if show or save:
    fig, ax = plt.subplots(1, 2)
    fig.suptitle("weights")
    for i in range(2):
        ax[i].imshow(w[:, :, i])
        ax[i].set_title("w" + str(i))
    if show:
        plt.show()
    if save:
        plt.savefig("w.png")
    plt.close()

# all layers nodata
w[1, 1, :] = -9999
# one layer nodata
w[2, 2, 0] = -9999

if write:
    for i in range(2):
        write_raster(w[:, :, i], "w" + str(i) + ".tif", nodata=-9999)
