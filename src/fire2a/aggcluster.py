#!/usr/bin/env python3
"""👋🌎 🌲🔥"""
import sys

import numpy as np


def check_shapes(data_list):
    """Check if all data arrays have the same shape and are 2D.
    Returns the shape of the data arrays if they are all equal.
    """
    from functools import reduce

    def equal_or_error(x, y):
        """Check if x and y are equal, returns x if equal else raises a ValueError."""
        if x == y:
            return x
        else:
            raise ValueError("All data arrays must have the same shape")

    shape = reduce(equal_or_error, (data.shape for data in data_list))
    if len(shape) != 2:
        raise ValueError("All data arrays must be 2D")
    height, width = shape
    return height, width


def neighbor_nan_filter(image, method="mean"):
    """Apply a filter to fill np.nan values in an image using the mean, median, mode, min, or max of the neighboring pixels"""
    from scipy.ndimage import generic_filter
    from scipy.stats import mode

    # Define custom functions for each method
    def mean_of_neighbors(values):
        valid_values = values[~np.isnan(values)]
        if len(valid_values) == 0:
            return np.nan
        return np.nanmean(valid_values)

    def median_of_neighbors(values):
        valid_values = values[~np.isnan(values)]
        if len(valid_values) == 0:
            return np.nan
        return np.nanmedian(valid_values)

    def mode_of_neighbors(values):
        valid_values = values[~np.isnan(values)]
        if len(valid_values) == 0:
            return np.nan
        return mode(valid_values).mode[0]

    def min_of_neighbors(values):
        valid_values = values[~np.isnan(values)]
        if len(valid_values) == 0:
            return np.nan
        return np.nanmin(valid_values)

    def max_of_neighbors(values):
        valid_values = values[~np.isnan(values)]
        if len(valid_values) == 0:
            return np.nan
        return np.nanmax(valid_values)

    # Select the appropriate function based on the method
    if method == "mean":
        func = mean_of_neighbors
    elif method == "median":
        func = median_of_neighbors
    elif method == "mode":
        func = mode_of_neighbors
    elif method == "min":
        func = min_of_neighbors
    elif method == "max":
        func = max_of_neighbors
    else:
        raise ValueError("Method must be 'mean', 'median', 'mode', 'min', or 'max'")

    # Apply the custom function to each pixel using a 3x3 window
    filtered_image = generic_filter(image, func, size=3, mode="constant", cval=np.nan)

    # Update only the np.nan pixels with the filtered values
    nodata_mask = np.isnan(image)
    updated_image = np.where(nodata_mask, filtered_image, image)

    return updated_image


def adjacency_connectivity(height, width):
    """Generate connectivity matrix for adjacent cells with 4 neighbors in a 2D grid.
    Args:
        height (int): The height of the 2D grid.
        width (int): The width of the 2D grid.
    Returns:
        scipy.sparse.lil_matrix: The connectivity matrix.
    """
    from scipy.sparse import lil_matrix

    n_points = height * width
    connectivity = lil_matrix((n_points, n_points))

    def xy2id(x, y):
        return x * width + y

    for i in range(height):
        for j in range(width):
            idx = xy2id(i, j)
            if i > 0:  # Up
                connectivity[idx, xy2id(i - 1, j)] = 1
            if i < height - 1:  # Down
                connectivity[idx, xy2id(i + 1, j)] = 1
            if j > 0:  # Left
                connectivity[idx, xy2id(i, j - 1)] = 1
            if j < width - 1:  # Right
                connectivity[idx, xy2id(i, j + 1)] = 1

    return connectivity


def agglomerative_clustering_data_list(*args, n_clusters=4):
    """Perform Agglomerative Clustering with connectivity on a list of spatial data arrays."""
    from sklearn.cluster import AgglomerativeClustering

    # Stack the layers to create a 3D array
    data = np.stack(args, axis=-1)

    # Reshape the data to a 2D array where each row is a point and each column is a feature
    data_reshaped = data.reshape(-1, len(args))

    height, width = args[0].shape
    # Generate the connectivity matrix using the custom callable
    connectivity = adjacency_connectivity(height, width)

    # Perform Agglomerative Clustering with connectivity
    clustering = AgglomerativeClustering(n_clusters=n_clusters, connectivity=connectivity)
    clustering.fit(data_reshaped)
    labels = clustering.labels_

    # Reshape the labels back to the original spatial map shape
    labels_reshaped = labels.reshape(height, width)
    return labels_reshaped, clustering


def hdbscan_data_list(*args, **kwargs):
    """Perform HDBSCAN clustering on a list of spatial data arrays."""
    from sklearn.cluster import HDBSCAN
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    # Combine the layers into a single feature space
    data = np.hstack(args)

    # Normalize the combined data
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(data)

    # Optionally, reduce dimensionality using PCA
    # pca = PCA(n_components=2)
    # reduced_data = pca.fit_transform(normalized_data)

    height, width = args[0].shape

    N = height * width
    if "min_cluster_size" not in kwargs:
        min_cluster_size = int(N * 0.1)
    if "max_cluster_size" not in kwargs:
        max_cluster_size = int(N * 0.5)

    # Perform HDDBSCAN clustering
    clustering = HDBSCAN(n_jobs=30, **kwargs)
    # clustering.fit(reduced_data)
    clustering.fit(normalized_data)
    labels = clustering.labels_

    return labels, clustering


def arg_parser(argv=None):
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Agglomerative Clustering with Connectivity")
    parser.add_argument("input", nargs="+", help="Input raster files")
    parser.add_argument("-n", "--n_clusters", type=int, default=4, help="Number of clusters")
    parser.add_argument("-o", "--output", help="Output raster file")
    return parser.parse_args(argv)


def main(argv=None):
    """

    args = arg_parser(["elevation.tif", "tree_height.tif", "age.tif", "-n", "3"])
    """
    if argv is sys.argv:
        argv = sys.argv[1:]
    args = arg_parser(argv)

    import matplotlib.pyplot as plt

    # Define map dimensions
    width = 160
    height = 90

    # Create random data for the 3 layers
    np.random.seed(0)  # For reproducibility

    # Create elevation data using a 2D normal distribution to emulate slopes
    x, y = np.meshgrid(np.linspace(-1, 1, width), np.linspace(-1, 1, height))
    elevation = np.exp(-(x**2 + y**2)) * 100  # Elevation with a peak in the center

    # Create tree height and age data using wavy patterns with dampening
    trees = np.exp(-x) * np.cos(y) * 50  # Tree height with a wavy pattern
    shrubs = np.exp(-y * x) * np.sin(x) * 50  # Age with a wavy pattern
    age = np.exp(-y) * np.cos(x) * 50  # Age with a wavy pattern

    # nodatas
    if False:
        elevation[0, 0] = np.nan
        trees[1, 1] = np.nan
        shrubs[2, 1] = np.nan
        age[2, 2] = np.nan

        elevation = neighbor_nan_filter(elevation, method="mean")
        trees = neighbor_nan_filter(trees, method="median")
        shrubs = neighbor_nan_filter(shrubs, method="min")
        age = neighbor_nan_filter(age, method="mode")

    data_list = [elevation, trees, shrubs, age]
    height, width = check_shapes(data_list)
    # labels_reshaped, clustering = agglomerative_clustering_data_list(*data_list, n_clusters=13)
    labels_reshaped, clustering = hdbscan_data_list(*data_list)

    # Create a 4x4 plot to visualize the three layers and the clustering results
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    # Plot elevation
    axs[0, 0].imshow(elevation, cmap="terrain", interpolation="nearest")
    axs[0, 0].set_title("Elevation")
    axs[0, 0].set_xticks(np.arange(0, width, 5))
    axs[0, 0].set_yticks(np.arange(0, height, 5))
    axs[0, 0].grid(True)

    # Plot tree height
    axs[0, 1].imshow(trees, cmap="YlGn", interpolation="nearest")
    axs[0, 1].set_title("Tree Height")
    axs[0, 1].set_xticks(np.arange(0, width, 5))
    axs[0, 1].set_yticks(np.arange(0, height, 5))
    axs[0, 1].grid(True)

    # Plot age
    axs[1, 0].imshow(age, cmap="cool", interpolation="nearest")
    axs[1, 0].set_title("Age")
    axs[1, 0].set_xticks(np.arange(0, width, 5))
    axs[1, 0].set_yticks(np.arange(0, height, 5))
    axs[1, 0].grid(True)

    # Plot clustering results
    im = axs[1, 1].imshow(labels_reshaped, cmap="viridis", interpolation="nearest")
    axs[1, 1].set_title("Clustering Results")
    axs[1, 1].set_xticks(np.arange(0, width, 5))
    axs[1, 1].set_yticks(np.arange(0, height, 5))
    axs[1, 1].grid(True)

    # Add colorbars
    fig.colorbar(axs[0, 0].imshow(elevation, cmap="terrain", interpolation="nearest"), ax=axs[0, 0])
    fig.colorbar(axs[0, 1].imshow(trees, cmap="YlGn", interpolation="nearest"), ax=axs[0, 1])
    fig.colorbar(axs[1, 0].imshow(age, cmap="cool", interpolation="nearest"), ax=axs[1, 0])
    fig.colorbar(im, ax=axs[1, 1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
