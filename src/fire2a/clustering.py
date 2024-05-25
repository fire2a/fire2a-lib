#!python3
"""👋🌎 🌲🔥
This is the raster module docstring
"""
__author__ = "Rodrigo Mahaluf-Recasens"
__revision__ = "$Format:%H$"

import numpy as np
from scipy.sparse import dok_matrix, lil_matrix
from sklearn.cluster import AgglomerativeClustering

from .adjacency import adjacent_cells


def raster_clusters(
    stacked_rasters: np.ndarray,
    cellsize: float,
    min_surface: float,
    max_surface: float,
    distance_threshold: float = 50.0,
    total_clusters: int = None,
    connectivity: int = None,
) -> np.ndarray:
    """
    This function receives as arguments:
    1. An array with the raster paths, e.g. raster_paths=[elevation_path,fuel_path,slope_path,...]
    You can provide as many as you want, just make sure all the raster layers are numerically defined, 
    even if there are cathegorical variables, you can not use string, transform them into a numerical raster.

    2. total_clusters: number of clusters defined by the user.

    3. min_surface: minimum area to consider into the cells aggregation process.

    4. min_surface: maximum area to condsider into the cells aggregation process.
    """  # fmt: skip
    if min_surface >= max_surface:
        raise ValueError("min_surface must be less than max_surface.")

    else:
        _, nrows, ncols = stacked_rasters.shape
        ncells = nrows * ncols
        cell_area = cellsize**2
        connectivity = connectivity if connectivity else 4
        assert connectivity == 4 or connectivity == 8, "Connectivity mut be either 4 or 8"

        flattened_data = stacked_rasters.T.reshape(-1, stacked_rasters.shape[0])  # validado

        id_pixel = list(range(1, ncells + 1))  # to set and id to every cell

        grid = lil_matrix((nrows, ncols), dtype=int)
        for idx, value in enumerate(id_pixel):
            row = idx // ncols
            col = idx % ncols
            grid[row, col] = value

        forest_grid_adjCells = adjacent_cells(grid, connectivity=connectivity)

        dict_forest_grid_adjCells = dict(
            zip(id_pixel, forest_grid_adjCells)
        )  # A dictionary of adjacents cells per id cell

        adjacency_matrix = dok_matrix((ncells, ncells))  # Create an empty matrix to save binaries adjacencies

        ## Iterate over the dictionary items and update the adjacency matrix with 1 when a cell is adjacent, 0 when is not.
        for key, values in dict_forest_grid_adjCells.items():
            for value in values:
                adjacency_matrix[key - 1, value - 1] = 1

        # Create an instance for the Agglomerative Clustering Algorithm with connectivity from the adjacency matrix
        clustering = AgglomerativeClustering(
            n_clusters=total_clusters, connectivity=adjacency_matrix, distance_threshold=distance_threshold
        )

        # Apply the algorithm over the whole data
        clustering.fit(flattened_data)
        # Reshape the cluster assignments to match the original raster shape
        cluster_raster = clustering.labels_.reshape((nrows, ncols))

        counts = np.bincount(cluster_raster.flatten())

        # Assuming square cells
        min_elements = min_surface / (cell_area)
        max_elements = max_surface / (cell_area)

        # Apply minimum and maximum surface filtering
        smaller_clusters = np.where(counts < min_elements)[0]
        larger_clusters = np.where(counts > max_elements)[0]

        for cluster in smaller_clusters:
            indices = np.where(cluster_raster == cluster)
            cluster_raster[indices] = -1

        for cluster in larger_clusters:
            indices = np.where(cluster_raster == cluster)
            cluster_raster[indices] = -1

        cluster_raster = cluster_raster.astype(np.int16)

        return cluster_raster
