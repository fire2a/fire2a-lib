#!python3
"""👋🌎 🌲🔥
This is the raster module docstring
"""
__author__ = "Rodrigo Mahaluf Recasens"
__version__ = '005725b-dirty'

import logging as _logging
import numpy as _np
import sys
from os import chdir
from statistics import mode
from scipy.sparse import dok_matrix, lil_matrix
from sklearn.cluster import AgglomerativeClustering
from fire2a.adjacency import adjacent_cells
from raster import stack_rasters_to_ndarray
from scipy.sparse import csr_matrix, find
from pathlib import Path
from osgeo import gdal as _gdal, ogr as _ogr, osr as _osr


def array_aggregation(array: _np.array, criteria: str) -> float:
    """
    Aggregate values in a NumPy array based on the specified criteria.

    Parameters:
    - array (np.array): A NumPy array containing numerical data.
    - criteria (str): A string specifying the aggregation criteria. Supported criteria:
        - "sum": Calculate the sum of all elements in the array.
        - "mean": Calculate the mean (average) of all elements in the array.
        - "min": Find the minimum value in the array.
        - "max": Find the maximum value in the array.
        - "prod": Calculate the product of all elements in the array.
        - "mode": Calculate the mode (most frequent value) in the flattened array.

    Returns:
    - float: The result of the aggregation operation based on the specified criteria.

    Raises:
    - LookupError: If the specified criteria is not supported.

    Example:
    >>> import numpy as np
    >>> data = np.array([1, 2, 3, 4, 5])
    >>> array_aggregation(data, "sum")
    15.0
    >>> array_aggregation(data, "mean")
    3.0
    >>> array_aggregation(data, "min")
    1.0
    >>> array_aggregation(data, "max")
    5.0
    >>> array_aggregation(data, "prod")
    120.0
    >>> array_aggregation(data, "mode")
    1.0 """# fmt: skip
    aggregation_functions = {
        "sum": _np.sum,
        "mean": _np.mean,
        "min": _np.min,
        "max": _np.max,
        "prod": _np.prod,
        "mode": lambda x: mode(x.flatten()),
    }
    # Use a dictionary.get() with a default value to handle the "else" case
    aggregation_function = aggregation_functions.get(criteria, None)

    if aggregation_function is not None:
        return aggregation_function(array)
    else:
        raise LookupError(f"Unsupported criteria: {criteria}")


def raster_clusters(
    stacked_rasters: _np.ndarray,
    cellsize: float,  # assuming square cells. TODO Modify for cellsize vertical and horizontal.
    min_surface: float,
    max_surface: float,
    distance_threshold: float = 50.0,
    total_clusters: int = None,
    connectivity: int = None,
) -> _np.array:
    """
    Perform clustering on a stack of rasters and labeled clusters based on surface area, if they fit in max-min criteria or not. If a cluster does not fit in, it is label its negative, and if it is positive the opposite is true.

    Parameters:
    - stacked_rasters (np.ndarray): A 3D NumPy array representing a stack of rasters.
    - cellsize (float): The size of each square cell in the raster data.
    - min_surface (float): The minimum surface area for a cluster to be retained.
    - max_surface (float): The maximum surface area for a cluster to be retained.
    - distance_threshold (float, optional): The distance threshold for Agglomerative Clustering.
    - total_clusters (int, optional): The desired number of clusters. If not provided, clusters are determined by distance_threshold.
    - connectivity (int, optional): The connectivity of adjacent cells for clustering. Should be either 4 or 8.

    Returns:
    - np.ndarray: A 2D NumPy array representing the clustered raster with filtered clusters.

    Raises:
    - ValueError: If min_surface is greater than or equal to max_surface.
    - AssertionError: If connectivity is not 4 or 8.

    Example:
    >>> import numpy as np
    >>> raster_data = np.random.rand(3, 5, 5)  # Create a stack of rasters (3 bands, 5x5 pixels each)
    >>> cell_size = 10.0  # Square cells with a side length of 10 units
    >>> min_area = 50.0  # Minimum cluster surface area
    >>> max_area = 200.0  # Maximum cluster surface area
    >>> clustered_raster = raster_clusters(raster_data, cell_size, min_area, max_area)
    >>> print(clustered_raster)
    [[-1 -1  0  0  0]
     [ 0  0  1  1  1]
     [ 2  2  3  3  3]
     [ 2  2  4  4  4]
     [ 5  5  5  5  5]]
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

        counts = _np.bincount(cluster_raster.flatten())

        # Assuming square cells
        min_elements = min_surface / (cell_area)
        max_elements = max_surface / (cell_area)

        # Apply minimum and maximum surface filtering
        smaller_clusters = _np.where(counts < min_elements)[0]
        larger_clusters = _np.where(counts > max_elements)[0]

        for cluster in smaller_clusters:
            indices = _np.where(cluster_raster == cluster)
            cluster_raster[indices] = -cluster_raster[indices]
        for cluster in larger_clusters:
            indices = _np.where(cluster_raster == cluster)
            cluster_raster[indices] = -cluster_raster[indices]

        cluster_raster = cluster_raster.astype(_np.int16)

        return cluster_raster


def aggregate_cells_at_cluster_level(stacked_raster: _np.ndarray, band_aggregation_criteria: dict) -> dict:
    """
    Aggregate values in a stack of rasters at the cluster level based on specified criteria.

    Parameters:
    - stacked_raster (np.ndarray): A 3D NumPy array representing a stack of rasters.
    - band_aggregation_criteria (dict): A dictionary specifying aggregation criteria for each band.
        - Keys: Band names (e.g., "band1", "band2").
        - Values: Aggregation criteria (e.g., "sum", "mean", "min", "max", "prod", "mode").

    Returns:
    - dict: A dictionary containing aggregated values for each band at the cluster level.
        - Keys: Band names.
        - Values: Lists of aggregated values, one for each cluster.

    Example:
    >>> import numpy as np
    >>> stacked_rasters = np.random.rand(3, 5, 5)  # Create a stack of rasters (3 bands, 5x5 pixels each)
    >>> criteria = {"band1": "sum", "band2": "mean", "band3": "max"}
    >>> aggregated_data = aggregate_cells_at_cluster_level(stacked_rasters, criteria)
    >>> print(aggregated_data)
    {'band1': [5.456, 6.789, ...], 'band2': [1.234, 2.345, ...], 'band3': [0.987, 1.678, ...]}
    """ # fmt: skip
    band_names = list(band_aggregation_criteria.keys())
    cluster_index = band_names.index("cluster")
    band_indices = [band_names.index(x) for x in band_names if x != "cluster"]

    cluster_array = stacked_raster[cluster_index]

    # Para armar esta parte necestio datos geospaciales (los inputeare con metadatos)
    cluster_polygons = None

    cluster_values = _np.unique(cluster_array).astype(_np.int16)
    print(f"cluster_values: {cluster_array}")

    band_values_dict = {"cluster": cluster_values.tolist()}
    for b_index in band_indices:
        band = stacked_raster[b_index]
        band_name = band_names[b_index]
        criteria = band_aggregation_criteria[band_name]
        band_values_dict[band_name] = []
        for c_index in cluster_values:
            map = cluster_array == c_index
            masked_band = band[map]
            value = array_aggregation(masked_band, criteria)
            band_values_dict[band_name].append(value)

    return band_values_dict


def parse_args(argv):
    parser = ArgumentParser(description="Aduanas webscrapper", epilog="Bye!")
    parser.add_argument(
        "-d",
        "--change-directory",
        help="the script will change directory for any output",
        default=Path(__file__).parent.absolute(),
    )
    parser.add_argument("-p", "--puertos", nargs="*", help="puertos to webscrape", default=[])
    return parser.parse_args(argv)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    args = parse_args(argv)

    from pathlib import Path
    from raster import get_metadata, array2raster, polygonize, add_features

    file_list = Path("/home/rodrigo/code/Cluster_Generator_C2FK/RASTERS PORTEZUELO").glob("*.asc")
    file_list = list(file_list)
    metadata = get_metadata(file_list[0].__str__())

    asciis = Path("tests", "data").glob("*.asc")
    stacked_rasters, layer_names = stack_rasters_to_ndarray(asciis)
    print(layer_names)

    cell_size = 100
    min_surface = 400_000
    max_surface = 800_000

    clusters = raster_clusters(
        stacked_rasters,
        cell_size,
        min_surface,
        max_surface,
        distance_threshold=None,
        total_clusters=200,
        connectivity=8,
    )
    raster = array2raster(clusters, metadata, band_name="Clusters", output_filename="cluster_raster.tif")
    shape = polygonize(raster, "poligono_salida.shp")

    stacked_rasters = _np.concatenate((stacked_rasters, _np.expand_dims(clusters, axis=0)))
    layer_names.append("cluster")
    band_dict = {"elevation": "mean", "fuels": "mode", "type_of_slope": "mode", "cluster": None}
    dct = aggregate_cells_at_cluster_level(stacked_rasters, band_dict)
    from pprint import pprint

    print(dct)


if __name__ == "__main__":
    sys.exit(main())
