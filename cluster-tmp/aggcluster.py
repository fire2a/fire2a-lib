#!/usr/bin/env python3
"""👋🌎 🌲🔥"""
import sys
from pathlib import Path

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
        return mode(valid_values, keepdims=True).mode[0]

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


def pipelie(data_list, info_list, **kwargs):
    """Agglomerative Clustering with connectivity on a list of spatial data arrays."""
    from sklearn.base import BaseEstimator, TransformerMixin
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

    class RescaleAllToCommonRange(BaseEstimator, TransformerMixin):
        def __init__(self):
            pass

        def fit(self, X, y=None):
            # Determine the combined range of all scaled features
            self.min_val = X.min()
            self.max_val = X.max()
            print(f"was fitted to {self.min_val=} and {self.max_val=}")
            return self

        def transform(self, X):
            # Rescale all features to match the common range
            rescaled_data = (X - self.min_val) / (self.max_val - self.min_val)
            print("was transformed")
            return rescaled_data

    # [ itm['fname'] for itm in info_list]
    # [data[:2,:2] for data in data_list]

    # Flatten the 2D arrays into individual columns
    flattened_data = np.hstack([data.reshape(-1, 1) for data in data_list])
    # filas observaciones
    # columnas cada feature o layer
    # flattened_data[0]
    # assert len(flattened_data[0]) == len(data_list)

    index_map = {}
    for method in ["robust", "standard", "onehot"]:
        index_map[method] = [i for i, info in enumerate(info_list) if info["scaling_method"] == method]
    # index_map
    # !cat config.toml

    # Create transformers for each type
    robust_transformer = Pipeline(steps=[("robust_step", RobustScaler())])
    standard_transformer = Pipeline(steps=[("standard_step", StandardScaler())])
    onehot_transformer = Pipeline(steps=[("onehot_step", OneHotEncoder(sparse_output=False))])

    # Combine transformers using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("robust", robust_transformer, index_map["robust"]),
            ("standard", standard_transformer, index_map["standard"]),
            ("onehot", onehot_transformer, index_map["onehot"]),
        ]
    )

    # 4 neighbor
    height, width = data_list[0].shape
    connectivity = adjacency_connectivity(height, width)

    # corner right, down
    # print(connectivity[0, 1], connectivity[0, width])
    # (1,1) up, left, right, down
    # print(
    #     connectivity[width + 1, 1],
    #     connectivity[width + 1, width],
    #     connectivity[width + 1, width + 2],
    #     connectivity[width + 1, 2 * width + 1],
    # )

    # Add AgglomerativeClustering step to the pipeline
    #  kwargs = {"n_clusters": 3}
    # print(f"clustering {kwargs=}")
    clustering = AgglomerativeClustering(connectivity=connectivity, **kwargs)

    # Create and apply the pipeline
    pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("rescale_all", RescaleAllToCommonRange()), ("clustering", clustering)],
        verbose=True,
    )

    # Fit and transform the data
    # fitted = pipeline.fit(flattened_data)
    labels = pipeline.fit_predict(flattened_data)

    # preprocessed_data = pipeline.named_steps["preprocessor"].transform(flattened_data)
    # print(preprocessed_data)

    # rescaled_data = pipeline.named_steps["rescale_all"].transform(preprocessed_data)
    # print(rescaled_data)

    # Reshape the labels back to the original spatial map shape
    labels_reshaped = labels.reshape(height, width)
    return labels_reshaped


def plot(data_list, info_list):
    """Plot a list of spatial data arrays. reading the name from the info_list["fname"]"""
    # for data_list make a plot of each layer, in a most squared grid
    from matplotlib import pyplot as plt

    # squared grid
    grid = int(np.ceil(np.sqrt(len(data_list))))
    grid_width = grid
    grid_height = grid
    # if not using last row
    if (grid * grid) - len(data_list) >= grid:
        grid_height -= 1
    # print(grid_width, grid_height)

    fig, axs = plt.subplots(grid_height, grid_width, figsize=(12, 10))
    for i, (data, info) in enumerate(zip(data_list, info_list)):
        ax = axs[i // grid, i % grid]
        im = ax.imshow(data, cmap="viridis", interpolation="nearest")
        ax.set_title(info["fname"] + " " + info["nodata_method"] + " " + info["scaling_method"])
        ax.grid(True)
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    plt.show()

    # make a histogram of the last plot
    flat_labels = data_list[-1].flatten()
    info = info_list[-1]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.hist(flat_labels)
    ax1.set_title(
        "histogram pixel count per cluster" + info["fname"] + " " + info["nodata_method"] + " " + info["scaling_method"]
    )

    # Get the unique labels and their counts
    unique_labels, counts = np.unique(flat_labels, return_counts=True)
    # Plot a histogram of the cluster sizes
    ax2.hist(counts)
    ax2.set_xlabel("Cluster Size (in pixels)")
    ax2.set_ylabel("Number of Clusters")
    ax2.set_title("Histogram of Cluster Sizes")

    plt.show()


def read_toml(config_toml="config.toml"):
    if sys.version_info >= (3, 11):
        import tomllib

        with open(config_toml, "rb") as f:
            config = tomllib.load(f)
    else:
        import toml

        config = toml.load(config_toml)
    return config


def write_raster(data, info, args, output, driver_name="GTiff", feedback=None, logger=None):
    """FIXME
    from osgeo import gdal
    from fire2a.processing_utils import get_output_raster_format, get_vector_driver_from_filename

    if not (authid := raster_props["Projection"]):
        if not (authid := args.authid):
            logger.error("No authid found on the base raster or provided")
            return 1
    logger.info("Read base raster, using authid: %s", authid)

    driver_name = get_output_raster_format(args.output, feedback=feedback)
    burn_prob_ds = gdal.GetDriverByName(driver_name).Create(burn_prob, W, H, 1, gdal.GDT_Float32)
    burn_prob_ds.SetGeoTransform(geotransform)
    burn_prob_ds.SetProjection(authid)
    band = burn_prob_ds.GetRasterBand(1)
    # band.SetUnitType("probability")
    if 0 != band.SetNoDataValue(-9999):
        fprint(f"Set NoData failed for Burn Probability {burn_prob}", level="warning", feedback=feedback, logger=logger)
    if 0 != band.WriteArray(data):
        fprint(f"WriteArray failed for Burn Probability {burn_prob}", level="warning", feedback=feedback, logger=logger)
    burn_prob_ds.FlushCache()
    burn_prob_ds = None
    """
    pass


def arg_parser(argv=None):
    """Parse command line arguments."""
    import argparse

    parser = argparse.ArgumentParser(description="Agglomerative Clustering with Connectivity")
    parser.add_argument(
        "config_file",
        nargs="?",
        type=str,
        help="Local Path to raster files & configuration file",
        default="config.toml",
    )

    aggclu = parser.add_mutually_exclusive_group()
    aggclu.add_argument("-d", "--distance_threshold", type=float, help="Distance threshold")
    aggclu.add_argument("-n", "--n_clusters", type=int, help="Number of clusters")

    parser.add_argument("-o", "--output", help="Output raster file", default="output.tif")
    return parser.parse_args(argv)


def main(argv=None):
    """

    args = arg_parser([])
    args = arg_parser(None)
    """
    if argv is sys.argv:
        argv = sys.argv[1:]
    args = arg_parser(argv)

    # 1 LEE ARGUMENTOS
    print(args)

    if True:
        # 2 LEE CONFIG
        config = read_toml(args.config_file)
        print(config)
        # itera = iter(config.items())
        # itm = next(itera)
        # itm
        # 2.1 ADD DEFAULTS
        for filename, file_config in config.items():
            if "nodata" not in file_config:
                config[filename]["nodata"] = "mean"
            if "scaling" not in file_config:
                config[filename]["scaling"] = "robust"
        print(config)

        # 3. LEE DATA
        from fire2a.raster import read_raster

        data_list, info_list = [], []
        for filename, file_config in config.items():
            data, info = read_raster(filename)
            info["fname"] = Path(filename).name
            info["nodata_method"] = file_config["nodata"]
            info["scaling_method"] = file_config["scaling"]
            data_list += [data]
            info_list += [info]

        print(info_list[0])

        # to pickle
        import pickle

        with open("forest.pkl", "wb") as f:
            pickle.dump((data_list, info_list), f)
    else:
        # read pickle
        import pickle

        with open("forest.pkl", "rb") as f:
            data_list, info_list = pickle.load(f)

    if True:
        # 4. VALIDAR 2d
        width, height = check_shapes(data_list)

        # 5. nodata -> np.nan -> replacement
        # TODO paralelizar
        # itera = iter(enumerate(zip(data_list, info_list)))
        # i, (data, info) = next(itera)
        for i, (data, info) in enumerate(zip(data_list, info_list)):
            data = data.astype(float)
            data[data == info["NoDataValue"]] = np.nan
            if info["nodata_method"] != "no":
                data = neighbor_nan_filter(data, method=info["nodata_method"])
            data_list[i] = data

        # to pickle
        import pickle

        with open("forest2.pkl", "wb") as f:
            pickle.dump((data_list, info_list, width, height), f)
    else:
        # read pickle
        import pickle

        with open("forest2.pkl", "rb") as f:
            data_list, info_list, width, height = pickle.load(f)

    if True:
        # 6. scaling -> equalizing -> clustering
        labels_reshaped = pipelie(
            data_list, info_list, n_clusters=args.n_clusters, distance_threshold=args.distance_threshold
        )
        effective_num_clusters = len(np.unique(labels_reshaped))

        # add final data as a new data
        data_list += [labels_reshaped]
        info_list += [
            {
                "fname": f"CLUSTERS n:{args.n_clusters},",
                "nodata_method": f"d:{args.distance_threshold},",
                "scaling_method": f"eff:{effective_num_clusters}",
            }
        ]
        # to pickle
        import pickle

        with open("forest3.pkl", "wb") as f:
            pickle.dump((data_list, info_list, width, height), f)
    else:
        # read pickle
        import pickle

        with open("forest3.pkl", "rb") as f:
            data_list, info_list, width, height = pickle.load(f)
    plot(data_list, info_list)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
