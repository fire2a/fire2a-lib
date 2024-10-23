#!/usr/bin/env python3
"""👋🌎 🌲🔥

pyqgisdev
python aggcluster.py -d 10.0 config_noscaling.toml
from fire2a.raster import xy2id, id2xy
"""
import logging
import sys
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import AgglomerativeClustering
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import radius_neighbors_graph
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

logger = logging.getLogger(__name__)


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


def get_map_neighbors(height, width, num_neighbors=8):
    """Get the neighbors of each cell in a 2D grid.
    n_jobs=-1 uses all available cores.
    """

    grid_points = np.indices((height, width)).reshape(2, -1).T

    nb4 = radius_neighbors_graph(grid_points, radius=1, metric="manhattan", include_self=False, n_jobs=-1)
    nb8 = radius_neighbors_graph(grid_points, radius=2 ** (1 / 2), metric="euclidean", include_self=False, n_jobs=-1)

    # assert nb4.shape[0] == width * height
    # assert nb8.shape[1] == width * height
    # for n in range(width * height):
    #     _, neighbors = np.nonzero(nb4[n])
    #     assert 2<= len(neighbors) <= 4, f"{n=} {neighbors=}"
    #     assert 3<= len(neighbors) <= 8, f"{n=} {neighbors=}"
    return nb4, nb8


# Define a custom imputer that treats a specified nodata_value as np.nan and supports different strategies per column
class NoDataImputer(BaseEstimator, TransformerMixin):
    def __init__(self, no_data_values, strategies, constants):
        self.no_data_values = no_data_values
        self.strategies = strategies
        self.constants = constants
        self.imputers = [
            SimpleImputer(strategy=strategy, missing_values=no_data_value, fill_value=constant)
            for (no_data_value, strategy, constant) in zip(no_data_values, strategies, constants)
        ]

    def fit(self, X, y=None):
        for i, imputer in enumerate(self.imputers):
            imputer.fit(X[:, [i]], y)
        return self

    def transform(self, X):
        for i, imputer in enumerate(self.imputers):
            X[:, [i]] = imputer.transform(X[:, [i]])
        return X


class RescaleAllToCommonRange(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Determine the combined range of all scaled features
        self.min_val = X.min()
        self.max_val = X.max()
        return self

    def transform(self, X):
        # Rescale all features to match the common range
        rescaled_data = (X - self.min_val) / (self.max_val - self.min_val)
        return rescaled_data


def pipelie(observations, info_list, height, width, **kwargs):
    # kwargs = {"n_clusters": args.n_clusters, "distance_threshold": args.distance_threshold}
    """Agglomerative Clustering with connectivity on a list of spatial data arrays."""

    no_data_values = [info["NoDataValue"] for info in info_list]
    no_data_strategies = [info["no_data_strategy"] for info in info_list]
    scaling_strategies = [info["scaling_strategy"] for info in info_list]
    fill_values = [info["fill_value"] for info in info_list]

    index_map = {}
    for strategy in ["robust", "standard", "onehot"]:
        index_map[strategy] = [i for i, info in enumerate(info_list) if info["scaling_strategy"] == strategy]
    # index_map
    # !cat config.toml

    # Create transformers for each type
    robust_transformer = Pipeline(steps=[("robust_step", RobustScaler())])
    standard_transformer = Pipeline(steps=[("standard_step", StandardScaler())])
    onehot_transformer = Pipeline(steps=[("onehot_step", OneHotEncoder(sparse_output=False))])

    # Combine transformers using ColumnTransformer
    feature_scaler = ColumnTransformer(
        transformers=[
            ("robust", robust_transformer, index_map["robust"]),
            ("standard", standard_transformer, index_map["standard"]),
            ("onehot", onehot_transformer, index_map["onehot"]),
        ]
    )

    grid_points = np.indices((height, width)).reshape(2, -1).T
    connectivity = radius_neighbors_graph(
        grid_points, radius=2 ** (1 / 2), metric="euclidean", include_self=False, n_jobs=-1
    )
    clustering = AgglomerativeClustering(connectivity=connectivity, **kwargs)

    # Create and apply the pipeline
    pipeline = Pipeline(
        steps=[
            ("no_data_imputer", NoDataImputer(no_data_values, no_data_strategies, fill_values)),
            ("feature_scaling", feature_scaler),
            ("common_rescaling", RescaleAllToCommonRange()),
            ("agglomerative_clustering", clustering),
        ],
        verbose=True,
    )

    # apply pipeLIE
    labels = pipeline.fit_predict(observations)

    # Reshape the labels back to the original spatial map shape
    labels_reshaped = labels.reshape(height, width)
    return labels_reshaped, pipeline


def postprocess(labels_reshaped, pipeline, data_list, info_list, width, height):
    print("EXPERIMENTE AQUI")
    print("exit para continuar")
    from IPython.terminal.embed import InteractiveShellEmbed

    InteractiveShellEmbed()()

    # preprocessed_data = pipeline.named_steps["preprocessor"].transform(flattened_data)
    # print(preprocessed_data)

    # rescaled_data = pipeline.named_steps["rescale_all"].transform(preprocessed_data)
    # print(rescaled_data)
    pass


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
        ax.set_title(info["fname"] + " " + info["no_data_strategy"] + " " + info["scaling_strategy"])
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
        "histogram pixel count per cluster"
        + info["fname"]
        + " "
        + info["no_data_strategy"]
        + " "
        + info["scaling_strategy"]
    )

    # Get the unique labels and their counts
    unique_labels, counts = np.unique(flat_labels, return_counts=True)
    # Plot a histogram of the cluster sizes
    ax2.hist(counts, log=True)
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


def arg_parser(argv=None):
    """Parse command line arguments."""
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

    parser = ArgumentParser(
        description="Agglomerative Clustering with Connectivity for raster data",
        formatter_class=ArgumentDefaultsHelpFormatter,
        epilog="More at https://fire2a.github.io/fire2a-lib",
    )
    parser.add_argument(
        "config_file",
        nargs="?",
        type=Path,
        help="For each raster file, configure its preprocess: nodata & scaling methods",
        default="config.toml",
    )

    aggclu = parser.add_mutually_exclusive_group(required=True)
    aggclu.add_argument(
        "-d",
        "--distance_threshold",
        type=float,
        help="Distance threshold (a good starting point when scaling is 10, higher means less clusters, 0 could take a long time)",
    )
    aggclu.add_argument("-n", "--n_clusters", type=int, help="Number of clusters")

    parser.add_argument("-o", "--output", help="Output raster file", default="output.tif")
    parser.add_argument("-a", "--authid", type=str, help="Output raster authid", default="EPSG:3857")
    parser.add_argument(
        "-g", "--geotransform", type=str, help="Output raster geotransform", default="(0, 1, 0, 0, 0, 1)"
    )

    args = parser.parse_args(argv)
    args.geotransform = tuple(map(float, args.geotransform[1:-1].split(",")))
    if Path(args.config_file).is_file() is False:
        parser.error(f"File {args.config_file} not found")
    return args


def main(argv=None):
    """

    args = arg_parser(["-d","10.0", "-g","(0, 10, 0, 0, 0, 10)", "config2.toml"])
    args = arg_parser(["-d","10.0"]])
    args = arg_parser(["-d","10.0", "config2.toml"])
    args = arg_parser(["-n","10"])
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
        # 2.1 ADD DEFAULTS
        for filename, file_config in config.items():
            if "no_data_strategy" not in file_config:
                config[filename]["no_data_strategy"] = "mean"
            if "scaling_strategy" not in file_config:
                config[filename]["scaling_strategy"] = "robust"
            if "fill_value" not in file_config:
                config[filename]["fill_value"] = 0
        print(config)

        # 3. LEE DATA
        from fire2a.raster import read_raster

        data_list, info_list = [], []
        for filename, file_config in config.items():
            data, info = read_raster(filename)
            info["fname"] = Path(filename).name
            info["no_data_strategy"] = file_config["no_data_strategy"]
            info["scaling_strategy"] = file_config["scaling_strategy"]
            info["fill_value"] = file_config["fill_value"]
            data_list += [data]
            info_list += [info]

        # 4. VALIDAR 2d todos mismo shape
        height, width = check_shapes(data_list)

        # 5. lista[mapas] -> OBSERVACIONES
        observations = np.column_stack([data.ravel() for data in data_list])

        # to pickle
        import pickle

        with open("pre.pkl", "wb") as f:
            pickle.dump((observations, data_list, info_list, height, width), f)
    else:
        # read pickle
        import pickle

        with open("pre.pkl", "rb") as f:
            observations, data_list, info_list, height, width = pickle.load(f)

    if True:
        # 6. nodata -> feature scaling -> all scaling -> clustering
        labels_reshaped, pipeline = pipelie(
            observations,
            info_list,
            height,
            width,
            n_clusters=args.n_clusters,
            distance_threshold=args.distance_threshold,
        )
        # to pickle
        import pickle

        with open("post.pkl", "wb") as f:
            pickle.dump((labels_reshaped, pipeline), f)
    else:
        # read pickle
        import pickle

        with open("post.pkl", "rb") as f:
            data_list, info_list, width, height = pickle.load(f)

    # trick to plot
    effective_num_clusters = len(np.unique(labels_reshaped))

    # add final data as a new data
    data_list += [labels_reshaped]
    info_list += [
        {
            "fname": f"CLUSTERS n:{args.n_clusters},",
            "no_data_strategy": f"d:{args.distance_threshold},",
            "scaling_strategy": f"eff:{effective_num_clusters}",
        }
    ]
    plot(data_list, info_list)

    postprocess(labels_reshaped, pipeline, data_list, info_list, width, height)

    from fire2a.raster import write_raster

    write_raster(
        labels_reshaped, outfile=args.output, authid=args.authid, geotransform=args.geotransform, logger=logger
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv))
