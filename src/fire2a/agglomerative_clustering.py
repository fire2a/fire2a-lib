#!/usr/bin/env python3
# fmt: off
"""👋🌎 🌲🔥
# Raster clustering
## Usage
### Overview
1. Choose your raster files
2. Configure nodata and scaling strategies in the `config.toml` file
3. Choose "number of clusters" or "distance threshold" for the [Agglomerative](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html) clustering algorithm
   - Start with a distance threshold of 10.0 and decrease for less or increase for more clusters
   - After calibrating the distance threshold; 
   - [Sieve](https://gdal.org/en/latest/programs/gdal_sieve.html) small clusters (merge them to the biggest neighbor) with the `--sieve integer_pixels_size` option 

### Execution
```bash
# get command line help
python -m fire2a.agglomerative_clustering -h
python -m fire2a.agglomerative_clustering --help

# activate your qgis dev environment
source ~/pyqgisdev/bin/activate 
# execute 
(qgis) $ python -m fire2a.agglomerative_clustering -d 10.0

# windows💩 users should use QGIS's python
C:\\PROGRA~1\\QGIS33~1.3\\bin\\python-qgis.bat -m fire2a.agglomerative_clustering -d 10.0
```
[More info on: How to windows 💩 using qgis's python](https://github.com/fire2a/fire2a-lib/tree/main/qgis-launchers)

### Preparation
#### 1. Choose your raster files
- Any [GDAL compatible](https://gdal.org/en/latest/drivers/raster/index.html) raster will be read
- Place them all in the same directory where the script will be executed
- "Quote them" if they have any non alphanumerical chars [a-zA-Z0-9]

#### 2. Preprocessing configuration
See the `config.toml` file for example of the configuration of the preprocessing steps. The file is structured as follows:

```toml
["filename.tif"]
no_data_strategy = "most_frequent"
scaling_strategy = "onehot"
fill_value = 0
```

1. __scaling_strategy__
   - can be "standard", "robust", "onehot"
   - default is "robust"
   - [Standard](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html): (x-mean)/stddev
   - [Robust](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html): same but droping the tails of the distribution
   - [OneHot](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html): __for CATEGORICAL DATA__

2. __no_data_strategy__
   - can be "mean", "median", "most_frequent", "constant"
   - default is "mean"
   - categorical data should use "most_frequent" or "constant"
   - "constant" will use the value in __fill_value__ (see below)
   - [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)

3. __fill_value__
   - used when __no_data_strategy__ is "constant"
   - default is 0
   - [SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html)


#### 3. Clustering configuration
1. __Agglomerative__ clustering algorithm is used. The following parameters are muttually exclusive:
- `-n` or `--n_clusters`: The number of clusters to form as well as the number of centroids to generate.
- `-d` or `--distance_threshold`: The linkage distance threshold above which, clusters will not be merged. When scaling start with 10.0 and downward (0.0 is compute the whole algorithm).

For passing more parameters, see [here](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html)
"""
# fmt: on
# from IPython.terminal.embed import InteractiveShellEmbed
# InteractiveShellEmbed()()
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

from fire2a.utils import fprint, read_toml

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


class NoDataImputer(BaseEstimator, TransformerMixin):
    """A custom Imputer that treats a specified nodata_value as np.nan and supports different strategies per column"""

    def __init__(self, no_data_values, strategies, constants):
        self.no_data_values = no_data_values
        self.strategies = strategies
        self.constants = constants
        self.imputers = []
        for no_data_value, strategy, constant in zip(no_data_values, strategies, constants):
            if no_data_value:
                self.imputers += [SimpleImputer(strategy=strategy, missing_values=no_data_value, fill_value=constant)]
            else:
                self.imputers += [SimpleImputer(strategy=strategy, fill_value=constant)]

    def fit(self, X, y=None):
        for i, imputer in enumerate(self.imputers):
            imputer.fit(X[:, [i]], y)
        return self

    def transform(self, X):
        for i, imputer in enumerate(self.imputers):
            X[:, [i]] = imputer.transform(X[:, [i]])
        self.output_data = X
        return X


class RescaleAllToCommonRange(BaseEstimator, TransformerMixin):
    """A custom transformer that rescales all features to a common range [0, 1]"""

    def __init__(self):
        pass

    def fit(self, X, y=None):
        # Determine the combined range of all scaled features
        self.min_val = [x.min() for x in X]
        self.max_val = [x.max() for x in X]
        return self

    def transform(self, X):
        # Rescale all features to match the common range
        for i, (x, mi, ma) in enumerate(zip(X, self.min_val, self.max_val)):
            if ma - mi == 0:
                X[i] = x
            else:
                X[i] = (x - mi) / (ma - mi)
        return X


class CustomAgglomerativeClustering(BaseEstimator, TransformerMixin):
    def __init__(self, height, width, neighbors=4, **kwargs):
        self.height = height
        self.width = width
        self.neighbors = neighbors

        self.grid_points = np.indices((height, width)).reshape(2, -1).T
        if neighbors == 4:
            connectivity = radius_neighbors_graph(
                self.grid_points, radius=1, metric="manhattan", include_self=False, n_jobs=-1
            )
        elif neighbors == 8:
            connectivity = radius_neighbors_graph(
                self.grid_points, radius=2 ** (1 / 2), metric="euclidean", include_self=False, n_jobs=-1
            )

        self.connectivity = connectivity
        self.kwargs = kwargs
        self.model = AgglomerativeClustering(connectivity=self.connectivity, **self.kwargs)

    def fit(self, X, y=None):
        logger.debug("not sure why, but this method is never called alas needed")
        self.model.fit(X)
        return self

    def fit_predict(self, X, y=None):
        self.input_data = X
        return self.model.fit_predict(X)


def pipelie(observations, info_list, height, width, **kwargs):
    """A scipy pipeline to achieve Agglomerative Clustering with connectivity on 2d matrix
    Steps are:
    1. Impute missing values
    2. Scale the features
    3. Rescale all features to a common range
    4. Cluster the data using Agglomerative Clustering with connectivity
    5. Reshape the labels back to the original spatial map shape
    6. Return the labels and the pipeline object

    Args:
        observations (np.ndarray): The input data to cluster (n_samples, n_features) shaped
        info_list (list): A list of dictionaries containing information about each feature
        height (int): The height of the spatial map
        width (int): The width of the spatial map
        kwargs: Additional keyword arguments for AgglomerativeClustering, at least one of n_clusters or distance_threshold

    Returns:
        np.ndarray: The labels of the clusters, reshaped to the original 2d spatial map shape
        Pipeline: The pipeline object containing all the steps of the pipeline
    """
    # kwargs = {"n_clusters": args.n_clusters, "distance_threshold": args.distance_threshold}

    # imputer strategies
    no_data_values = [info["NoDataValue"] for info in info_list]
    no_data_strategies = [info["no_data_strategy"] for info in info_list]
    fill_values = [info["fill_value"] for info in info_list]
    # scaling_strategies = [info["scaling_strategy"] for info in info_list]

    # scaling strategies
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

    # # Create a temporary directory for caching calculations
    # # FOR ACCESING STEPS LATER ON VERY LARGE DATASETS
    # import tempfile
    # import joblib
    # temp_dir = tempfile.mkdtemp()
    # memory = joblib.Memory(location=temp_dir, verbose=0)

    # Create and apply the pipeline
    pipeline = Pipeline(
        steps=[
            ("no_data_imputer", NoDataImputer(no_data_values, no_data_strategies, fill_values)),
            ("feature_scaling", feature_scaler),
            ("common_rescaling", RescaleAllToCommonRange()),
            ("agglomerative_clustering", CustomAgglomerativeClustering(height, width, neighbors=4, **kwargs)),
        ],
        # memory=memory,
        verbose=True,
    )

    # apply pipeLIE
    labels = pipeline.fit_predict(observations)

    # Reshape the labels back to the original spatial map shape
    labels_reshaped = labels.reshape(height, width)
    return labels_reshaped, pipeline


def write(
    label_map,
    width,
    height,
    output_raster="",
    output_poly="output.shp",
    authid="EPSG:3857",
    geotransform=(0, 1, 0, 0, 0, 1),
    nodata=None,
    feedback=None,
):
    from osgeo import gdal, ogr, osr

    from fire2a.processing_utils import get_output_raster_format, get_vector_driver_from_filename

    # setup drivers for raster and polygon output formats
    if output_raster == "":
        raster_driver = "MEM"
    else:
        try:
            raster_driver = get_output_raster_format(output_raster, feedback=feedback)
        except Exception:
            raster_driver = "GTiff"
    try:
        poly_driver = get_vector_driver_from_filename(output_poly)
    except Exception:
        poly_driver = "ESRI Shapefile"

    # create raster output
    src_ds = gdal.GetDriverByName(raster_driver).Create(output_raster, width, height, 1, gdal.GDT_Int64)
    src_ds.SetGeoTransform(geotransform)  # != 0 ?
    src_ds.SetProjection(authid)  # != 0 ?
    #  src_band = src_ds.GetRasterBand(1)
    #  if nodata:
    #      src_band.SetNoDataValue(nodata)
    #  src_band.WriteArray(label_map)

    # create polygon output
    drv = ogr.GetDriverByName(poly_driver)
    dst_ds = drv.CreateDataSource(output_poly)
    sp_ref = osr.SpatialReference()
    sp_ref.SetFromUserInput(authid)  # != 0 ?
    dst_lyr = dst_ds.CreateLayer("clusters", srs=sp_ref, geom_type=ogr.wkbPolygon)
    dst_lyr.CreateField(ogr.FieldDefn("DN", ogr.OFTInteger64))  # != 0 ?
    dst_lyr.CreateField(ogr.FieldDefn("pixel_count", ogr.OFTInteger64))
    # dst_lyr.CreateField(ogr.FieldDefn("area", ogr.OFTInteger))
    # dst_lyr.CreateField(ogr.FieldDefn("perimeter", ogr.OFTInteger))

    # != 0 ?
    # gdal.Polygonize( srcband, maskband, dst_layer, dst_field, options, callback = gdal.TermProgress)

    # A todo junto
    #  src_band = src_ds.GetRasterBand(1)
    #  if nodata:
    #      src_band.SetNoDataValue(nodata)
    #  src_band.WriteArray(label_map)
    # gdal.Polygonize(src_band, None, dst_lyr, 0, callback=gdal.TermProgress)  # , ["8CONNECTED=8"])

    # B separado
    # for loop for creating each label_map value into a different polygonized feature
    mem_drv = ogr.GetDriverByName("Memory")
    tmp_ds = mem_drv.CreateDataSource("tmp_ds")
    # itera = iter(np.unique(label_map))
    # cluster_id = next(itera)
    areas = []
    for cluster_id in np.unique(label_map):
        # temporarily write band
        src_band = src_ds.GetRasterBand(1)
        data = np.zeros_like(label_map)
        data -= 1  # labels in 0..NC
        data[label_map == cluster_id] = label_map[label_map == cluster_id]
        src_band.WriteArray(data)
        # create feature
        tmp_lyr = tmp_ds.CreateLayer("", srs=sp_ref)
        gdal.Polygonize(src_band, None, tmp_lyr, -1)
        # set
        feat = tmp_lyr.GetNextFeature()
        geom = feat.GetGeometryRef()
        featureDefn = dst_lyr.GetLayerDefn()
        feature = ogr.Feature(featureDefn)
        feature.SetGeometry(geom)
        feature.SetField("DN", float(cluster_id))
        areas += [geom.GetArea()]
        feature.SetField("pixel_count", float(px_count))
        # feature.SetField("area", int(geom.GetArea()))
        # feature.SetField("perimeter", int(geom.Boundary().Length()))
        dst_lyr.CreateFeature(feature)

    fprint(f"Clusters: {min(areas)=} {max(areas)=}", level="info", feedback=feedback, logger=logger)
    # fix temporarily written band
    src_band = src_ds.GetRasterBand(1)
    if nodata:
        src_band.SetNoDataValue(nodata)
    src_band.WriteArray(label_map)
    # close datasets
    src_ds.FlushCache()
    src_ds = None
    dst_ds.FlushCache()
    dst_ds = None
    return True


def plot(labels_reshaped, pipeline, info_list):  # , filename="plot.png"):
    from matplotlib import pyplot as plt

    no_data_imputed = pipeline.named_steps["no_data_imputer"].output_data
    pre_aggclu_data = pipeline.named_steps["agglomerative_clustering"].input_data

    # filtrar onehot
    num_onehots = sum([1 for i in info_list if i["scaling_strategy"] == "onehot"])
    num_no_onehots = len(info_list) - num_onehots
    pre_aggclu_data = pre_aggclu_data[:, :num_no_onehots]

    # indices sin onehots
    nohots_idxs = [i for i, info in enumerate(info_list) if info["scaling_strategy"] != "onehot"]

    # filtrar onehot de no_data_treated
    no_data_imputed = no_data_imputed[:, nohots_idxs]

    # reordenados en robust y despues standard
    rob_std_idxs = [i for i, j in enumerate(nohots_idxs) if info_list[j]["scaling_strategy"] == "robust"]
    rob_std_idxs += [i for i, j in enumerate(nohots_idxs) if info_list[j]["scaling_strategy"] == "standard"]

    # reordenar rob then std
    pre_aggclu_data = pre_aggclu_data[:, rob_std_idxs]

    names = [info_list[i]["fname"] for i in nohots_idxs]

    fig, axs = plt.subplots(3, 2)

    # plot violin plot
    axs[0, 0].violinplot(no_data_imputed, showmeans=False, showmedians=True, showextrema=True)
    axs[0, 0].set_title("Violin Plot of NoData Imputed")
    axs[0, 0].yaxis.grid(True)
    axs[0, 0].set_xticks([y + 1 for y in range(num_no_onehots)], labels=names)
    axs[0, 0].set_ylabel("Observed values")

    # plot boxplot
    axs[0, 1].boxplot(no_data_imputed)
    axs[0, 1].set_title("Box Plot of NoData Imputed")
    axs[0, 1].yaxis.grid(True)
    axs[0, 1].set_xticks([y + 1 for y in range(num_no_onehots)], labels=names)
    axs[0, 1].set_ylabel("Observed values")

    # plot violin plot
    axs[1, 0].violinplot(pre_aggclu_data, showmeans=False, showmedians=True, showextrema=True)
    axs[1, 0].set_title("Violin Plot of Common Rescaled")
    axs[1, 0].yaxis.grid(True)
    axs[1, 0].set_xticks([y + 1 for y in range(num_no_onehots)], labels=names)

    # plot boxplot
    axs[1, 1].boxplot(pre_aggclu_data)
    axs[1, 1].set_title("Box Plot of Common Rescaled")
    axs[1, 1].yaxis.grid(True)
    axs[1, 1].set_xticks([y + 1 for y in range(num_no_onehots)], labels=names)

    # cluster history
    unique_labels, counts = np.unique(labels_reshaped, return_counts=True)
    axs[2, 0].plot(unique_labels, counts, marker="o", color="blue")
    axs[2, 0].set_title("Cluster history size (in pixels)")
    axs[2, 0].set_xlabel("Algorithm Step")
    axs[2, 0].set_ylabel("Size (in pixels)")

    # cluster histogram
    axs[2, 1].hist(counts, log=True)
    axs[2, 1].set_xlabel("Cluster Size (in pixels)")
    axs[2, 1].set_ylabel("Number of Clusters")
    axs[2, 1].set_title("Histogram of Cluster Sizes")

    # plt.show(block=False)
    plt.show()
    # if filename:
    #     logger.info(f"Saving plot to {filename}")
    #     plt.savefig(filename)


def sieve_filter(data, threshold=2, connectedness=4, feedback=None):
    """Apply a sieve filter to the data to remove small clusters. The sieve filter is applied using the GDAL library. https://gdal.org/en/latest/programs/gdal_sieve.html#gdal-sieve
    Args:
        data (np.ndarray): The input data to filter
        threshold (int): The maximum number of pixels in a cluster to keep
        connectedness (int): The number of connected pixels to consider when filtering 4 or 8
        feedback (QgsTaskFeedback): A feedback object to report progress to use inside QGIS plugins
    Returns:
        np.ndarray: The filtered data
    """
    logger.info("Applying sieve filter")
    from osgeo import gdal

    height, width = data.shape
    # fprint("antes", np.sort(np.unique(data, return_counts=True)), len(np.unique(data)), level="info", feedback=feedback, logger=logger)
    num_clusters = len(np.unique(data))
    src_ds = gdal.GetDriverByName("MEM").Create("sieve", width, height, 1, gdal.GDT_Int64)
    src_band = src_ds.GetRasterBand(1)
    src_band.WriteArray(data)
    if 0 != gdal.SieveFilter(src_band, None, src_band, threshold, connectedness):
        fprint("Error applying sieve filter", level="error", feedback=feedback, logger=logger)
    else:
        sieved = src_band.ReadAsArray()
        src_band = None
        src_ds = None
        num_sieved = len(np.unique(sieved))
        # fprint("despues", np.sort(np.unique(sieved, return_counts=True)), len(np.unique(sieved)), level="info", feedback=feedback, logger=logger)
        fprint(
            f"Reduced from {num_clusters} to {num_sieved} clusters, {num_clusters-num_sieved} less",
            level="info",
            feedback=feedback,
            logger=logger,
        )
        fprint(
            "Please try again increasing distance_threshold or reducing n_clusters instead...",
            level="info",
            feedback=feedback,
            logger=logger,
        )
        # from matplotlib import pyplot as plt
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # ax1.imshow(data)
        # ax1.set_title("before sieve" + str(len(np.unique(data))))
        # ax2.imshow(sieved)
        # ax2.set_title("after sieve" + str(len(np.unique(sieved))))
        # plt.show()
        # data = sieved
        return sieved


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

    parser.add_argument("-or", "--output_raster", help="Output raster file, warning overwrites!", default="")
    parser.add_argument("-op", "--output_poly", help="Output polygons file, warning overwrites!", default="output.gpkg")
    parser.add_argument("-a", "--authid", type=str, help="Output raster authid", default="EPSG:3857")
    parser.add_argument(
        "-p",
        "--plots",
        action="store_true",
        help="Raise a matplotlib window with input/output data related to the clustering. For example, the rescaled data distributions and the clustering size history",
    )
    parser.add_argument(
        "-g", "--geotransform", type=str, help="Output raster geotransform", default="(0, 1, 0, 0, 0, 1)"
    )
    parser.add_argument(
        "-nw",
        "--no_write",
        action="store_true",
        help="Do not write outputs raster nor polygons",
        default=False,
    )
    parser.add_argument(
        "-s",
        "--script",
        action="store_true",
        help="Run in script mode, returning the label_map and the pipeline object",
        default=False,
    )
    parser.add_argument(
        "--sieve",
        type=int,
        help="Use GDAL sieve filter to merge small clusters (number of pixels) into the biggest neighbor",
    )
    parser.add_argument("--verbose", "-v", action="count", default=0, help="WARNING:1, INFO:2, DEBUG:3")
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

    if args.verbose != 0:
        global logger
        from fire2a import setup_logger

        logger = setup_logger(verbosity=args.verbose)

    logger.info("args %s", args)

    # 2 LEE CONFIG
    config = read_toml(args.config_file)
    # logger.debug(config)

    # 2.1 ADD DEFAULTS
    for filename, file_config in config.items():
        if "no_data_strategy" not in file_config:
            config[filename]["no_data_strategy"] = "mean"
        if "scaling_strategy" not in file_config:
            config[filename]["scaling_strategy"] = "robust"
        if "fill_value" not in file_config:
            config[filename]["fill_value"] = 0
    logger.debug(config)

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
        logger.debug("%s", data[:2, :2])
        logger.debug("%s", info)

    # 4. VALIDAR 2d todos mismo shape
    height, width = check_shapes(data_list)

    # 5. lista[mapas] -> OBSERVACIONES
    observations = np.column_stack([data.ravel() for data in data_list])

    # 6. nodata -> feature scaling -> all scaling -> clustering
    labels_reshaped, pipeline = pipelie(
        observations,
        info_list,
        height,
        width,
        n_clusters=args.n_clusters,
        distance_threshold=args.distance_threshold,
    )  # insert more keyworded arguments to the clustering algorithm here!

    # SIEVE
    if args.sieve:
        logger.info(f"Number of clusters before sieving: {len(np.unique(labels_reshaped))}")
        labels_reshaped = sieve_filter(labels_reshaped, args.sieve)

    logger.info(f"Final number of clusters: {len(np.unique(labels_reshaped))}")

    # 7 debbuging plots
    if args.plots:
        plot(labels_reshaped, pipeline, info_list)

    # 8. ESCRIBIR RASTER
    if not args.no_write:
        if not write(
            labels_reshaped,
            width,
            height,
            output_raster=args.output_raster,
            output_poly=args.output_poly,
            authid=args.authid,
            geotransform=args.geotransform,
        ):
            logger.error("Error writing output raster")

    # 9. SCRIPT MODE
    if args.script:
        return labels_reshaped, pipeline  # en ipython me falla tambien

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
