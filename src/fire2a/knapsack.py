#!/usr/bin/env python3
# fmt: off
"""👋🌎 🌲🔥
# MultiObjective Knapsack Rasters
Select the best set of pixels maximizing the sum of several weighted rasters, minding capacity constraints.
## Usage
### Overview
1. Choose your raster files
2. Configure, for values: scaling strategies and absolute weights in the `config.toml` file
3. Configure, for capacites: capacity ratio in the `config.toml` file
### Execution
```bash
# get command line help
python -m fire2a.knapsack --help

# run
python -m fire2a.knapsack [config.toml]
```
### Preparation
#### 1. Choose your raster files
- Any [GDAL compatible](https://gdal.org/en/latest/drivers/raster/index.html) raster will be read
- Place them all in the same directory where the script will be executed
- "Quote them" if they have any non alphanumerical chars [a-zA-Z0-9]

#### 2. Preprocessing configuration
See the `config.toml` file for example of the configuration of the preprocessing steps. The file is structured as follows:

```toml
["filename.tif"]
scaling_strategy = "onehot"
value_weight = 0.5
capacity_ratio = -0.1
```
This example states the raster `filename.tif` values will be rescaled using the `OneHot` strategy, then multiplied by 0.5 in the sought objective; Also that at leat 10% of its weighted pixels must be selected. 

1. __scaling_strategy__
   - can be "minmax", "standard", "robust", "onehot"
   - default is "minmax", notice: other strategies may not scale into [0,1)
   - [MinMax](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html): (x-min)/(max-min)
   - [Standard](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html): (x-mean)/stddev
   - [Robust](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html): same but droping the tails of the distribution
   - [OneHot](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html): __for CATEGORICAL DATA__

2. __value_weight__
   - can be any real number, although zero does not make sense
   - positive maximizes, negative minimizes

3. __capacity_ratio__
   - can be any real number, between -1 and 1
   - is proportional to the sum of the values of the pixels in the raster
   - positive is upper bound (less or equal to), negative will be lower bound (greater or equal to the positive value)
   - zero is no constraint
   - for categorical data it does not make sense!

"""
import logging
import sys
from pathlib import Path

import numpy as np

from fire2a.utils import fprint, read_toml

logger = logging.getLogger(__name__)

config_attrs = ["scaling_strategy", "value_weight", "capacity_sense", "capacity_ratio"]
config_types = [str, float, str, float]
config_def = ["minmax", 1.0, "u", 0.0]

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

def pipelie(observations, config):
    """Create a pipeline for the observations and the configuration."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, RobustScaler, StandardScaler

    # 1. SCALING
    scalers = {
        "minmax": MinMaxScaler(),
        "standard": StandardScaler(),
        "robust": RobustScaler(),
        "onehot": OneHotEncoder(),
    }

    # 2. PIPELINE
    pipe = Pipeline(
        [
            (
                "scaler",
                ColumnTransformer(
                    [
                        (filename, scalers[cfg["scaling_strategy"]], [i])
                        for i, (filename, cfg) in enumerate(config.items())
                    ],
                    remainder="passthrough",
                ),
            )
        ],
        verbose=True,
    )

    # 3. FIT
    scaled = pipe.fit_transform(observations)
    feat_names = pipe.named_steps['scaler'].get_feature_names_out()
    return scaled, pipe, feat_names 

def arg_parser(argv=None):
    """Parse command line arguments."""
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

    parser = ArgumentParser(
        description="MultiObjective Knapsack Rasters",
        formatter_class=ArgumentDefaultsHelpFormatter,
        epilog="More at https://fire2a.github.io/fire2a-lib",
    )
    parser.add_argument(
        "config_file",
        nargs="?",
        type=Path,
        help="For each raster file, configure its preprocess: rescaling method, weight, and capacity ratio",
        default="config.toml",
    )
    parser.add_argument("-or", "--output_raster", help="Output raster file, warning overwrites!", default="")
    parser.add_argument("-a", "--authid", type=str, help="Output raster authid", default="EPSG:3857")
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
    parser.add_argument("--verbose", "-v", action="count", default=0, help="WARNING:1, INFO:2, DEBUG:3")
    args = parser.parse_args(argv)
    args.geotransform = tuple(map(float, args.geotransform[1:-1].split(",")))
    if Path(args.config_file).is_file() is False:
        parser.error(f"File {args.config_file} not found")
    return args

def main(argv=None):
    """

    args = arg_parser(["-vvv", "tmpp9r9e05_.toml"])
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
        for cfg in config_attrs:
            if cfg not in file_config:
                config[filename][cfg] = config_def[config_attrs.index(cfg)]
        logger.info("%s: %s", filename, config[filename])
    logger.debug(config)

    # 3. LEE DATA
    from fire2a.raster import read_raster

    data_list = []
    for filename, cfg in config.items():
        data, info = read_raster(filename)
        cfg.update(info)
        data_list += [data]

    # 4. VALIDAR 2d todos mismo shape
    height, width = check_shapes(data_list)

    # 5. lista[mapas] -> OBSERVACIONES
    all_observations = np.column_stack([data.ravel() for data in data_list])

    # 6. if all rasters are nodata then mask out 
    nodatas = [ cfg["NoDataValue"] for cfg in config.values() ]
    nodata_mask = np.all(all_observations == nodatas, axis=1)
    observations = all_observations[~nodata_mask]

    # 7. nodata -> 0 
    for col, nd in zip(observations.T, nodatas):
        col[col == nd ] = 0

    # scaling
    # 8. PIPELINE
    scaled, pipe, feat_names = pipelie(observations, config)

    # weights
    value_weights = []
    for name in feat_names:
        for filename, cfg in config.items():
            if name.startswith(filename):
                value_weights += [cfg["value_weight"]]

    # capacities
    cap = []
    for i,(filename, cfg) in enumerate(config.items()):
        if cr:=cfg.get(["capacity_ratio"):
            cap += [observations[:,i].sum() * cfg["capacity_ratio"]]
        else:
            cap += [None]

    # 9. KnapSack
    from pyomo import environ as pyo

    # m model
    m = pyo.ConcreteModel("MultiObjectiveKnapsack")
    # sets
    m.N = pyo.RangeSet(0, observations.shape[0] - 1)
    m.V = pyo.RangeSet(0, observations.shape[1] - 1)
    # parameters
    m.v = pyo.Param(m.V,m.N, initialize=lambda v, n: scaled[n,v])
    # variables
    m.X = pyo.Var(m.N, within=pyo.Binary)
    # constraints
    m.capacity = pyo.Constraint(rule=lambda m: pyo.sum_product(m.X, m.w, index=m.N) <= m.C)
    # objective
    m.obj = pyo.Objective(
        expr=pyo.sum_product(m.value_weight[v] * pyo.sum_product(m.X, m.v[v], index=m.N), index=m.N)
        sense=pyo.maximize,
    )
