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

config_attrs = ["value_rescaling", "value_weight", "capacity_sense", "capacity_ratio"]
config_types = [str, float, str, float]
config_def = ["", np.nan, "", np.nan]
config_allowed = {"value_rescaling": ["minmax", "standard", "robust", "onehot"], "capacity_sense": ["<=", ">=", "leq", "geq", "ub", "lb", ""]}

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
                        (item['name'], scalers.get(item.get("value_rescaling")), [i])
                        for i, item in enumerate(config) if item.get("value_rescaling")
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
    args = arg_parser(["-vvv", "tmpaca3giia.toml"])
     --authid EPSG:32718 --geotransform (720874.0, 100.0, 0.0, 5680572.0, 0.0, -100.0)
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
    a,b = list(config.keys()), list(config.values())
    config = [ {'name': Path(a).name, 'filename': Path(a), **b} for a,b in zip(a,b) ]
    for itm in config:
        logger.debug(itm)

    # 2.1 CHECK PAIRS
    for itm in config:
        if cr:=itm.get("capacity_ratio"):
            if not (-1 < cr < 1):
                logger.critical("Wrong value for capacity_ratio in %s, should be (-1,1)", itm)
                sys.exit(1)
            if "capacity_sense" not in itm:
                logger.warning("capacity_ratio without capacity_sense for item: %s\n ASSUMING SENSE IS UPPER BOUND", itm["name"])
                itm["capacity_sense"] = "ub"
        elif "capacity_sense" in itm:
            logger.critical("capacity_sense without capacity_ratio for item: %s", itm["name"])
            sys.exit(1)
        if vr:=itm.get("value_rescaling"):
            if vr not in config_allowed["value_rescaling"]:
                logger.critical("Wrong value for value_rescaling in %s", itm)
                sys.exit(1)
            if "value_weight" not in itm:
                logger.warning("value_rescaling without value_weight for item: %s\n ASSUMING VALUE WEIGHT IS 1", itm["name"])
                itm["value_weight"] = 1
        elif "value_weight" in itm:
            logger.warning("value_weight without value_rescaling for item: %s\n PASS THROUGH RESCALING", itm["name"])
    # 2.1 ADD DEFAULTS
    # for item in config:
    #     for attr in config_attrs:
    #         if attr not in item:
    #             item[attr] = config_def[config_attrs.index(attr)]
    #             print("adding default", attr, item[attr])
    # for item in config:
    #     #  val_resc & !val_weig ->  ?
    #     #  !val_resc & val_weig ->  ?
    #     #  cap_rati & !cap_sens ->  cap_sens = ub
    #     if np.isfinite(item['capacity_ratio']) and item['capacity_sense'] == "":
    #         item['capacity_sense'] = 'ub'
    #         print("adding default", 'capacity_sense', item['capacity_sense'])
    #     # !cap_rati &  cap_sens -> !cap_sens
    #     if item['capacity_ratio'] is np.nan and item['capacity_sense'] != "":
    #         item['capacity_sense'] = ""
    #         print("adding default", 'capacity_sense', item['capacity_sense'])
    # for item in config:
    #     for key in item:
    #         if key in config_allowed:
    #             assert item[key] in config_allowed[key], f"Wrong value for {key} in {item}"
    #     logger.info("%s", item)
    # logger.debug(config)

    # 3. LEE DATA
    from fire2a.raster import read_raster

    data_list = []
    for item in config:
        data, info = read_raster(str(item["filename"]))
        item.update(info)
        data_list += [data]

    # 4. VALIDAR 2d todos mismo shape
    height, width = check_shapes(data_list)

    # 5. lista[mapas] -> OBSERVACIONES
    all_observations = np.column_stack([data.ravel() for data in data_list])

    # 6. if all rasters are nodata then mask out 
    nodatas = [ item["NoDataValue"] for item in config ]
    nodata_mask = np.all(all_observations == nodatas, axis=1)
    logger.info("All rasters NoData: %s pixels", nodata_mask.sum())
    observations = all_observations[~nodata_mask]

    # 7. nodata -> 0 
    for col, nd in zip(observations.T, nodatas):
        col[col == nd ] = 0

    # scaling
    # 8. PIPELINE
    scaled, pipe, feat_names = pipelie(observations, config)
    assert observations.shape[0] == scaled.shape[0]
    assert observations.shape[1] <= scaled.shape[1]

    # weights
    values_weights = []
    for name in feat_names:
        for item in config:
            if name.startswith(item["name"]):
                values_weights += [item["value_weight"]]

    # capacities
    # ["<=", "leq", "ub"]
    # [">=", "geq", "lb"]
    cap_cfg= [ {"idx": i,
            "name": item["filename"].stem,
             "cap": observations[:,i].sum() * item["capacity_ratio"],
             "sense": item["capacity_sense"]}
              for i,item in enumerate(config) if item["capacity_ratio"] is not np.nan
           ]
    cap_data = observations[:,[itm["idx"] for itm in cap_cfg]] 

    # 9. PYOMO MODEL & SOLVE
    from pyomo import environ as pyo

    # m model
    m = pyo.ConcreteModel("MultiObjectiveKnapsack")
    # sets
    m.V = pyo.RangeSet(0, scaled.shape[1] - 1)
    scaled_n, scaled_v = scaled.nonzero()
    m.NV = pyo.Set( initialize=[(n,v) for n,v in zip(scaled_n, scaled_v)] )
    m.W = pyo.RangeSet(0, len(cap_cfg) - 1)
    cap_data_n, cap_data_v = cap_data.nonzero()
    m.NW = pyo.Set( initialize=[(n,w) for n,w in zip(cap_data_n, cap_data_v)] )
    both_nv_nw = list(set(scaled_n) | set(cap_data_n))
    both_nv_nw.sort()
    m.N = pyo.Set( initialize=both_nv_nw)
    # parameters
    m.scaled_values = pyo.Param(m.NV, initialize=lambda m, n, v: scaled[n,v])
    m.values_weight = pyo.Param(m.V, initialize=values_weights)
    m.total_capacity = pyo.Param(m.W, initialize=[itm["cap"] for itm in cap_cfg])
    m.capacity_weight = pyo.Param(m.NW, initialize=lambda m, n, w: cap_data[n,w])
    # variables
    m.X = pyo.Var(m.N, within=pyo.Binary)
    # constraints
    def capacity_rule(m, w):
        if cap_cfg[w]["sense"] in ["<=", "leq", "ub"]:
            return sum(m.X[n] * m.capacity_weight[n,w] for n,ww in m.NW if ww==w) <= m.total_capacity[w]
        elif cap_cfg[w]["sense"] in [">=", "geq", "lb"]:
            return sum(m.X[n] * m.capacity_weight[n,w] for n,ww in m.NW if ww==w) >= m.total_capacity[w]
        else:
            logger.critical("Wrong sense for capacity constraint")
            return pyo.Constraint.Skip
    m.capacity = pyo.Constraint(m.W, rule=capacity_rule)

    # objective
    m.obj = pyo.Objective(
        expr=sum(m.values_weight[v] * sum(m.X[n]* m.scaled_values[n,v] for n,vv in m.NV if vv==v) for v in m.V),
        sense=pyo.maximize,
    )

    from pyomo.opt import SolverFactory
    opt = SolverFactory("cplex")
    results = opt.solve(m, tee=True)
    soln = np.array([pyo.value(m.X[i], exception=False) for i in m.X], dtype=float)
    slacks = m.capacity[:].slack()
    [ print(f"{itm['name']} cap:{itm['cap']} sense:{itm['sense']} slack:{slacks[i]}") for i,itm in enumerate(cap_cfg) ]
    m.obj()