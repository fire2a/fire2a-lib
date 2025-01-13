#!/usr/bin/env python3
# fmt: off
"""👋🌎 🌲🔥
# MultiObjective Knapsack Rasters
Select the best set of pixels maximizing the sum of several rescaled and weighted rasters, minding capacity constraints.
## Usage
### Overview
1. Choose your raster files (absolute path or relative to the script execution directory)
2. Configure, for values: rescaling strategies (minmax, onehot, standard, robust or pass) and absolute weights (any real number)
3. Configure, for capacites: capacity sense (lower or upper bound) and ratio (between -1 and 1)
4. Set output options (raster, authid, geotransform, plots, ...)
### Command line execution
    ```bash
    # get interface help
    python -m fire2a.knapsack --help

    # run (in a qgis/gdal aware python environment)
    python -m fire2a.knapsack [config.toml]
    ```
### Script integration
    ```python
    from fire2a.knapasack import main
    solution, model, instance, args = main(["--script","config.toml"])
    ```
### Preparation
#### 1. Choose your raster files
- Any [GDAL compatible](https://gdal.org/en/latest/drivers/raster/index.html) raster will be read
- Mind that any nodata value will exclude that pixel from the optimization (this can be overriden but not recommended, see `--exclude_nodata`, specially for weight constraining rasters)
- A good practice is to place them all in the same directory where the script will be executed
- "Quote them" if they have any non alphanumerical chars [a-zA-Z0-9]

#### 2. Preprocessing configuration
See the `config.toml` file for example of the configuration of the preprocessing steps. The file is structured as follows:

```toml
["a_filename.tif"]
value_rescaling = "onehot"
value_weight = 0.5

["b_filename.tif"]
capacity_sense = "<="
capacity_ratio = 0.1
```
This example states the raster `filename.tif` values will be rescaled using the `OneHot` strategy, then multiplied by 0.5 in the sought objective; Also that at leat 10% of its weighted pixels must be selected. 

1. __value_rescaling__
   - can be minmax, onehot, standard, robust or pass for no rescaling.
   - minmax (default) and onehot scale into [0,1], standard and robust not, mix them adjusting the value_weight
   - [MinMax](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html): (x-min)/(max-min)
   - [Standard Scaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html): (x-mean)/stddev
   - [Robust Scaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.RobustScaler.html): same but droping the tails of the distribution
   - [OneHot Encoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html): __for CATEGORICAL DATA__

2. __value_weight__
   - can be any real number, although zero does not make sense
   - positive maximizes, negative minimizes

3. __capacity_sense__
    - can be >=, ≥, ge, geq, lb for lower bound, <=, ≤, le, leq, ub for upper bound
    - default is upper bound

4. __capacity_ratio__
   - can be any real number, between -1 and 1
   - is proportional to the sum of all values of the pixels in the raster, meaning if all values are the same it represents the proportion of pixels to be selected

#### 3. Other options
- __output_raster__ (default "")
- __authid__ (default "EPSG:3857")
- __geotransform__ (default "(0, 1, 0, 0, 0, -1)")
- __plots__ (default False) saves 3 .png files to the same output than the raster, showing the data, the scaled data and the weighted scaled data. Great for debugging but Can really slow down the process.
- __exclude_nodata__ (default "any") if "any" layer is nodata, it's excluded. It can be relaxed by setting "all" (layers must be nodata to be excluded) can cause great problems with the capacity rasters selecting pixels that weight 0.
- __script__ (default False) only for integrating into other scripts
- __no_write__ (default False)
- __verbose__ (default 0)

"""
# fmt: on
import logging
import sys
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from pyomo import environ as pyo

from fire2a.utils import read_toml

logger = logging.getLogger(__name__)

allowed_ub = ["<=", "≤", "le", "leq", "ub"]
allowed_lb = [">=", "≥", "ge", "geq", "lb"]
config_allowed = {
    "value_rescaling": ["minmax", "standard", "robust", "onehot", "pass"],
    "capacity_sense": allowed_ub + allowed_lb,
}


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
        "passthrough": "passthrough",
    }

    # 2. PIPELINE
    pipe = Pipeline(
        [
            (
                "scaler",
                ColumnTransformer(
                    [
                        (item["name"], scalers.get(item.get("value_rescaling")), [i])
                        for i, item in enumerate(config)
                        if item.get("value_rescaling")
                    ],
                    remainder="drop",
                ),
            )
        ],
        verbose=True,
    )

    # 3. FIT
    scaled = pipe.fit_transform(observations)
    feat_names = pipe.named_steps["scaler"].get_feature_names_out()
    logger.debug("Pipeline input: %s", [{itm.get("name"): itm.get("value_rescaling")} for itm in config])
    logger.debug("Pipeline output: feat_names:%s", feat_names)
    logger.debug("Pipeline output: scaled.shape:%s", scaled.shape)
    return scaled, pipe, feat_names


def arg_parser(argv=None):
    """Parse command line arguments."""
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

    parser = ArgumentParser(
        description="MultiObjective Knapsack Rasters",
        formatter_class=ArgumentDefaultsHelpFormatter,
        epilog="Full documentation at https://fire2a.github.io/fire2a-lib/fire2a/knapsack.html",
    )
    parser.add_argument(
        "config_file",
        nargs="?",
        type=Path,
        help="A toml formatted file, with a section header for each raster [file_name], with items: 'value_rescaling', 'value_weight', 'capacity_sense' and 'capacity_ratio'",
        default="config.toml",
    )
    parser.add_argument("-or", "--output_raster", help="Output raster file, warning overwrites!", default="")
    parser.add_argument(
        "-e",
        "--exclude_nodata",
        type=str,
        help="By default if 'any' layer is nodata, it's excluded. It can be relaxed by setting 'all' (layers must be nodata to be excluded)",
        choices=["any", "all"],
        default="any",
    )
    parser.add_argument("-a", "--authid", type=str, help="Output raster authid", default="EPSG:3857")
    parser.add_argument(
        "-g", "--geotransform", type=str, help="Output raster geotransform", default="(0, 1, 0, 0, 0, -1)"
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
    parser.add_argument(
        "-p",
        "--plots",
        action="store_true",
        help="Activate the plotting routines (saves 3 .png files to the same output than the raster)",
        default=False,
    )
    args = parser.parse_args(argv)
    args.geotransform = tuple(map(float, args.geotransform[1:-1].split(",")))
    if Path(args.config_file).is_file() is False:
        parser.error(f"File {args.config_file} not found")
    return args


def aplot(data: np.ndarray, title: str, series_names: list[str], outpath: Path, show=False):  # __name__ == "__main__"
    """
    names = [itm["name"] for i, itm in enumerate(config)]
    outpath = Path(args.output_raster).parent
    fname =  outpath / "observations.png"
    """
    if not isinstance(data, np.ndarray):
        data = data.toarray()
    if not isinstance(data[0], np.ndarray):
        for itm in data:
            itm = itm.toarray()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle(title)
    ax[0].violinplot(data, showmeans=False, showmedians=True, showextrema=True)
    ax[0].set_title("violinplot")
    ax[0].set_xticks(range(1, len(series_names) + 1), series_names)
    ax[1].boxplot(data)
    ax[1].set_title("boxplot")
    ax[1].set_xticks(range(1, len(series_names) + 1), series_names)
    if show:
        plt.show()
    fname = outpath / (title + ".png")
    plt.savefig(fname)
    plt.close()


def get_model(scaled=None, values_weights=None, cap_cfg=None, cap_data=None, **kwargs):

    # m model
    m = pyo.ConcreteModel("MultiObjectiveKnapsack")
    # sets
    m.V = pyo.RangeSet(0, scaled.shape[1] - 1)
    scaled_n, scaled_v = scaled.nonzero()
    m.NV = pyo.Set(initialize=[(n, v) for n, v in zip(scaled_n, scaled_v)])
    m.W = pyo.RangeSet(0, len(cap_cfg) - 1)
    cap_data_n, cap_data_v = cap_data.nonzero()
    m.NW = pyo.Set(initialize=[(n, w) for n, w in zip(cap_data_n, cap_data_v)])
    both_nv_nw = list(set(scaled_n) | set(cap_data_n))
    both_nv_nw.sort()
    m.N = pyo.Set(initialize=both_nv_nw)
    # parameters
    m.scaled_values = pyo.Param(m.NV, initialize=lambda m, n, v: scaled[n, v])
    m.values_weight = pyo.Param(m.V, initialize=values_weights)
    m.total_capacity = pyo.Param(m.W, initialize=[itm["cap"] for itm in cap_cfg])
    m.capacity_weight = pyo.Param(m.NW, initialize=lambda m, n, w: cap_data[n, w])
    # variables
    m.X = pyo.Var(m.N, within=pyo.Binary)

    # constraints
    def capacity_rule(m, w):
        if cap_cfg[w]["sense"] in allowed_ub:
            return sum(m.X[n] * m.capacity_weight[n, w] for n, ww in m.NW if ww == w) <= m.total_capacity[w]
        elif cap_cfg[w]["sense"] in allowed_lb:
            return sum(m.X[n] * m.capacity_weight[n, w] for n, ww in m.NW if ww == w) >= m.total_capacity[w]
        else:
            logger.critical("Skipping capacity constraint %s, %s", w, cap_cfg[w])
            return pyo.Constraint.Skip

    m.capacity = pyo.Constraint(m.W, rule=capacity_rule)
    # objective
    m.obj = pyo.Objective(
        expr=sum(m.values_weight[v] * sum(m.X[n] * m.scaled_values[n, v] for n, vv in m.NV if vv == v) for v in m.V),
        sense=pyo.maximize,
    )
    return m


def solve_pyomo(m, tee=True, solver="cplex", **kwargs):
    from pyomo.opt import SolverFactory

    opt = SolverFactory(solver)
    results = opt.solve(m, tee=tee, **kwargs)
    return opt, results


def pre_solve(argv):
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
    # dict -> list[dict]
    a, b = list(config.keys()), list(config.values())
    config = [{"name": Path(a).name, "filename": Path(a), **b} for a, b in zip(a, b)]
    for itm in config:
        if "value_rescaling" in itm:
            itm["value_rescaling"] = itm["value_rescaling"].lower()
            if itm["value_rescaling"] not in config_allowed["value_rescaling"]:
                logger.critical("Wrong value for value_rescaling in %s", itm)
                sys.exit(1)
        if "capacity_sense" in itm:
            itm["capacity_sense"] = itm["capacity_sense"].lower()
            if itm["capacity_sense"] not in config_allowed["capacity_sense"]:
                logger.critical("Wrong value for capacity_sense in %s", itm)
                sys.exit(1)
        logger.debug(itm)

    # 2.1 CHECK PAIRS + defaults
    # vr : value_rescaling
    # vw : value_weight
    # cr : capacity_ratio
    # cs : capacity_sense
    for itm in config:
        if vr := itm.get("value_rescaling"):
            # vr & !vw => vw = 1
            if "value_weight" not in itm:
                logger.warning(
                    "value_rescaling without value_weight for item: %s\n ASSUMING VALUE WEIGHT IS 1", itm["name"]
                )
                itm["value_weight"] = 1
            if vr == "pass":
                itm["value_weight"] = "passthrough"
        # !vr & vw => vr = passthrough
        elif "value_weight" in itm:
            logger.warning("value_weight without value_rescaling for item: %s\n DEFAULTING TO MINMAX", itm["name"])
            itm["value_rescaling"] = "minmax"
        if cr := itm.get("capacity_ratio"):
            # cr not in (-1,1) =>!<=
            if not (-1 < cr < 1):
                logger.critical("Wrong value for capacity_ratio in %s, should be (-1,1)", itm)
                sys.exit(1)
            # cr & !cs => cs = ub
            if "capacity_sense" not in itm:
                logger.warning(
                    "capacity_ratio without capacity_sense for item: %s\n ASSUMING SENSE IS UPPER BOUND", itm["name"]
                )
                itm["capacity_sense"] = "ub"
        # !cr & cs =>!<=
        elif "capacity_sense" in itm:
            logger.critical("capacity_sense without capacity_ratio for item: %s", itm["name"])
            sys.exit(1)

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

    # 6. if all|any rasters are nodata then mask out
    nodatas = [item["NoDataValue"] for item in config]
    if args.exclude_nodata == "any":
        nodata_mask = np.any(all_observations == nodatas, axis=1)
    if args.exclude_nodata == "all":
        nodata_mask = np.all(all_observations == nodatas, axis=1)

    logger.info("Sum of all rasters NoData: %s pixels", nodata_mask.sum())
    observations = all_observations[~nodata_mask]

    # 7. nodata -> 0
    if args.exclude_nodata == "all":
        for col, nd in zip(observations.T, nodatas):
            col[col == nd] = 0

    if args.plots:
        aplot(
            observations, "observations", [itm["name"] for itm in config], Path(args.output_raster).parent, show=False
        )

    # scaling
    # 8. PIPELINE
    scaled, pipe, feat_names = pipelie(observations, config)
    # assert observations.shape[0] == scaled.shape[0]
    # assert observations.shape[1] >= scaled.shape[1]
    logger.info(f"{observations.shape=}")
    logger.info(f"{scaled.shape=}")

    if args.plots:
        # exclude multiple ocurrences of onehots
        first_occurrences = {}
        for i, fn in enumerate(feat_names):
            for bn in [itm["name"] for itm in config]:
                if fn.startswith(bn) and bn not in first_occurrences:
                    first_occurrences[bn] = i
                    break
        idxs = list(first_occurrences.values())
        aplot(scaled[:, idxs], "scaled", feat_names[idxs], Path(args.output_raster).parent, show=False)

    # weights
    values_weights = []
    for name in feat_names:
        for item in config:
            if name.startswith(item["name"]):
                values_weights += [item["value_weight"]]
    values_weights = np.array(values_weights)
    logger.info(f"{values_weights.shape=}")

    if args.plots:
        from scipy.sparse import csr_matrix

        values_weights_diag = csr_matrix(np.diag(values_weights[idxs]))
        scaled_weighted = scaled[:, idxs].dot(values_weights_diag)

        aplot(
            scaled_weighted,
            "scaled_weighted",
            feat_names[idxs],
            Path(args.output_raster).parent,
            show=False,
        )

    # capacities
    # "name": item["filename"].name.replace('.','_'),
    cap_cfg = [
        {
            "idx": i,
            "name": item["filename"].stem,
            "cap": observations[:, i].sum() * item["capacity_ratio"],
            "sense": item["capacity_sense"],
        }
        for i, item in enumerate(config)
        if "capacity_ratio" in item
    ]
    cap_data = observations[:, [itm["idx"] for itm in cap_cfg]]
    instance = {
        "scaled": scaled,
        "values_weights": values_weights,
        "cap_cfg": cap_cfg,
        "cap_data": cap_data,
        "feat_names": feat_names,
        "height": height,
        "width": width,
        "nodata_mask": nodata_mask,
        "all_observations": all_observations,
        "observations": observations,
        "pipe": pipe,
        "nodatas": nodatas,
    }
    return instance, args


def main(argv=None):
    """This main is split in 3 parts with the objective of being called from within QGIS fire2a's toolbox-plugin.
    Nevertheless, it can be called from the command line:
    ```bash
    python -m fire2a.knapsack [config.toml]
    ```
    Or integrated into other scripts.
    from fire2a.knapasack import main
    ```python
    soln, m, instance, args = main(["config.toml"])
    ```
    """
    # 0..9 PRE
    instance, args = pre_solve(argv)

    # 9. PYOMO MODEL
    m = get_model(**instance)

    # 10. PYOMO SOLVE
    opt, results = solve_pyomo(m, tee=True, solver="cplex")
    instance["opt"] = opt
    instance["results"] = results

    # 11. POST
    return post_solve(m, args=args, **instance)


def post_solve(
    m,
    args=None,
    scaled=None,
    values_weights=None,
    cap_cfg=None,
    cap_data=None,
    feat_names=None,
    height=None,
    width=None,
    nodata_mask=None,
    all_observations=None,
    observations=None,
    pipe=None,
    nodatas=None,
    **kwargs,
):
    soln = np.array([pyo.value(m.X[i], exception=False) for i in m.X], dtype=np.float32)
    logger.info("solution pseudo-histogram: %s", np.unique(soln, return_counts=True))
    soln[~soln.astype(bool)] = 0

    try:
        slacks = m.capacity[:].slack()
        logger.info("objective: %s", m.obj(exception=False))
    except Exception as e:
        logger.error(e)
        slacks = [0] * len(cap_cfg)

    if not isinstance(scaled, np.ndarray):
        scaled = scaled.toarray()
    vx = np.matmul(scaled.T, soln)

    logger.info("Values per objective:")
    for f, v in zip(feat_names, vx):
        logger.info(f"{f}\t\t{v:.4f}")

    if args.plots:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle("solution")
        ax[0].set_title("positive objectives, abs(w*v*x)")
        ax[0].pie(np.absolute(vx * values_weights), labels=feat_names, autopct="%1.1f%%")

        cap_ratio = [slacks[i] / itm["cap"] for i, itm in enumerate(cap_cfg)]
        ax[1].set_title("capacity slack ratios")
        ax[1].bar([itm["name"] for itm in cap_cfg], cap_ratio)

        # if __name__ == "__main__":
        #      plt.show()
        plt.savefig(Path(args.output_raster).parent / "solution.png")
        plt.close()

    logger.info("Capacity slack:")
    for i, itm in enumerate(cap_cfg):
        logger.info(f"{i}: name:{itm['name']} cap:{itm['cap']} sense:{itm['sense']} slack:{slacks[i]}")

    if args.script:
        instance = {
            "scaled": scaled,
            "values_weights": values_weights,
            "cap_cfg": cap_cfg,
            "cap_data": cap_data,
            "feat_names": feat_names,
            "height": height,
            "width": width,
            "nodata_mask": nodata_mask,
            "all_observations": all_observations,
            "observations": observations,
            "pipe": pipe,
            "nodatas": nodatas,
        }
        return soln, m, instance, args
    else:
        # 10. WRITE OUTPUTS
        from fire2a.raster import write_raster

        if args.output_raster:
            # put soln into the original shape of all_observations using the reversal of nodata_mask
            data = np.zeros(height * width, dtype=np.float32) - 9999
            data[~nodata_mask] = soln
            data = data.reshape(height, width)
            if not write_raster(
                data,
                outfile=str(args.output_raster),
                nodata=-9999,
                authid=args.authid,
                geotransform=args.geotransform,
                logger=logger,
            ):
                logger.error("Error writing output raster")
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
