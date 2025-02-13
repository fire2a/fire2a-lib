#!python
"""
pytest
    from IPython.terminal.embed import InteractiveShellEmbed

    InteractiveShellEmbed()()
"""
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from numpy import array, ndarray
from osgeo.gdal import Dataset
from osgeo.gdal_array import DatasetReadAsArray, LoadFile
from pytest import MonkeyPatch

from fire2a.raster.gdal_calc_sum import main

# Define the path to the test assets directory
ASSETS_DIR = Path(__file__).parent / "assets_gdal_calc"


# Fixture to copy test assets to a temporary directory
@pytest.fixture
def setup_test_assets(tmp_path):
    # Copy the test assets to the temporary directory
    for asset in ASSETS_DIR.iterdir():
        shutil.copy(asset, tmp_path)
    return tmp_path


@pytest.mark.parametrize("infiles", [["fuels.tif"], ["fuels.tif", "elevation.tif"]])
@pytest.mark.parametrize(
    "weights",
    [[2.0], [1.0, -1.0], [0.0, 0.0], None],
)
def test_main(weights, infiles, setup_test_assets):
    """ """
    if weights:
        if len(infiles) != len(weights):
            print("Invalid input")
            return
        def_weights = weights
    else:
        def_weights = [1.0] * len(infiles)

    print(f"{infiles=}, {weights=}, {def_weights=}")

    with MonkeyPatch.context() as mp:
        mp.chdir(setup_test_assets)
        # assert setup_test_assets == Path().cwd()

        # from IPython.terminal.embed import InteractiveShellEmbed
        # InteractiveShellEmbed()()
        # numpy result
        data = np.array([LoadFile(infile) for infile in infiles])
        np_shape = data.shape[1:]
        data = data.reshape(data.shape[0], -1)
        np_result = np.dot(def_weights, data)
        print(f"{np_result.shape=}, {np_shape=}")

        # gdal result
        cmd = ["-r"]
        if weights:
            cmd += [
                "-w",
                *map(str, weights),
                "--",
            ]
        cmd += infiles
        print(f"{cmd=}")
        ds = main(cmd)
        assert isinstance(ds, Dataset)
        gdal_result = DatasetReadAsArray(ds)
        assert gdal_result.shape == np_shape
        print(f"{gdal_result.shape=}")
        assert isinstance(gdal_result, np.ndarray)
        gdal_result = gdal_result.flatten()
        # gdal_result = gdal_result.reshape(-1)

        # compare results
        assert np.allclose(np_result[gdal_result != -9999], gdal_result[gdal_result != -9999])
