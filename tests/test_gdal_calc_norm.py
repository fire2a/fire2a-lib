#!python
"""
pytest
    from IPython.terminal.embed import InteractiveShellEmbed

    InteractiveShellEmbed()()
"""
import shutil
import subprocess
from pathlib import Path

import pytest
from numpy import all as np_all
from numpy import ndarray
from osgeo.gdal import Dataset, Open
from osgeo.gdal_array import DatasetReadAsArray, LoadFile
from pytest import MonkeyPatch

# Define the path to the test assets directory
ASSETS_DIR = Path(__file__).parent / "assets_gdal_calc"


# Fixture to copy test assets to a temporary directory
@pytest.fixture
def setup_test_assets(tmp_path):
    # Copy the test assets to the temporary directory
    for asset in ASSETS_DIR.iterdir():
        shutil.copy(asset, tmp_path)
    return tmp_path


def run_cli(args, tmp_path=None):
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=tmp_path)
    return result


@pytest.mark.parametrize(
    "method",
    [
        "minmax",
        "maxmin",
        "stepup",
        "stepdown",
        "bipiecewiselinear",
        "bipiecewiselinear_percent",
        "stepdown_percent",
        "stepup_percent",
    ],
)
def test_cli(method, setup_test_assets):
    """
    python -m fire2a.raster.gdal_calc_norm -i fuels.tif -m minmax
    python -m fire2a.raster.gdal_calc_norm -i fuels.tif -m maxmin
    python -m fire2a.raster.gdal_calc_norm -i fuels.tif -m stepup 30
    python -m fire2a.raster.gdal_calc_norm -i fuels.tif -m stepdown 30
    python -m fire2a.raster.gdal_calc_norm -i fuels.tif -m bipiecewiselinear 30 60
    python -m fire2a.raster.gdal_calc_norm -i fuels.tif -m bipiecewiselinear_percent 30 60
    python -m fire2a.raster.gdal_calc_norm -i fuels.tif -m stepdown_percent 30
    python -m fire2a.raster.gdal_calc_norm -i fuels.tif -m stepup_percent 30

    """
    with MonkeyPatch.context() as mp:
        mp.chdir(setup_test_assets)

        infile = setup_test_assets / "fuels.tif"
        outfile = setup_test_assets / f"outfile_{method}.tif"

        cmd = [
            "python",
            "-m",
            "fire2a.raster.gdal_calc_norm",
            "-i",
            str(infile),
            "-o",
            str(outfile),
            "-m",
            method,
        ]
        if "step" in method:
            cmd += ["30"]
        if method in ["bipiecewiselinear", "bipiecewiselinear_percent"]:
            cmd += ["30", "60"]
        print(f"{cmd=}")
        result = run_cli(cmd)
        # print(cmd, result.stdout, result.stderr, file=open(setup_test_assets / (method + ".log"), "w"), sep="\n")
        assert result.returncode == 0, print(result.stdout, result.stderr)
        assert outfile.exists()
        ds = Open(str(outfile))
        assert isinstance(ds, Dataset)
        array = DatasetReadAsArray(ds)
        assert isinstance(array, ndarray)
        assert np_all((0 <= array[array != -9999]) & (array[array != -9999] <= 1))


@pytest.mark.parametrize(
    "method",
    [
        "minmax",
        "maxmin",
        "stepup",
        "stepdown",
        "bipiecewiselinear",
        "bipiecewiselinear_percent",
        "stepdown_percent",
        "stepup_percent",
    ],
)
def test_main(method, setup_test_assets):
    from fire2a.raster.gdal_calc_norm import main

    with MonkeyPatch.context() as mp:
        mp.chdir(setup_test_assets)

        infile = setup_test_assets / "fuels.tif"
        # inarray = LoadFile(infile)
        outfile = setup_test_assets / f"outfile_{method}.tif"

        cmd = ["-i", str(infile), "-o", str(outfile), "-m", method, "--return_dataset"]
        if "step" in method:
            cmd += ["30"]
        if method in ["bipiecewiselinear", "bipiecewiselinear_percent"]:
            cmd += ["30", "60"]
        print(f"{cmd=}")
        ds = main(cmd)
        assert isinstance(ds, Dataset)
        array = DatasetReadAsArray(ds)
        assert isinstance(array, ndarray)
        assert np_all((0 <= array[array != -9999]) & (array[array != -9999] <= 1))

        # if method == "stepup"
        #    func = lambda minimum, maximum: f"(A-{minimum})/({maximum} - {minimum})"
        #    ds = calc(func, **vars(args))
