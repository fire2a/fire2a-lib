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

        infile = "fuels.tif"
        outfile = f"outfile_{method}.tif"

        # numpy result
        data = LoadFile(infile)
        mask = data != -9999
        if method == "minmax":
            data[mask] = (data[mask] - data[mask].min()) / (data[mask].max() - data[mask].min())
        elif method == "maxmin":
            data[mask] = (data[mask] - data[mask].max()) / (data[mask].min() - data[mask].max())
        elif method == "stepup":
            threshold = 30
            data[mask] = 0 * (data[mask] < threshold) + 1 * (data[mask] >= threshold)
        elif method == "stepdown":
            threshold = 30
            data[mask] = 1 * (data[mask] < threshold) + 0 * (data[mask] >= threshold)
        elif method == "bipiecewiselinear":
            a = 30
            b = 60
            data[mask] = (data[mask] - a) / (b - a)
            data[mask] = 0 * (data[mask] < 0) + 1 * (data[mask] > 1)
        elif method == "bipiecewiselinear_percent":
            a = 30
            b = 60
            rela_delta = (data[mask].max() - data[mask].min()) / 100
            real_a = rela_delta * a
            real_b = rela_delta * b
            data[mask] = (data[mask] - real_a) / (real_b - real_a)
            data[mask] = 0 * (data[mask] < 0) + 1 * (data[mask] > 1)
        elif method == "stepdown_percent":
            threshold = 30
            data[mask] = 1 * (data[mask] < threshold) + 0 * (data[mask] >= threshold)
        elif method == "stepup_percent":
            threshold = 30
            data[mask] = 0 * (data[mask] < threshold) + 1 * (data[mask] >= threshold)

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
        assert data.shape == array.shape, f"{data.shape=} {array.shape=}"
        assert np_all((0 <= array[array != -9999]) & (array[array != -9999] <= 1))
        assert np_allclose(data[array != -9999], array[array != -9999], equal_nan=True)
