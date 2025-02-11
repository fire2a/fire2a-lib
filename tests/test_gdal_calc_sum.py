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
from numpy import ndarray
from osgeo.gdal import Dataset, Open
from osgeo.gdal_array import DatasetReadAsArray
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


@pytest.mark.parametrize("outfile", [None, "outfile.tif"])
@pytest.mark.parametrize(
    "weights",
    [[1.0, 2.0], [1.0, -1.0], [0.0, 0.0], None],
)
def test_cli(weights, outfile, setup_test_assets):
    """
    python -m fire2a.raster.gdal_calc_sum -w 1.0 2.0 -- fuels.tif fuels.tif
    python -m fire2a.raster.gdal_calc_sum -w 1.0 2.0 -o outfile.tif fuels.tif fuels.tif
    python -m fire2a.raster.gdal_calc_sum -w 1.0 -1.0 -- fuels.tif fuels.tif
    python -m fire2a.raster.gdal_calc_sum -w 1.0 -1.0 -o outfile.tif fuels.tif fuels.tif
    python -m fire2a.raster.gdal_calc_sum -w 0.0 0.0 -- fuels.tif fuels.tif
    python -m fire2a.raster.gdal_calc_sum -w 0.0 0.0 -o outfile.tif fuels.tif fuels.tif
    python -m fire2a.raster.gdal_calc_sum fuels.tif fuels.tif
    python -m fire2a.raster.gdal_calc_sum -o outfile.tif fuels.tif fuels.tif
    """
    with MonkeyPatch.context() as mp:
        mp.chdir(setup_test_assets)
        # assert setup_test_assets == Path().cwd()

        infile = "fuels.tif"
        infile2 = "fuels.tif"

        cmd = [
            "python",
            "-m",
            "fire2a.raster.gdal_calc_sum",
        ]
        if weights:
            cmd += [
                "-w",
                *map(str, weights),
            ]
        if outfile:
            cmd += [
                "-o",
                outfile,
            ]
        elif weights:
            cmd += [
                "--",
            ]
        cmd += [
            infile,
            infile2,
        ]
        print("command:", " ".join(cmd))
        result = run_cli(cmd)
        # print(cmd, result.stdout, result.stderr, sep="\n")
        assert result.returncode == 0, print(result.stdout, result.stderr)
        if outfile:
            assert Path(outfile).exists(), f"{outfile=} does not exist.\n{result.stdout=}\n{result.stderr=}"
            ds = Open(str(outfile))
        else:
            assert Path("outfile.tif").exists(), f"outfile.tif does not exist.\n{result.stdout=}\n{result.stderr=}"
            ds = Open(str("outfile.tif"))
        assert isinstance(ds, Dataset), f"{ds=}"
        assert isinstance(DatasetReadAsArray(ds), ndarray), f"{DatasetReadAsArray(ds)=}"


@pytest.mark.parametrize("outfile", [None, "outfile.tif"])
@pytest.mark.parametrize(
    "weights",
    [[1.0, 2.0], [1.0, -1.0], [0.0, 0.0], None],
)
def test_main(weights, outfile, setup_test_assets):
    """
    ['-r', '-w', '1.0', '2.0', '--', 'fuels.tif', 'fuels.tif']
    ['-r', '-w', '1.0', '2.0', '-o', 'outfile.tif', 'fuels.tif', 'fuels.tif']
    ['-r', '-w', '1.0', '-1.0', '--', 'fuels.tif', 'fuels.tif']
    ['-r', '-w', '1.0', '-1.0', '-o', 'outfile.tif', 'fuels.tif', 'fuels.tif']
    ['-r', '-w', '0.0', '0.0', '--', 'fuels.tif', 'fuels.tif']
    ['-r', '-w', '0.0', '0.0', '-o', 'outfile.tif', 'fuels.tif', 'fuels.tif']
    ['-r', 'fuels.tif', 'fuels.tif']
    ['-r', '-o', 'outfile.tif', 'fuels.tif', 'fuels.tif']
    """
    from fire2a.raster.gdal_calc_sum import main

    with MonkeyPatch.context() as mp:
        mp.chdir(setup_test_assets)
        # assert setup_test_assets == Path().cwd()

        infile = "fuels.tif"
        infile2 = "fuels.tif"

        cmd = ["-r"]
        if weights:
            cmd += [
                "-w",
                *map(str, weights),
            ]
        if outfile:
            cmd += [
                "-o",
                outfile,
            ]
        elif weights:
            cmd += [
                "--",
            ]
        cmd += [
            infile,
            infile2,
        ]
        # print("command:", " ".join(cmd))
        print(f"{cmd=}")
        ds = main(cmd)
        assert isinstance(ds, Dataset)
        assert isinstance(DatasetReadAsArray(ds), ndarray)

        #    func = lambda minimum, maximum: f"(A-{minimum})/({maximum} - {minimum})"
        #    ds = calc(func, **vars(args))
