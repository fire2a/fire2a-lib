#!python3
"""
agglomerative tests
"""
__author__ = "Fernando Badilla"
import os
from pathlib import Path
from shutil import copy

import numpy as np
from pytest import MonkeyPatch


def test_agg0(request, tmp_path):
    """-n 10 config.toml"""
    from osgeo import ogr

    from fire2a.agglomerative_clustering import main

    assets_path = request.config.rootdir / "tests" / "agglomerative_clustering"

    for afile in ["cbd.tif", "cbh.tif", "elevation.tif", "fuels.tif"]:
        copy(assets_path / afile, tmp_path)

    copy(assets_path / "config.toml", tmp_path)

    with MonkeyPatch.context() as mp:
        mp.chdir(tmp_path)
        retval = main(["-n", "10", "config.toml"])
        assert retval == 0
        assert Path("output.gpkg").is_file()

        ds = ogr.Open("output.gpkg")
        lyr = ds.GetLayer()
        assert lyr.GetFeatureCount() == 10


def test_agg1(request, tmp_path):
    """-n 10 -s config.toml"""
    from fire2a.agglomerative_clustering import main

    assets_path = request.config.rootdir / "tests" / "agglomerative_clustering"

    for afile in ["cbd.tif", "cbh.tif", "elevation.tif", "fuels.tif"]:
        copy(assets_path / afile, tmp_path)

    copy(assets_path / "config.toml", tmp_path)

    with MonkeyPatch.context() as mp:
        mp.chdir(tmp_path)
        label_map, pipe1, pipe2 = main(["-s", "-n", "10", "config.toml"])
        assert Path("output.gpkg").is_file()

    assert label_map.shape == (597, 658)

    uniq, coun = np.unique(label_map, return_counts=True)

    assert all(uniq == np.arange(10))
    # assert all(coun == np.array([78912, 28685, 62451, 27905, 14933, 152605, 6218, 14799, 4848, 1470]))

    # from IPython.terminal.embed import InteractiveShellEmbed

    # InteractiveShellEmbed()()
