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


def test_agg(request, tmp_path):
    """this test checks if the Data.csv file is generated from a fire Instance Folder
    TODO add more raster layer
    """
    from fire2a.agglomerative_clustering import main

    assets_path = request.config.rootdir / "tests" / "agglomerative_clustering"

    for afile in ["cbd.tif", "cbh.tif", "elevation.tif", "fuels.tif"]:
        copy(assets_path / afile, tmp_path)

    copy(assets_path / "config.toml", tmp_path)

    with MonkeyPatch.context() as mp:
        mp.chdir(tmp_path)
        label_map, pipeline = main(["-s", "-n", "10", "config.toml"])
        assert Path("output.tif").is_file()

    assert label_map.shape == (597, 658)

    uniq, coun = np.unique(label_map, return_counts=True)

    assert all(uniq == np.arange(10))
    assert all(coun == np.array([78912, 28685, 62451, 27905, 14933, 152605, 6218, 14799, 4848, 1470]))

    # from IPython.terminal.embed import InteractiveShellEmbed

    # InteractiveShellEmbed()()
