#!python3
"""
managedata test
"""
__author__ = "David Palacios Meneses"
__version__ = 'v0.0.1-42-g648b7fd-dirty'
from pathlib import Path
from shutil import copy

from pandas import read_csv


def test_DataCsv_isGenerated(request, tmp_path):
    """this test checks if the Data.csv file is generated from a fire Instance Folder
    TODO add more raster layer
    """
    from fire2a.managedata import GenDataFile

    assets_path = request.config.rootdir / "tests" / "manage_data"
    copy(assets_path / "spain_lookup_table.csv", tmp_path)
    copy(assets_path / "fuels.asc", tmp_path)
    GenDataFile(tmp_path, "S")
    output_file = tmp_path / "Data.csv"
    assert output_file.exists()
    df = read_csv(output_file)
    assert all(
        df.columns
        == [
            "fueltype",
            "lat",
            "lon",
            "elev",
            "ws",
            "waz",
            "ps",
            "saz",
            "cur",
            "cbd",
            "cbh",
            "ccf",
            "ftypeN",
            "fmc",
            "py",
        ]
    )
