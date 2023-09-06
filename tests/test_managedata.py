#!python3
"""
managedata test
"""
__author__ = "David Palacios Meneses"
__version__ = "v0.0.1+0-gf866f08"
from pathlib import Path

from pandas import read_csv

global assets_path
assets_path = Path("tests", "assets")


def test_DataCsv_isGenerated():
    """this test checks if the Data.csv file is generated
    TODO generate the file in a temporary folder https://docs.pytest.org/en/7.4.x/how-to/tmp_path.html#tmp-path-handling
    """
    from fire2a.managedata import GenDataFile

    output_file = assets_path / "Data.csv"
    # prepare
    if output_file.exists():
        output_file.unlink()
    GenDataFile(assets_path.absolute(), "S")
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
    output_file.unlink()
