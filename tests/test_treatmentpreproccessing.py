#!python3
"""
managedata test
"""
__author__ = "David Palacios Meneses"
__version__ = "v0.0.1+0-gf866f08"
from pathlib import Path
from tempfile import NamedTemporaryFile
from pandas import read_csv

def test_ids2firebreak_csv(tmp_path):
    from fire2a.treatmentpreproccessing import bin_to_nod

    tmpfile = tmp_path / 'treatment.csv' # NamedTemporaryFile(suffix=".csv", prefix="test_treatment", dir=tmp_path, delete=True)
    vector_test = [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1]  # define set of firebreak plan
    bin_to_nod(vector_test, str(tmpfile))
    df = read_csv(tmpfile)
    assert all(df.values[0][1:] - 1 == vector_test)
