#!python3

import os
from datetime import datetime

from fire2a.meteo import generate

date = datetime.now()
rowres = 60
numrows = 10
numsims = 15
x = -36
y = -72


def test_create_dir(tmp_path):
    outdir = tmp_path / "sub"
    outdir.mkdir()
    generate(x, y, date, rowres, numrows, numsims, outdir)
    assert outdir.is_dir()


# Corrobora numero de archivos
def test_create_weather(tmp_path):
    outdir = tmp_path / "sub"
    outdir.mkdir()
    generate(x, y, date, rowres, numrows, numsims, outdir)
    assert len(os.listdir(outdir)) == numsims


# Corrobra filas en el archivo
def test_weather_lenght(tmp_path):
    outdir = tmp_path / "sub"
    outdir.mkdir()
    generate(x, y, date, rowres, numrows, numsims, outdir)
    for afile in outdir.iterdir():
        assert len(afile.read_text().splitlines()) - 1 == numrows
