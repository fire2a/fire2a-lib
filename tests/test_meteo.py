#!python3
import pytest

from datetime import datetime
from fire2a.meteo import generate
import os

date = datetime.now()
rowres = 60
numrows = 20
numsims = 15

def test_create_dir(tmp_path):
    outdir = tmp_path / "sub"
    outdir.mkdir()
    generate(-36, -73, date, rowres, numrows, numsims, outdir)
    assert outdir.is_dir()

#Corrobora numero de archivos
def test_create_weather(tmp_path):
    outdir = tmp_path / "sub"
    outdir.mkdir()
    generate(-36, -73, date, rowres, numrows, numsims, outdir)
    assert len(os.listdir(outdir)) == numsims

#Corrobra filas en el archivo
def test_weather_lenght(tmp_path):
    outdir = tmp_path / "sub"
    outdir.mkdir()
    generate(-36, -73, date, rowres, numrows, numsims, outdir)
    for x in os.listdir(outdir):
        with open(outdir/x,'r') as file:
            assert len(file.readlines()) - 1 == numrows
            file.close()