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


# Corrobora numero de archivos
def test_create_weather(tmp_path):
    print(generate(x, y, date, rowres, numrows, numsims, tmp_path))
    print(f"{tmp_path=}, {os.listdir(tmp_path)=}")
    assert len(os.listdir(tmp_path)) == numsims, "Numero de archivos incorrecto"


# Corrobra filas en el archivo
def test_weather_lenght(tmp_path):
    print(generate(x, y, date, rowres, numrows, numsims, tmp_path))
    print(f"{tmp_path=}, {os.listdir(tmp_path)=}")
    for afile in tmp_path.iterdir():
        assert len(afile.read_text().splitlines()) - 1 == numrows, "Numero de filas incorrecto"


def test_invert_wind():
    from numpy import linspace

    from fire2a.meteo import barlo_sota, meteo_to_c2f

    for a in linspace(1, 360, 12):
        assert barlo_sota(a) - meteo_to_c2f(a) < 0.01, "Conversion incorrecta"
    # for a in np.linspace(-360, 720, 12):
    #     print(a, barlo_sota(a), meteo_to_c2f(a))
