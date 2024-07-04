#!python3
"""👋🌎
Some functions related to kitral weather scenario creation.
"""
__author__ = "Caro"
__revision__ = "$Format:%H$"

from datetime import timedelta
from datetime import datetime

import numpy as np

from pandas import DataFrame
from pandas import read_csv
from random import randint

from pathlib import Path

aqui = Path(__file__).parent

# Ruta a la lista de estaciones
ruta_stn= aqui / "./DB_DMC/Estaciones.csv"
#Ruta a los datos de estaciones
ruta_data=aqui / "./DB_DMC/"

#rng = default_rng()


def generate(x, y, start_datetime, rowres, numrows, numsims, outdir):
    """dummy generator function
    Args:
        x (float): x-coordinate of the weather station, EPSG 4326
        y (float): y-coordinate of the weather station, EPSG 4326
        starttime (starttime): start datetime of the weather scenario label
        rowres (int): time resolution in minutes (not implemented yet)
        numrows (int): number of rows in the weather scenario
        numsims (int): number of weather scenarios
        outdir (Path): output directory
    Return:
        retval (int): 0 if successful, 1 otherwise, 2...
        outdict (dict): output dictionary at least 'filelist': list of filenames created
    """
    # try:
    #print(datetime.now())
    
    if not outdir.is_dir():
        outdir.mkdir()
    
    # numrows_width = len(str(numrows))
    numsims_width = len(str(numsims))
    
    def file_name(i, numsims):
        if numsims > 1:
            return f"Weather{i+1}.csv"
        return "Weather.csv"
    
    def scenario_name(i, numsims):
        if numsims > 1:
            return f"DMC {i+1}"
        return "DMC"
    
    def distancia(fila, y_i, x_i):
        if y_i == fila["lat"] and x_i == fila["lon"]:
            return 0
        return np.sqrt((fila["lat"] - x_i) ** 2 + (fila["lon"] - y_i) ** 2)
    
    def meteo_to_c2f(alfa):
        if alfa >= 0 and alfa < 180:
            return alfa + 180
        elif alfa >= 180 and alfa <= 360:
            return alfa - 180
        return np.nan


    dn=3
    list_stn = read_csv(ruta_stn)
    list_stn["Distancia"] = list_stn.apply(distancia, args=(y, x), axis=1)  # calculo distancia
    stn= list_stn.sort_values(by=["Distancia"]).head(dn)["nombre"].tolist()
    
    meteo=[]
    for i in range(len(stn)):
        station = stn[i] + ".csv"
        data = read_csv(ruta_data / station, sep=',')  # , index_col=0, parse_dates=True)
        meteo.append(data)
    
    
    st_time= datetime(datetime.now().year,datetime.now().month,datetime.now().day,12,0,0)
    filelist = []
    for i in range(numsims):
        scenario = [scenario_name(i, numsims) for k in range(numrows)]
        dt = [st_time + timedelta(hours=k) for k in range(numrows)]
    
        j = randint(0, dn - 1)
        m = randint(0, len(meteo[j]["TMP"]) - numrows)
    
        wd = []
        for x in meteo[j]["WD"].iloc[m:m + numrows].tolist():
            wd.append(meteo_to_c2f(x))
    
        ws = meteo[j]["WS"].iloc[m:m + numrows].tolist()
        tmp = meteo[j]["TMP"].iloc[m:m + numrows].tolist()
        rh = meteo[j]["RH"].iloc[m:m + numrows].tolist()
    
        WS = np.array(ws).round(2)
        WD = np.array(wd).round(2) % 360
        TMP = np.array(tmp).round(2)
        RH = np.array(rh).round(2)

        df = DataFrame(
        np.vstack((scenario, dt, WS, WD, TMP, RH)).T,
            columns=["Scenario", "datetime", "WS", "WD", "TMP", "RH"],
        )
        tmpfile = outdir / file_name(i, numsims)
        filelist += [tmpfile.name]
        df.to_csv(tmpfile, header=True, index=False)
    return 0, {"filelist": filelist}
    #     return 0, {"filelist": filelist}
    # except Exception as e:
    #     return 1, {"filelist": filelist, "exception": e}


if __name__ == "__main__":
    #
    # TEMPORARY TESTS
    #
    #from datetime import datetime

    date = datetime.now()
    rowres = 60
    numrows = 20
    numsims = 15
    from pathlib import Path

    outdir = Path("./weather")
    generate(-36, -73, date, rowres, numrows, numsims, outdir)
