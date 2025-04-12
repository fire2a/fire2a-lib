#!python3
"""👋🌎
Some functions related to kitral weather scenario creation.
"""
__author__ = "Caro"
__revision__ = "$Format:%H$"

from datetime import datetime, timedelta, time
from pathlib import Path

import numpy as np
import pandas as pd

# debug aqui = Path()
aqui = Path(__file__).parent
# Ruta a los datos de estaciones
ruta_data = aqui / "DB_DMC"


def file_name(i, numsims):
    if numsims > 1:
        return f"Weather{i+1}.csv"
    return "Weather.csv"


def scenario_name(i, numsims):
    if numsims > 1:
        return f"DMC_{i+1}"
    return "DMC"


def distancia(fila, y_i, x_i):
    if y_i == fila["lat"] and x_i == fila["lon"]:
        return 0
    return np.sqrt((fila["lat"] - x_i) ** 2 + (fila["lon"] - y_i) ** 2)


def meteo_to_c2f(alfa):
    if alfa >= 0 and alfa < 180:
        return round(alfa + 180, 2)
    elif alfa >= 180 and alfa <= 360:
        return round(alfa - 180, 2)
    return np.nan


def barlo_sota(a):
    return round((a + 180) % 360, 2)


time_arg=datetime(1989,1,12,16,0,0)
dir_arg=Path("./weather")


def generate(x=-36.0, y=-73.2, start_datetime=time_arg, rowres=60, numrows=12, numsims=100, percn=0.5 ,outdir=dir_arg):
    """dummy generator function
    Args:
        x (float): x-coordinate of the ignition point, EPSG 4326
        y (float): y-coordinate of the ignition point, EPSG 4326
        start_datetime (datetime): starting time of the weather scenario
        rowres (int): time resolution in minutes (not implemented yet)
        numrows (int): number of hours in the weather scenario
        numsims (int): number of weather scenarios
        percen (float): daily maximum temperature quantil
        outdir (Path): output directory
    Return:
        retval (int): 0 if successful, 1 otherwise, 2...
        outdict (dict): output dictionary at least 'filelist': list of filenames created
    """



    filelist = []
    try:

        if not outdir.is_dir():
            outdir.mkdir()

        dn = 3
        list_stn = pd.read_csv(ruta_data / "Estaciones.csv")
        list_stn["Distancia"] = list_stn.apply(distancia, args=(y, x), axis=1)  # calculo distancia
        stn = list_stn.sort_values(by=["Distancia"]).head(dn)["nombre"].tolist()


        meteos = pd.DataFrame()
        for st in stn:
            df = pd.read_csv(ruta_data / f"{st}.csv", sep=",", index_col=0, parse_dates=True)
            df1 = df["TMP"].resample('D').max()
            qn_date = df1[df1 >= df1.quantile(percn)].index
            df["station"] = st
            meteos = pd.concat([meteos, df[df.index.floor('D').isin(qn_date)].reset_index()], ignore_index=True)
        meteos["datetime"] = pd.to_datetime(meteos["datetime"], errors="coerce")
        # available days by stations
        days = meteos.groupby(meteos.datetime.dt.date).first()["station"]

        for i in range(numsims):
            # draw station and day
            cd = 0
            ch = 0
            while True:
                station = np.random.choice(stn)
                chosen_days = days[days == station]
                if chosen_days.empty:
                    if cd > 10:
                        # print("Not enough data days", cd, ch)
                        return 1, {"filelist": [], "exception": "No data in closest stations"}
                    cd += 1
                    continue
                day = np.random.choice(chosen_days.index)
                start = datetime.combine(day - timedelta(days=ch), start_datetime.time())
                chosen_meteo = meteos[(meteos["datetime"] >= start) & (meteos["station"] == station)]
                if len(chosen_meteo) < numrows:
                    if ch > len(meteos):
                        # print("Not enough data hours", cd, ch)
                        return 1, {"filelist": [], "exception": "Not enough data"}
                    ch += 1
                    continue
                break
            # take rows
            chosen_meteo = chosen_meteo.head(numrows)
            # drop station
            chosen_meteo = chosen_meteo.drop(columns=["station"])
            # wind direction
            chosen_meteo.loc[:, "WD"] = chosen_meteo["WD"].apply(barlo_sota)
            # scenario name
            chosen_meteo.loc[:, "Scenario"] = scenario_name(i, numsims)
            # datetime format
            chosen_meteo.loc[:, "datetime"] = chosen_meteo["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S")
            #chosen_meteo.loc[:, "datetime"] = [chosen_meteo["datetime"].iloc[0] + timedelta(hours=i) for i in range(numrows)]
            # reorder
            chosen_meteo = chosen_meteo[["Scenario", "datetime", "WS", "WD", "TMP", "RH"]]
            # write
            # print("head", chosen_meteo.head())
            tmpfile = outdir / file_name(i, numsims)
            filelist += [tmpfile.name]
            chosen_meteo.to_csv(tmpfile, header=True, index=False)
        return 0, {"filelist": filelist}

    except Exception as e:
        return 1, {"filelist": filelist, "exception": e}
