#!python3
"""👋🌎
Some functions related to kitral weather scenario creation.
"""
__author__ = "Caro"
__revision__ = "$Format:%H$"

from pathlib import Path
import numpy as np
import pandas as pd
import random


aqui = Path(__file__).parent
# Ruta a los datos de estaciones
ruta_data = aqui / "./DB_DMC/"


def generate(x, y, start_datetime, rowres, numrows, numsims, outdir):
    """dummy generator function
    Args:
        x (float): x-coordinate of the ignition point, EPSG 4326
        y (float): y-coordinate of the ignition point, EPSG 4326
        starttime (starttime): start datetime of the weather scenario label (not implemented yet)
        rowres (int): time resolution in minutes (not implemented yet)
        numrows (int): number of hours in the weather scenario
        numsims (int): number of weather scenarios
        outdir (Path): output directory
    Return:
        retval (int): 0 if successful, 1 otherwise, 2...
        outdict (dict): output dictionary at least 'filelist': list of filenames created
    """
    try:

        if not outdir.is_dir():
            outdir.mkdir()

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
                return round(alfa + 180,2)
            elif alfa >= 180 and alfa <= 360:
                return round(alfa - 180,2)
            return np.nan

        dn = 3
        list_stn = pd.read_csv(ruta_data / "Estaciones.csv")
        list_stn["Distancia"] = list_stn.apply(distancia, args=(y, x), axis=1)  # calculo distancia
        stn = list_stn.sort_values(by=["Distancia"]).head(dn)["nombre"].tolist()

        meteos= pd.DataFrame()

        for st in stn:
            df=pd.read_csv(ruta_data /  f"{st}.csv", sep=",")
            df["station"]=st
            meteos=pd.concat([meteos, df], ignore_index=True)


        filelist = []
        for i in range(numsims):
            scenario = [scenario_name(i, numsims)] * numrows

            selected_station = meteos[ meteos['station'] == np.random.choice(stn) ]

            if len(selected_station["TMP"])<numrows:
                return 1, {"filelist": [], "exception": "Not enough data"}

            m = random.randint(0,len(selected_station["TMP"]) - numrows)

            WD=[]
            for x in selected_station["WD"].iloc[m:m+numrows]:
                WD.append((meteo_to_c2f(x)))

            dt = selected_station["datetime"].iloc[m : m + numrows]
            WS = selected_station["WS"].iloc[m : m + numrows]
            TMP = selected_station["TMP"].iloc[m : m + numrows]
            RH = selected_station["RH"].iloc[m : m + numrows].tolist()

            df = pd.DataFrame(
            np.vstack((scenario, dt, WS, WD, TMP, RH)).T,
                columns=["Scenario", "datetime", "WS", "WD", "TMP", "RH"],
            )
            tmpfile = outdir / file_name(i, numsims)
            filelist += [tmpfile.name]
            df.to_csv(tmpfile, header=True, index=False)
        return 0, {"filelist": filelist}

    except Exception as e:
        return 1, {"filelist": filelist, "exception": e}
