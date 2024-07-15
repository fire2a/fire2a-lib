#!python3
"""👋🌎
Some functions related to kitral weather scenario creation.
"""
__author__ = "Caro"
__revision__ = "$Format:%H$"


from pathlib import Path

import numpy as np
from pandas import DataFrame, read_csv

aqui = Path(__file__).parent

# Ruta a los datos de estaciones
ruta_data = aqui / "DB_DMC"

# rng = default_rng()


def generate(x, y, start_datetime, rowres, numrows, numsims, outdir):
    # FIX NEEDED
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

    if not outdir.is_dir():
        outdir.mkdir()

    # numrows_width = len(str(numrows))
    # numsims_width = len(str(numsims))

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
            return alfa + 180
        elif alfa >= 180 and alfa <= 360:
            return alfa - 180
        return np.nan

    dn = 3
    list_stn = read_csv(ruta_data / "Estaciones.csv")
    list_stn["Distancia"] = list_stn.apply(distancia, args=(y, x), axis=1)  # calculo distancia
    stn = list_stn.sort_values(by=["Distancia"]).head(dn)["nombre"].tolist()

    meteo = []
    for i in range(len(stn)):
        station = stn[i] + ".csv"
        data = read_csv(ruta_data / station, sep=",")
        meteo.append(data)

    # MEJORA
    # meteo = pd.DataFrame()
    # for st in stn:
    #     df = pd.read_csv(ruta_data / f"{st}.csv", sep=",")
    #     df['station'] = st
    #     meteo = pd.concat([meteo, df], ignore_index=True)

    filelist = []
    for i in range(numsims):
        # scenario = [scenario_name(i, numsims) for k in range(numrows)]
        scenario = [scenario_name(i, numsims)] * numrows

        j = np.random.randint(dn - 1)
        # MEJORA
        # selected_station = meteo[ meteo['station'] == np.random.choice(stn) ]

        # FIX NEEDED!
        if len(meteo[j]["TMP"]) > numrows:
            return 1, {"filelist": [], "exception": "Not enough data"}
        m = np.random.randint(len(meteo[j]["TMP"]) - numrows)

        # MEJORA: no es necesario transformarlo a lista
        wd = []
        for x in meteo[j]["WD"].iloc[m : m + numrows].tolist():
            wd.append(meteo_to_c2f(x))

        dt = meteo[j]["datetime"].iloc[m : m + numrows].tolist()
        ws = meteo[j]["WS"].iloc[m : m + numrows].tolist()
        tmp = meteo[j]["TMP"].iloc[m : m + numrows].tolist()
        rh = meteo[j]["RH"].iloc[m : m + numrows].tolist()

        # MEJORA: esto se podria tener precalculado
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
    # poner antes de publicar
    #     return 0, {"filelist": filelist}
    # except Exception as e:
    #     return 1, {"filelist": filelist, "exception": e}
