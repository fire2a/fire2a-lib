#!python3
"""<b>Meteo Kitral</b> generates weather scenarios files for Cell2Fire-W Simulator using the Kitral fuel model standard.<br>
Using real Chilean weather station data from the Valparaíso to the Araucanía region in summer; Defining the target area (32S to 40S)<br>
<br>
Usage:<br>
- Selecting a point layer as <b>location</b> will pick the three nearest weather stations for sampling.<br>
- <b>Start hour</b>: Time of day from where to start picking station data.<br>
- <b>Temperature quantile</b>: A number greater than or equal to 0 and less than 1. It takes the daily maximum temperature values that are above the desired proportion. Example: quantile 0.75 takes the days when the daily maximum temperature is above the 75 &#37; <br>
- <b>Length of each scenario </b>: Indicates the duration, in hours, of each scenario<br>
- <b>Number_of_simulations</b>: files to generate<br>
- <b>output_directory</b>: where the files are written containing Weather(+digit).csv numbered files with each weather scenario<br>
<br>
Future Roadmap:<br>
- <b>step resolution</b>: Do other than hourly weather scenarios, to be used with the --Weather-Period-Length option (that defaults to 60)<br>
- Draw an animated vector layer representing the weather scenarios as arrows<br>
"""
__author__ = "Caro"
__revision__ = "$Format:%H$"

import sys
from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# debug
try:
    aqui = Path(__file__).parent
except:
    aqui = Path()
# Ruta a los datos de estaciones
ruta_data = aqui / "DB_DMC"
assert len(list(ruta_data.glob("*.csv"))) > 1
assert (ruta_data / "Estaciones.csv").is_file()


def file_name(i, numsims):
    if numsims > 1:
        return f"Weather{i+1}.csv"
    return "Weather.csv"


def scenario_name(i, numsims):
    if numsims > 1:
        return f"DMC_{i+1}"
    return "DMC"


ok = False
try:
    from qgis.core import QgsApplication  # type: ignore[import]
    from qgis.core import QgsDistanceArea, QgsPointXY, QgsUnitTypes  # , QgsCoordinateReferenceSystem, QgsProject

    if QgsApplication.instance():  # qgis is running

        def distancia(fila, lat, lon):
            dist = QgsDistanceArea()
            dist.setEllipsoid("WGS84")
            # dist.setSourceCrs(QgsCoordinateReferenceSystem("EPSG:4326"), QgsProject.instance().transformContext())
            assert dist.lengthUnits() == QgsUnitTypes.DistanceMeters
            p1 = QgsPointXY(fila["lon"], fila["lat"])
            p2 = QgsPointXY(lon, lat)
            d = dist.measureLine(p1, p2)
            # return d
            return dist.convertLengthMeasurement(d, QgsUnitTypes.DistanceDegrees)

        ok = True
except ModuleNotFoundError:
    ok = False

if not ok:

    def distancia(fila, lat, lon):
        if lat == fila["lat"] and lon == fila["lon"]:
            return 0
        return np.sqrt((fila["lat"] - lat) ** 2 + (fila["lon"] - lon) ** 2)


def meteo_to_c2f(alfa):
    if alfa >= 0 and alfa < 180:
        return round(alfa + 180, 2)
    elif alfa >= 180 and alfa <= 360:
        return round(alfa - 180, 2)
    return np.nan


def barlo_sota(a):
    return round((a + 180) % 360, 2)


def generate(
    lat=-36.0,
    lon=-73.2,
    start_datetime=datetime(1989, 1, 12, 12, 0, 0),
    rowres=60,
    numrows=12,
    numsims=100,
    percentile=0.5,
    outdir=Path("weather"),
    dn=3,
):
    """Carolina Lorem Ipsum Dolor Sit Amet Consectetur Adipiscing Elit
    Args:
        lat (float): latitude-coordinate of the ignition point, EPSG 4326
        lon (float): longitude-coordinate of the ignition point, EPSG 4326
        start_datetime (datetime): starting time of the weather scenario
        rowres (int): time resolution in minutes (not implemented yet)
        numrows (int): number of hours in the weather scenario
        numsims (int): number of weather scenarios
        percentile (float): daily maximum temperature quantil
        outdir (Path): output directory
        dn (int): number of closest stations to base the scenario on
    Return:
        retval (int): 0 if successful, 1 otherwise, 2...
        outdict (dict): output dictionary at least 'filelist': list of filenames created
    """

    filelist = []
    try:
        if not outdir.is_dir():
            outdir.mkdir()
            # print(f"Creating directory {outdir.absolute()}")

        # leer estaciones
        list_stn = pd.read_csv(ruta_data / "Estaciones.csv")
        # calcular distancia a input point
        list_stn["Distancia"] = list_stn.apply(distancia, args=(lat, lon), axis=1)

        #
        # Vincenty’s formula >> euclidean distance
        #
        # list_stn["d1"] = list_stn.apply(distancia, args=(lat, lon), axis=1)
        # list_stn["d2"] = list_stn.apply(distancia2, args=(lat, lon), axis=1)
        # list_stn["dd"] = list_stn["d1"] - list_stn["d2"]
        # list_stn[["dd","d1","d2"]]
        # dd = list_stn[["dd","d1","d2"]].sort_values(by=["d1"], ascending=True)
        # assert dd['d2'].is_monotonic_increasing
        #           dd        d1        d2
        # 0   0.006414  0.791903  0.785489
        # 2   0.196967  1.302510  1.105544
        # 1   0.094724  1.725775  1.631051
        # 4   0.031122  1.868370  1.837248 <- !
        # 3   0.305407  2.005146  1.699739
        # 5   0.062915  2.387851  2.324937
        # 6   0.029269  2.825473  2.796204 <- !
        # 12  0.155427  2.830431  2.675004

        # get 3 closest stations
        stn = list_stn.sort_values(by=["Distancia"]).head(dn)["nombre"].tolist()

        meteos = pd.DataFrame()
        for st in stn:
            # df = pd.read_csv(ruta_data / f"{st}.csv", sep=",", index_col=0, parse_dates=True)
            # https://stackoverflow.com/questions/29206612/difference-between-data-type-datetime64ns-and-m8ns
            df = pd.read_csv(ruta_data / f"{st}.csv", sep=",", index_col=0)
            df.index = pd.to_datetime(df.index, errors="coerce")
            df["station"] = st
            # serie temperatura diaria
            ser_tmp = df["TMP"].resample("D").max()
            qn_date = ser_tmp[ser_tmp >= ser_tmp.quantile(percentile)].index
            meteos = pd.concat([meteos, df[df.index.floor("D").isin(qn_date)].reset_index()], ignore_index=True)  # ?
        # meteos["datetime"] = pd.to_datetime(meteos["datetime"], errors="coerce") # ?
        assert meteos["datetime"].dtype == "datetime64[ns]"
        # available days by stations
        days = meteos.groupby(meteos.datetime.dt.date).first()["station"]

        for i in range(numsims):
            # draw station and day
            cd = 0
            ch = 0
            while True:
                station = np.random.choice(stn)
                # TODO mejora ocupar estaciones sorteadas, en vez de posiblemente repetir
                # station = np.random.choice(stn, size=len(stn), replace=False)
                chosen_days = days[days == station]
                if chosen_days.empty:
                    if cd > dn * 10:  # 10 veces el numero de estaciones asegura que todas fueron sorteadas
                        # print("Not enough data days", cd, ch)
                        # break
                        return 1, {"filelist": [], "exception": "No [enough] data in closest stations"}
                    cd += 1
                    continue
                day = np.random.choice(chosen_days.index)
                # retroceder un dia cada vez que falta
                start = datetime.combine(day - timedelta(days=ch), start_datetime.time())  
                chosen_meteo = meteos[(meteos["datetime"] >= start) & (meteos["station"] == station)]
                if len(chosen_meteo) < numrows:
                    if ch > len(meteos):
                        # print("Not enough data hours", cd, ch)
                        # break
                        return 1, {"filelist": [], "exception": "Not enough data"}
                    ch += 1
                    continue
                break
            # take rows
            chosen_meteo = chosen_meteo.head(numrows)
            # print(f"Selected {chosen_meteo.shape} from {station} on {day}")
            # drop station
            # TODO no drop ?
            # chosen_meteo = chosen_meteo.drop(columns=["station"])
            # wind direction
            chosen_meteo.loc[:, "WD"] = chosen_meteo["WD"].apply(barlo_sota)
            # scenario name
            chosen_meteo.loc[:, "Scenario"] = scenario_name(i, numsims)
            # TODO sobra: datetime format
            # chosen_meteo.loc[:, "datetime"] = chosen_meteo["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S")
            # TODO no aplanar al mismo dia
            # chosen_meteo.loc[:, "datetime"] = [
            #     chosen_meteo["datetime"].iloc[0] + timedelta(hours=i) for i in range(numrows)
            # ]
            # reorder
            # chosen_meteo = chosen_meteo[["Scenario", "datetime", "WS", "WD", "TMP", "RH"]]
            all_cols = chosen_meteo.columns.tolist()
            first_cols = ["Scenario", "datetime", "WS", "WD", "TMP", "RH"]
            for col in first_cols:
                all_cols.remove(col)
            chosen_meteo = chosen_meteo[first_cols + all_cols]
            # write
            tmpfile = outdir / file_name(i, numsims)
            filelist += [tmpfile.name]
            chosen_meteo.to_csv(tmpfile, header=True, index=False)
            # print(f"Writing {tmpfile.name} with {len(chosen_meteo)} rows")

        return 0, {"filelist": filelist}
    except Exception as e:
        return 1, {"filelist": filelist, "exception": e}


if __name__ == "__main__":
    return_code, return_dict = generate()
    if return_code == 0:
        filelist = return_dict["filelist"]
        print(f"Generated {len(filelist)} files: {filelist[0]}..{filelist[-1]}")
    else:
        print(f"Error generating files: {return_dict['exception']}")

    sys.exit(return_code)
