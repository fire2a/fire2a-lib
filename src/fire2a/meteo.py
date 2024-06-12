#!python3
"""👋🌎
Some functions related to kitral weather scenario creation.
"""
__author__ = "Caro"
__revision__ = "$Format:%H$"

from datetime import timedelta

import numpy as np
from numpy.random import default_rng
from pandas import DataFrame

rng = default_rng()


def generate(x, y, start_datetime, rowres, numrows, numsims, outdir):
    """dummy generator function
    Args:
        x (float): x-coordinate of the weather station, EPSG 4326
        y (float): y-coordinate of the weather station, EPSG 4326
        starttime (starttime): start datetime of the weather scenario
        rowres (int): time resolution in minutes
        numrows (int): number of rows in the weather scenario
        numsims (int): number of weather scenarios
        outdir (Path): output directory
    Return:
        retval (int): 0 if successful, 1 otherwise, 2...
        outdict (dict): output dictionary at least 'filelist': list of filenames created
    """
    try:
        if not outdir.is_dir():
            outdir.mkdir()

        # numrows_width = len(str(numrows))
        numsims_width = len(str(numsims))

        def file_name(i, numsims):
            if numsims > 1:
                return f"Weather{i}.csv"
            return "Weather.csv"

        def scenario_name(i, numsims):
            if numsims > 1:
                return f"{outdir.name}_s{str(i).zfill(numsims_width)}"
            return outdir.name

        filelist = []
        for i in range(numsims):
            scenario = [scenario_name(i, numsims) for j in range(numrows)]
            dt = [(start_datetime + timedelta(hours=j)).isoformat(timespec="minutes") for j in range(numrows)]
            # moving average with drift
            ws = [33.33]
            wd = [33.33]
            tmp = [33.33]
            rh = [33.33]
            for j in range(1, numrows):
                ws += [ws[-1] * 0.61 + 0.39 * rng.normal(wd[-1], 4.20)]
                wd += [wd[-1] * 0.61 + 0.39 * rng.normal(wd[-1], 22.5)]
                tmp += [tmp[-1] * 0.61 + 0.39 * rng.normal(tmp[-1], 1.665)]
                rh += [rh[-1] * 0.61 + 0.39 * rng.normal(rh[-1], 1.665)]
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
        return 0, {"filelist": filelist, "test":True}
    except Exception as e:
        return 1, {"filelist": filelist, "exception": e}


if __name__ == "__main__":
    #
    # TEMPORARY TESTS
    #
    from datetime import datetime

    date = datetime.now()
    rowres = 60
    numrows = 12
    numsims = 1
    from pathlib import Path

    outdir = Path(".")
    generate(0, 0, date, rowres, numrows, numsims, outdir)
