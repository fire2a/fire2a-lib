#!python
"""
A Cell2Fire-W simulator firebreak.csv file is a one row csv with values identifying the unburnable cells.
Cells are identified by their cell number, starting from 1. See fire2a.raster.xy2id and add 1
Headers are: Year, Ncell

Example: The topleft cell and the one below is a firebreak (in a 80 pixels width raster)
        Year,Ncell
        1,1,81

CFW-W only supports single year, so Year,1 is always the first column.

Use QGIS to draw firebreaks on a raster layer and export the firebreaks to a csv file for CF2-W
Requires:
 - QGIS
 - Serval QGIS plugin, check the manual here: https://github.com/lutraconsulting/serval/blob/master/Serval/docs/user_manual.md
 - A raster layer loaded in QGIS

How To:
1. User Serval to draw firebreaks on the raster layer, you can create a new layer for the firebreaks or set a peculiar value on the raster
2. Run this script in the QGIS Python console, identifying the layer and the value used for the firebreaks
3. The script will create a csv file with the firebreaks
4. Run C2F-W with the firebreaks csv file:

    $ cd /path/to/instance
    $ rm -f Data.csv
    $ rm -fr result_wFirebreaks
    $ mkdir result_wFirebreaks
    $ /path/to/Cell2Fire.Linux.x86_64 --final-grid --sim K --input-instance-folder . --output-folder result_wFirebreaks --FirebreakCells firebreak.csv | tee result_wFirebreaks/LogFile.txt

Or append the fullpath to the firebreaks.csv file to the Advanced Parameters in the GUI as: --FirebreakCells /full/path/firebreaks.csv
"""

from numpy import array, where
from qgis.utils import iface

from fire2a.raster import get_rlayer_data, xy2id


def write_firebreak_csv_from_qgis_raster(layer=iface.activeLayer(), firebreak_val=666, output_file="firebreaks.csv"):
    """Write a (Cell2)Fire Simulator (C2F-W) firebreak csv file from a QGIS raster layer
    Args:
        layer (QgsRasterLayer): A QGIS raster layer, default is the active layer
        firebreak_val (int): The value used to identify the firebreaks, default is 666
        output_file (str | Path): The path to the output csv file, default is firebreaks.csv
    Returns:
        None
    Raises:
        None
    """
    width = layer.width()
    data = get_rlayer_data(layer)

    # numpy is hh,ww indexing
    yy, xx = where(data == firebreak_val)
    ids = array([xy2id(x, y, width) for x, y in zip(xx, yy)])
    ids += 1

    with open(output_file, "w") as f:
        f.write("Year,Ncell\n")
        f.write(f"1,{','.join(map(str,ids))}\n")
