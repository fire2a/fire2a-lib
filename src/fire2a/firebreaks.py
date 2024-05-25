#!python
# fmt: off
"""
To graphically define firebreaks, use QGIS to draw firebreaks on a raster layer then export it using this module.

Cell2Fire-W simulator features placing firebreaks (that internally are represented as unburnable cells); making it easier than to manually create a new fuels raster for every firebreak experiment/simulation.

This is done by appending a `.csv` file to the simulation command. For example: `--FirebreakCells fbv.csv`, containing:

        Year,Ncell
        1,1,81,161,162

Values represents an "L" shaped firebreak: topleft cell is 1, the one below 81, below 161, to the right 162. (in a 80 pixels width raster)

Headers are: Year, Ncell.  
The first column always is Year, 1 (as CFW-W only supports one fire before resetting the landscape).  
Unburnable values are placed as a row of cells id (starting from 1)

Requires:  
 - QGIS  
 - Serval QGIS plugin, check the manual here: https://github.com/lutraconsulting/serval/blob/master/Serval/docs/user_manual.md  
 - A raster layer loaded in QGIS  

Steps:  
1. User Serval to draw firebreaks on the raster layer, you can create a new layer for the firebreaks or set a peculiar value on the raster  
2. Run this script in the QGIS Python console, identifying the layer and the value used for the firebreaks  
3. The script will create a csv file with the firebreaks  
4. Run C2F-W with the firebreaks csv file:  

    $ cd /path/to/instance
    $ rm -f Data.csv
    $ rm -fr result_wFirebreaks
    $ mkdir result_wFirebreaks
    $ /path/to/Cell2Fire.Linux.x86_64 --final-grid --sim K --input-instance-folder . --output-folder result_wFirebreaks --FirebreakCells fbv.csv | tee result_wFirebreaks/LogFile.txt

Or append the fullpath to the `fbv.csv` file to the Advanced Parameters in the GUI as: `--FirebreakCells /full/path/fbv.csv`
"""
# fmt: on

from pathlib import Path

from numpy import array, where
from qgis.core import QgsRasterLayer

from fire2a.raster import get_rlayer_data, xy2id


def raster_layer_to_firebreak_csv(
    layer: QgsRasterLayer, firebreak_val: int = 666, output_file: str | Path = "firebreaks.csv"
):
    """Write a (Cell2)Fire Simulator (C2F-W) firebreak csv file from a QGIS raster layer  
    Usage cli argument `--FirebreakCells firebreaks.csv`

    Args:
        layer (QgsRasterLayer): A QGIS raster layer, default is the active layer
        firebreak_val (int): The value used to identify the firebreaks, default is 666
        output_file (str or Path): The path to the output csv file, default is firebreaks.csv

    QGIS Console Example:  
    ```
        layer = iface.activeLayer()  
        from fire2a.firebreaks import raster_layer_to_firebreak_csv
        raster_layer_to_firebreak_csv(layer)
        import processing
        processing.run("fire2a:cell2firesimulator", ...
    ```
    """ # fmt: skip
    width = layer.width()
    data = get_rlayer_data(layer)

    # numpy is hh,ww indexing
    yy, xx = where(data == firebreak_val)
    ids = array([xy2id(x, y, width) for x, y in zip(xx, yy)])
    ids += 1

    with open(output_file, "w") as f:
        f.write("Year,Ncell\n")
        f.write(f"1,{','.join(map(str,ids))}\n")
