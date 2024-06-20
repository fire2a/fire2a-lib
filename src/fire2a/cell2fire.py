#!python
__author__ = "Fernando Badilla"
__revision__ = "$Format:%H$"
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
from typing import Tuple

from numpy import array, where
from qgis.core import QgsRasterLayer

from .raster import get_rlayer_data, xy2id


def raster_layer_to_firebreak_csv(
    layer: QgsRasterLayer, firebreak_val: int = 666, output_file: str | Path = "firebreaks.csv"
):
    """Write a (Cell2)Fire Simulator (C2F-W) firebreak csv file from a QGIS raster layer  
    Usage cli argument `--FirebreakCells firebreaks.csv`

    Args:
        layer (QgsRasterLayer): A QGIS raster layer, default is the active layer
        firebreak_val (int): The value used to identify the firebreaks, default is 666
        output_file (str or Path): The path to the output csv file, default is firebreaks.csv

    QGIS Desktop Example:
        A. Use the 'Create constant raster layer' tool to create a with the same extent extent, crs and pixel size than the fuels raster layer.
            Recommended: 
                constant value = 0, and (Advanced Parameters) output raster data type = Byte
        B. Use the 'raster calculator' with a base layer and 0 in the formula
        Use 'Serval' tool to draw the firebreaks on the new raster layer (values =1).
            Reload or save as to see changes

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


# def burn_probability(in_data, out_fname='bp.tif', fformat='GTiff', geotransform, epsg,  qprint = print):
#     try:
#         h,w,s = in_data.shape
#         if s==1:
#             qprint("Single band raster", level='warning')
#         burn_prob_ds = gdal.GetDriverByName(fformat).Create(out_fname, w, h, 1, GDT_Float32)
#         burn_prob_ds.SetGeoTransform(geotransform)  # specify coords
#         burn_prob_ds.SetProjection(epsg)  # export coords to file
#         band = burn_prob_ds.GetRasterBand(1)
#         band.SetUnitType("probability")
#         if 0 != band.SetNoDataValue(0):
#             qprint(f"Set No Data failed for {afile}", level='warning')
#         if 0 != band.WriteArray(burn_prob_data):
#             qprint(f"WriteArray failed for {afile}", level='warning')
#         burn_prob_ds.FlushCache()  # write to disk
#         burn_prob_ds = None
#         return 0
#     except Exception as e:
#         qprint(f"Error: {e}", level='error')
#         return 1


def get_scar_files(sample_file: Path) -> Tuple[list[Path], Path, str, str]:
    """Get a list of files with the same name (+ any digit) and extension and the directory and name of the sample file
    sample_file = Path('/home/fdo/source/C2F-W/data/Vilopriu_2013/firesim_231001_145657/results/Grids/Grids1/ForestGrid00.csv')
    sample_file = Path('/home/fdo/source/C2FSB/examples/Hom_Fuel/Grids/Grids1/ForestGrid00.csv')
    """
    from os import sep
    from re import compile as re_compile
    from re import search as re_search

    ext = sample_file.suffix
    if match := re_search("(\d+)$", sample_file.stem):
        num = match.group()
    # else:
    #     return False, f"sample_file: {sample_file} does not contain a number at the end", [], None, None, None
    file_name = sample_file.stem[: -len(num)]
    parent1 = sample_file.absolute().parent
    parent2 = sample_file.absolute().parent.parent
    if match := re_search("(\d+)$", parent1.name):
        num = match.group()
    # else:
    #     return False, f"sample_file parent: {sample_file} does not contain a number at the end", [], None, None, None
    parent1name = parent1.name[: -len(num)]
    file_gen = parent2.rglob(parent1name + "[0-9]*" + sep + file_name + "[0-9]*" + ext)
    files = []
    pattern = re_compile(parent1name + "(\d+)" + sep + file_name + "(\d+)" + ext)
    for afile in sorted(file_gen):
        sim_id, per_id = map(int, re_search(pattern, str(afile)).groups())
        if afile.is_file() and afile.stat().st_size > 0:
            files += [afile.relative_to(parent2)]
    # QgsMessageLog.logMessage(f"files: {files}, adir: {adir}, aname: {aname}, ext: {ext}", "fire2a", Qgis.Info)

    return True, "", files, parent2, file_name, ext


def get_scars_files(sample_file: Path) -> Tuple[bool, str, list[Path], list[list[Path]]]:
    """Get a sorted list of (non-empty) files with the same pattern 'root/directory(+any digit)/name(+any digit).(any extension)'.
    Args:
        sample_file (Path): A sample file to extract the root, directory, name and extension
    Returns:
        bool: True if successful, False otherwise
        str: Error message if any
        list[Path]: A list of directories
        list[list[Path]]: A SORTED list of lists of files
    """
    from re import search

    ext = sample_file.suffix
    if match := search("(\d+)$", sample_file.stem):
        num = match.group()
    else:
        return False, f"sample_file: {sample_file} does not contain a number at the end", None, None
    file_name = sample_file.stem[: -len(num)]
    parent1 = sample_file.absolute().parent
    root = sample_file.absolute().parent.parent
    if match := search("(\\d+)$", parent1.name):
        num = match.group()
    else:
        return False, f"sample_file parent: {sample_file} does not contain a number at the end", None, None

    parent1_wo_num = parent1.name[: -len(num)]
    parent1list = sorted(root.glob(parent1_wo_num + "[0-9]*"), key=lambda x: int(search("(\d+)$", x.name).group(0)))
    parent1list = [x for x in parent1list if x.is_dir()]

    files = []
    for adir in parent1list:
        dir_files = sorted(adir.glob(file_name + "[0-9]*" + ext), key=lambda x: int(search("(\d+)$", x.stem).group(0)))
        dir_files = [x for x in dir_files if x.is_file() and x.stat().st_size > 0]
        files += [dir_files]

    return True, "", parent1list, files


def calc_burn_probabilities(sample_file="Grids/Grids1/ForestGrid00.csv"):
    retval, retmsg, files, scar_dir, scar_name, ext = get_scar_files(Path(sample_file))
