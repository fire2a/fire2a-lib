#!python
# fmt: off
"""
Classes and methods auxiliary to using Cell2FireW and its QGIS integration

Currently:
* Writes C2FW's firebreak specification (a.csv file) from a QGIS raster layer

Processes C2FW plain text outputs 
* Fire Scars (Grids directories) into rasters and polygons
* Statistics (Intensity, HitRos, ... directories) into rasters

Auxiliary:
* get_scars_files (all together but a bit slower than the next two methods)
* get_scars_indexed (part 1/2)
* group_scars (part 2/2)

Sample Usage:
```bash
python -m fire2a.cell2fire -vvv --base-raster ../fuels.asc --authid EPSG:25831 --scar-sample Grids/Grids2/F
orestGrid01.csv --scar-raster scar_raster.tif --scar-poly scar_poly.shp --burn-prob burn_prob.tif --stat-sample RateOfSpread/ROSFile11.asc --stat-raster ros.tif --stat-summary ros_stats.tif
```
"""
# fmt: on
__author__ = "Fernando Badilla"
__revision__ = "$Format:%H$"

import logging
import sys
from pathlib import Path

from qgis.core import QgsRasterLayer

from fire2a import setup_file
from fire2a.utils import count_header_lines, fprint, loadtxt_nodata

logger = logging.getLogger(__name__)
"""captures the local logger"""

NAME, FILEPATH = setup_file(name="cell2fire", filepath=Path("/home/fdo/source/fire2a-lib/src/fire2a"))
"""setups the file name and path for logger (set path manually when pasting into ipython or jupyter session)"""


def raster_layer_to_firebreak_csv(layer: QgsRasterLayer, firebreak_val: int = 1, output_file="firebreaks.csv") -> None:
    """Write a (Cell2) Fire Simulator (C2FW) firebreak csv file from a QGIS raster layer  
    Usage as cli argument `Cell2Fire --FirebreakCells firebreaks.csv ...`

    Args:
    -   layer (QgsRasterLayer): A QGIS raster layer, default is the active layer
    -   firebreak_val (int): The value used to identify the firebreaks, default is 666
    -   output_file (str or Path): The path to the output csv file, default is firebreaks.csv

    QGIS Desktop Example: Choose method A or B, draw with serval (2)

    1.A. Use the 'Create constant raster layer' tool to create one with the same extent extent, crs and pixel size than the fuels raster layer. Recommended: 
    - constant value = 0
    - (Advanced Parameters) output raster data type = Byte

    1.B. Use the 'raster calculator' with a base layer and 0 in the formula

    2. Use 'Serval' plugin-tool to draw with the mouse the firebreaks on the new raster layer (values =1). Reload or save as to see changes.

    QGIS Python Console Example:  
    ```
    layer = iface.activeLayer()  
    from fire2a.firebreaks import raster_layer_to_firebreak_csv
    raster_layer_to_firebreak_csv(layer)
    import processing
    processing.run("fire2a:cell2firesimulator", ...
    ```
    See also: https://fire2a.github.io/docs/docs/qgis-toolbox/c2f_firebreaks.html
    """  # fmt: skip
    from numpy import array as np_array
    from numpy import where as np_where

    from fire2a.raster import get_rlayer_data, xy2id

    width = layer.width()
    data = get_rlayer_data(layer)

    # numpy is hh,ww indexing
    yy, xx = np_where(data == firebreak_val)
    ids = np_array([xy2id(x, y, width) for x, y in zip(xx, yy)])
    ids += 1

    with open(output_file, "w") as f:
        f.write("Year,Ncell\n")
        f.write(f"1,{','.join(map(str,ids))}\n")


def get_scars_files(sample_file: Path):
    """Get sorted lists of (non-empty) files matching the pattern `root/parent(+any digit)/children(+any digit).(any extension)`

    Normally used to read Cell2FireW scars `results/Grids/Grids*/ForestGrid*.csv`

    Args:
    - sample_file (Path): A sample file to extract the name and extension, parent and root directories

    Returns a tuple:
    - bool: True if successful, False otherwise.  
    - str: Error message, if any.  
    - Path: `root` - all other paths are relative to this and must be used as root/parent or root/child.  
    - list[Path]: `parents` - sorted list of parent directories.  
    - list[int]: `parents_ids` - corresponding simulation ids of these parents.  
    - list[list[Path]]: `children` - sorted list of lists of children files (grouped by simulation)
    - list[list[int]]: `children_ids` - list of lists of corresponding period ids of each simulation

    Sample Usage:
    ```python
    ret_val, msg, root, parent_dirs, parent_ids, children, children_ids = get_scars_files(Path(sample_file))
    if not ret_val:
        logger.error(msg)
    ```
    """  # fmt: skip
    from re import search as re_search

    ext = sample_file.suffix
    if match := re_search(r"(\d+)$", sample_file.stem):
        num = match.group()
    else:
        msg = f"sample_file: {sample_file} does not contain a number at the end"
        return False, msg, None, None, None, None, None
    file_name_wo_num = sample_file.stem[: -len(num)]
    parent = sample_file.absolute().parent
    root = sample_file.absolute().parent.parent
    parent = parent.relative_to(root)
    if match := re_search(r"(\d+)$", parent.name):
        num = match.group()
    else:
        msg = f"sample_file:{sample_file} parent:{parent} does not contain a number at the end"
        return False, msg, root, None, None, None, None

    parent_wo_num = parent.name[: -len(num)]
    parent_dirs = []
    parent_ids = []
    for par in root.glob(parent_wo_num + "[0-9]*"):
        if par.is_dir():
            par = par.relative_to(root)
            parent_ids += [int(re_search(r"(\d+)$", par.name).group(0))]
            parent_dirs += [par]
    adict = dict(zip(parent_dirs, parent_ids))
    parent_dirs.sort(key=lambda x: adict[x])
    parent_ids.sort()

    children = []
    children_ids = []
    for par in parent_dirs:
        chl_files = []
        chl_ids = []
        for afile in (root / par).glob(file_name_wo_num + "[0-9]*" + ext):
            if afile.is_file() and afile.stat().st_size > 0:
                afile = afile.relative_to(root)
                chl_ids += [int(re_search(r"(\d+)$", afile.stem).group(0))]
                chl_files += [afile]
        adict = dict(zip(chl_files, chl_ids))
        chl_files.sort(key=lambda x: adict[x])
        chl_ids.sort()
        children += [chl_files]
        children_ids += [chl_ids]

    # msg = f"Got {len(parent_dirs)} parent directories with {sum([len(chl) for chl in children])} children files"
    msg = ""
    return True, msg, root, parent_dirs, parent_ids, children, children_ids


def get_scars_indexed(sample_file: Path):
    """Get a sorted list of files with the pattern `root/parent(+any digit)/children(+any digit).(any extension)`

    Args:
    - sample_file (Path): A sample file to extract the extension, children name (wo ending number),  parent (wo ending number) and root directory

    Returns a tuple:
    - return_value (bool): True if successful, False otherwise
    - return_message (str): Debug/Error message if any
    - root (Path): all paths are relative to this and must be used as root/file
    - parent_wo_num (str): parent name without the ending number
    - child_wo_num (str): children name without the ending number
    - extension (str): file extension
    - files (list[Path]): sorted list of (relative paths) files
    - indexes (list[Tuple[int,int]]]): list of tuples of simulation and period ids

    Sample Usage
    ```python
    retval, retmsg, root, parent_wo_num, child_wo_num, ext, files, indexes = get_scars_indexed(sample_file)
    if not ret_val:
        logger.error(msg)
    ```
    """
    from os import sep
    from re import findall as re_findall
    from re import search as re_search

    from numpy import array as np_array
    from numpy import fromiter as np_fromiter

    ext = sample_file.suffix
    if match := re_search(r"(\d+)$", sample_file.stem):
        num = match.group()
    else:
        msg = f"sample_file: {sample_file} does not contain a number at the end"
        return False, msg, None, None, None, ext, None, None
    child_wo_num = sample_file.stem[: -len(num)]
    parent = sample_file.absolute().parent
    root = sample_file.absolute().parent.parent
    parent = parent.relative_to(root)
    if match := re_search(r"(\d+)$", parent.name):
        num = match.group()
    else:
        msg = f"sample_file:{sample_file} parent:{parent} does not contain a number at the end"
        return False, msg, root, None, child_wo_num, None, None

    parent_wo_num = parent.name[: -len(num)]

    files = np_array(
        [
            afile.relative_to(root)
            for afile in root.glob(parent_wo_num + "[0-9]*" + sep + child_wo_num + "[0-9]*" + ext)
            if afile.is_file() and afile.stat().st_size > 0
        ]
    )

    if sep == "\\":
        sep = "\\\\"
    indexes = np_fromiter(
        # re_findall(parent_wo_num + r"(\d+)" + sep + child_wo_num + r"(\d+)" + ext, " ".join(map(str, files))),
        re_findall(r"(\d+)" + sep + child_wo_num + r"(\d+)", " ".join(map(str, files))),
        dtype=[("sim", int), ("per", int)],
        count=len(files),
    )

    files = files[indexes.argsort(order=("sim", "per"))]
    indexes.sort()

    msg = ""
    # msg = f"Got {len(files)} files\n"
    # msg += f"{len(np_unique(indexes['sim']))} simulations\n"
    # msg += f"{len(np_unique(indexes['per']))} different periods"

    return True, msg, root, parent_wo_num, child_wo_num, ext, files, indexes


def group_scars(root, parent_wo_num, child_wo_num, ext, files, indexes):
    """Group scars files by simulation and period

    Args:
    - root (Path): root directory
    - parent_wo_num (str): parent name without the ending number
    - child_wo_num (str): children name without the ending number
    - ext (str): file extension
    - files (list[Path]): list of files
    - indexes (list[Tuple[int,int]]): list of tuples of simulation and period ids

    Returns:
    - parent_ids (list[int]): list of simulation ids
    - parent_dirs (list[Path]): list of parent directories
    - children_ids (list[Tuple[int,int]]): list of tuples of simulation and period ids
    - children_files (list[Path]): list of children files
    - final_scars_ids (list[Tuple[int,int]]): list of tuples of simulation and period ids
    - final_scars_files (list[Path]): list of final scars files

    Sample:
    ```python
    retval, retmsg, root, parent_wo_num, child_wo_num, ext, files, indexes = get_scars_indexed(sample_file)
    if not retval:
        logger.error(retmsg)
        sys.exit(1)
    parent_ids, parent_dirs, children_ids, children_files, final_scars_ids, final_scars_files = group_scars(
        root, parent_wo_num, child_wo_num, ext, files, indexes
    )
    ```
    """
    from numpy import unique as np_unique
    from numpy import where as np_where

    parent_ids = [sim_id for sim_id in np_unique(indexes["sim"])]
    children_ids = [[(sim_id, per_id) for per_id in indexes["per"][indexes["sim"] == sim_id]] for sim_id in parent_ids]
    children_files = [[afile for afile in files[np_where(indexes["sim"] == pid)[0]]] for pid in parent_ids]

    final_idx = [np_where(indexes["sim"] == pid)[0][-1] for pid in parent_ids]

    parent_dirs = [afile.parent for afile in files[final_idx]]

    final_scars_files = files[final_idx]
    final_scars_ids = indexes[final_idx]

    return parent_ids, parent_dirs, children_ids, children_files, final_scars_ids, final_scars_files


def build_scars(
    scar_raster: str,
    scar_poly: str,
    burn_prob: str,
    sample_file: Path,
    W: int,
    H: int,
    geotransform: tuple,
    authid: str,
    callback=None,
    feedback=None,
):
    """Builds the final scars raster, evolution scars polygons and/or burn probability raster files

    Args:
    - scar_raster (str): The output file name for the final scars raster
    - scar_poly (str): The output file name for the evolution scars polygons
    - burn_prob (str): The output file name for the burn probability raster
    - sample_file (Path): Any scar sample file usually `results/Grids/Grids1/ForestGrid1.csv`
    - W (int): Width of the raster
    - H (int): Height of the raster
    - geotransform (tuple): The geotransform of the raster
    - authid (str): The projection of the raster
    - callback (function): A function to call with the progress percentage
    - feedback (QgsFeedback): A feedback object

    Returns:
    - int: 0 if successful, 1 otherwise
    """
    from numpy import any as np_any
    from numpy import float32 as np_float32
    from numpy import int8 as np_int8
    from numpy import loadtxt as np_loadtxt
    from numpy import zeros as np_zeros
    from osgeo import gdal, ogr, osr

    from fire2a.processing_utils import get_output_raster_format, get_vector_driver_from_filename

    gdal.UseExceptions()

    retval, retmsg, root, parent_wo_num, child_wo_num, ext, files, indexes = get_scars_indexed(sample_file)
    if not retval:
        fprint(retmsg, level="error", feedback=feedback, logger=logger)
        return 1
    parent_ids, parent_dirs, children_ids, children_files, final_scars_ids, final_scars_files = group_scars(
        root, parent_wo_num, child_wo_num, ext, files, indexes
    )

    if burn_prob:
        burn_prob_arr = np_zeros((H, W), dtype=np_float32)
    else:
        burn_prob_arr = None

    if scar_raster:
        driver_name = get_output_raster_format(scar_raster, feedback=feedback)
        scar_raster_ds = gdal.GetDriverByName(driver_name).Create(
            scar_raster, W, H, len(final_scars_ids), gdal.GDT_Byte
        )
        scar_raster_ds.SetGeoTransform(geotransform)
        scar_raster_ds.SetProjection(authid)
    else:
        scar_raster_ds = None

    def final_scar_step(i, data, afile, scar_raster, scar_raster_ds, burn_prob, burn_prob_arr, feedback=None):
        if scar_raster:
            band = scar_raster_ds.GetRasterBand(i)
            # band.SetUnitType("burned")
            if 0 != band.SetNoDataValue(0):
                fprint(
                    f"Set NoData failed for Final Scar {i}: {afile}", level="warning", feedback=feedback, logger=logger
                )
            if 0 != band.WriteArray(data):
                fprint(
                    f"WriteArray failed for Final Scar {i}: {afile}", level="warning", feedback=feedback, logger=logger
                )
            if i % 100 == 0:
                scar_raster_ds.FlushCache()
        if burn_prob:
            if np_any(data == -1):
                mask = data != -1
                burn_prob_arr[ mask ] += data[ mask ]
            else:
                burn_prob_arr += data

    if scar_poly:
        # raster for each grid
        src_ds = gdal.GetDriverByName("MEM").Create("", W, H, 1, gdal.GDT_Byte)
        src_ds.SetGeoTransform(geotransform)
        src_ds.SetProjection(authid)  # export coords to file

        # datasource for shadow geometry vector layer (polygonize output)
        ogr_ds = ogr.GetDriverByName("Memory").CreateDataSource("")

        # spatial reference
        sp_ref = osr.SpatialReference()
        sp_ref.SetFromUserInput(authid)

        if scar_poly.startswith("memory:"):
            driver_name = "GPKG"
            fprint(f"Memory layer, using {driver_name} driver", level="info", feedback=feedback, logger=logger)
        else:
            driver_name = get_vector_driver_from_filename(scar_poly)
            fprint(f"NOT Mem lyr, using {driver_name} driver", level="info", feedback=feedback, logger=logger)

        drv = ogr.GetDriverByName(driver_name)
        if 0 != drv.DeleteDataSource(scar_poly):
            fprint(f"Failed to delete {scar_poly}", level="error", feedback=feedback, logger=logger)
        otrods = drv.CreateDataSource(scar_poly)
        otrolyr = otrods.CreateLayer("propagation_scars", srs=sp_ref, geom_type=ogr.wkbPolygon)
        otrolyr.CreateField(ogr.FieldDefn("simulation", ogr.OFTInteger))
        otrolyr.CreateField(ogr.FieldDefn("time", ogr.OFTInteger))
        otrolyr.CreateField(ogr.FieldDefn("area", ogr.OFTInteger))
        otrolyr.CreateField(ogr.FieldDefn("perimeter", ogr.OFTInteger))

        count_evo = 0
        count_fin = 0
        total_files = sum(len(files) for files in children_files)

        for sim_id, ids, files in zip(parent_ids, children_ids, children_files):
            count_fin += 1
            for (_, per_id), afile in zip(ids, files):
                count_evo += 1

                try:
                    data = np_loadtxt(root / afile, delimiter=",", dtype=np_int8)
                except:
                    fprint(
                        f"Error reading {afile}, retrying with nodata = 0",
                        level="error",
                        feedback=feedback,
                        logger=logger,
                    )
                    data = loadtxt_nodata(root / afile, delimiter=",", dtype=np_int8, no_data=0)

                if not np_any(data == 1):
                    fprint(
                        f"no fire in {afile}, skipping propagation polygon",
                        level="warning",
                        feedback=feedback,
                        logger=logger,
                    )
                else:
                    src_band = src_ds.GetRasterBand(1)
                    src_band.SetNoDataValue(0)
                    src_band.WriteArray(data)

                    ogr_layer = ogr_ds.CreateLayer("", srs=sp_ref)
                    gdal.Polygonize(src_band, src_band, ogr_layer, -1, ["8CONNECTED=8"])

                    feat = ogr_layer.GetNextFeature()
                    geom = feat.GetGeometryRef()
                    featureDefn = otrolyr.GetLayerDefn()
                    feature = ogr.Feature(featureDefn)
                    feature.SetGeometry(geom)
                    feature.SetField("simulation", int(sim_id))
                    feature.SetField("time", int(per_id))
                    feature.SetField("area", int(geom.GetArea()))
                    feature.SetField("perimeter", int(geom.Boundary().Length()))
                    otrolyr.CreateFeature(feature)
                    if count_evo % 100 == 0:
                        otrods.FlushCache()

                if callback:
                    callback(count_evo / len(files) * 100, f"Processed Propagation-Scar {count_evo}/{len(indexes)}")
                else:
                    fprint(
                        f"Processed Propagation-Scar {count_evo}/{len(indexes)}",
                        level="info",
                        feedback=feedback,
                        logger=logger,
                    )

            if scar_raster or burn_prob:
                final_scar_step(
                    count_fin, data, afile, scar_raster, scar_raster_ds, burn_prob, burn_prob_arr, feedback=feedback
                )
                if callback:
                    callback(None, f"Processed +Final-Scar {count_evo}/{len(indexes)}")
                else:
                    fprint(
                        f"Processed Final-Scar {count_fin}/{len(files)}", level="info", feedback=feedback, logger=logger
                    )
        # clean up
        if scar_raster:
            scar_raster_ds.FlushCache()
            scar_raster_ds = None
        otrods.FlushCache()
        otrods = None
        # otrolyr.SyncToDisk() CRASHES QGIS
        # otrolyr.FlushCache()
        otrolyr = None
        src_ds.FlushCache()
        src_ds = None
    else:
        # final scar loop
        count_fin = 0
        for (sim_id, per_id), afile in zip(final_scars_ids, final_scars_files):
            count_fin += 1
            try:
                data = np_loadtxt(root / afile, delimiter=",", dtype=np_int8)
            except:
                fprint(
                    f"Error reading {afile}, retrying with nodata = 0", level="error", feedback=feedback, logger=logger
                )
                data = loadtxt_nodata(root / afile, delimiter=",", dtype=np_int8, no_data=0)
            if not np_any(data == 1):
                fprint(f"no fire in Final-Scar {afile}", level="warning", feedback=feedback, logger=logger)
            final_scar_step(
                count_fin, data, afile, scar_raster, scar_raster_ds, burn_prob, burn_prob_arr, feedback=feedback
            )
            if callback:
                callback(count_fin / len(final_scars_files) * 100, f"Processed Final-Scar {count_fin}/{len(files)}")
            else:
                fprint(f"Processed Final-Scar {count_fin}/{len(files)}", level="info", feedback=feedback, logger=logger)
        if scar_raster:
            scar_raster_ds.FlushCache()
            scar_raster_ds = None

    if burn_prob:
        driver_name = get_output_raster_format(burn_prob, feedback=feedback)
        burn_prob_ds = gdal.GetDriverByName(driver_name).Create(burn_prob, W, H, 1, gdal.GDT_Float32)
        burn_prob_ds.SetGeoTransform(geotransform)
        burn_prob_ds.SetProjection(authid)
        band = burn_prob_ds.GetRasterBand(1)
        # band.SetUnitType("probability")
        if 0 != band.SetNoDataValue(0):
            fprint(
                f"Set NoData failed for Burn Probability {burn_prob}", level="warning", feedback=feedback, logger=logger
            )
        if 0 != band.WriteArray(burn_prob_arr / len(final_scars_files)):  # type: ignore
            fprint(
                f"WriteArray failed for Burn Probability {burn_prob}", level="warning", feedback=feedback, logger=logger
            )
        burn_prob_ds.FlushCache()
        burn_prob_ds = None

    return 0


def glob_numbered_files(sample_file: Path) -> tuple[list[Path], Path, str, str]:
    """Get a list of files with the same name (+ any digit) and extension and the directory and name of the sample file

    Args:
    - sample_file (Path): A sample file to extract the extension, name and directory
      - e.g., results/Intensity/Intensity1.asc, RateOfSpread/ROSFile2.asc, FlameLength/FL.asc, CrownFractionBurn/Cfb.asc, etc.

    Returns a tuple:
    - files (list[Path]): sorted list of files
    - adir (Path): directory of the sample file
    - aname (str): name of the sample file
    - ext (str): extension of the sample file
    """
    from re import search

    ext = sample_file.suffix
    if match := search(r"(\d+)$", sample_file.stem):
        num = match.group()
    else:
        raise ValueError(f"sample_file: {sample_file} does not contain a number at the end")
    aname = sample_file.stem[: -len(num)]
    adir = sample_file.absolute().parent
    files = []
    for afile in sorted(adir.glob(aname + "[0-9]*" + ext)):
        if afile.is_file() and afile.stat().st_size > 0:
            files += [afile]
    # QgsMessageLog.logMessage(f"files: {files}, adir: {adir}, aname: {aname}, ext: {ext}", "fire2a", Qgis.Info)
    return files, adir, aname, ext


def build_stats(
    stat_raster: str,
    stat_summary: str,
    sample_file: Path,
    W: int,
    H: int,
    geotransform: tuple,
    authid: str,
    callback=None,
    feedback=None,
):
    """Builds final statistics raster (1 band per-simulation) and summary raster (2 bands: mean against pixel burn count and stdev against total number of simulations) files
    """
    from numpy import float32 as np_float32
    from numpy import loadtxt as np_loadtxt
    from numpy import sqrt as np_sqrt
    from numpy import zeros as np_zeros
    from osgeo import gdal

    from fire2a.processing_utils import get_output_raster_format

    gdal.UseExceptions()

    files, root, aname, ext = glob_numbered_files(sample_file)
    num_headers = count_header_lines(files[0], sep=" ", feedback=feedback)

    if stat_raster:
        driver_name = get_output_raster_format(stat_raster, feedback=feedback)
        stat_raster_ds = gdal.GetDriverByName(driver_name).Create(stat_raster, W, H, len(files), gdal.GDT_Float32)
        stat_raster_ds.SetGeoTransform(geotransform)
        stat_raster_ds.SetProjection(authid)
    else:
        stat_raster_ds = None

    if stat_summary:
        summed = np_zeros((H, W), dtype=np_float32)
        sumsquared = np_zeros((H, W), dtype=np_float32)
        burncount = np_zeros((H, W), dtype=np_float32)
    else:
        summed = None
        sumsquared = None
        burncount = None

    count = 0
    for afile in files:
        count += 1
        try:
            data = np_loadtxt(root / afile, delimiter=" ", dtype=np_float32, skiprows=num_headers)
        except Exception as e:
            fprint(
                f"Error reading {afile}, retrying with nodata = -9999: {e}",
                level="error",
                feedback=feedback,
                logger=logger,
            )
            data = loadtxt_nodata(root / afile, delimiter=" ", dtype=np_float32, skiprows=num_headers, no_data=-9999)

        if stat_raster_ds:
            band = stat_raster_ds.GetRasterBand(count)
            if 0 != band.SetNoDataValue(0):
                fprint(
                    f"Set NoData failed for Statistic {count}: {afile}",
                    level="warning",
                    feedback=feedback,
                    logger=logger,
                )
            if 0 != band.WriteArray(data):
                fprint(
                    f"WriteArray failed for Statistic {count}: {afile}",
                    level="warning",
                    feedback=feedback,
                    logger=logger,
                )
            if count % 100 == 0:
                stat_raster_ds.FlushCache()

        if stat_summary:
            mask = data != -9999
            tmp = data[mask]
            summed[mask] += tmp
            sumsquared[mask] += tmp ** 2
            burncount[mask & (data != 0)] += 1

        if callback:
            callback(count / len(files) * 100, f"Processed Statistic {count}/{len(files)}")
        else:
            fprint(f"Processed Statistic {count}/{len(files)}", level="info", feedback=feedback, logger=logger)

    if stat_raster_ds:
        stat_raster_ds.FlushCache()
        stat_raster_ds = None

    if stat_summary:
        driver_name = get_output_raster_format(stat_summary, feedback=feedback)
        stat_summary_ds = gdal.GetDriverByName(driver_name).Create(stat_summary, W, H, 2, gdal.GDT_Float32)
        stat_summary_ds.SetGeoTransform(geotransform)
        stat_summary_ds.SetProjection(authid)
        # mean
        mask = burncount != 0
        mean = np_zeros((H, W), dtype=np_float32) - 9999
        mean[mask] = summed[mask] / burncount[mask]
        band = stat_summary_ds.GetRasterBand(1)
        if 0 != band.SetNoDataValue(-9999):
            fprint(
                f"Set NoData failed for Statistic {count}: {afile}", level="warning", feedback=feedback, logger=logger
            )
        if 0 != band.WriteArray(mean):
            fprint(
                f"WriteArray failed for Statistic {count}: {afile}", level="warning", feedback=feedback, logger=logger
            )
        # std
        # from all simulations
        N = len(files)
        stddev = np_sqrt(sumsquared / N - (summed/N)**2)
        # beacause this is always zero:
        # stddev = np_zeros((H, W), dtype=np_float32) - 9999
        # zero_mask = burncount == 0
        # burncount[zero_mask] = 1
        # stddev = np_sqrt(sumsquared / burncount - mean ** 2)
        # stddev[~mask] = -9999
        band = stat_summary_ds.GetRasterBand(2)
        if 0 != band.SetNoDataValue(-9999):
            fprint(
                f"Set NoData failed for Statistic {count}: {afile}", level="warning", feedback=feedback, logger=logger
            )
        if 0 != band.WriteArray(stddev):
            fprint(
                f"WriteArray failed for Statistic {count}: {afile}", level="warning", feedback=feedback, logger=logger
            )
        stat_summary_ds.FlushCache()
        stat_summary_ds = None

    return 0


def arg_parser(argv):
    """Arguments parses to coordinate: scar and stats processing, verbosity, logfile

    Args:
    - argv (list): list of strings
      - e.g., interactive: `args = arg_parser(['-vvv'])`

    Returns:
    - argparse.Namespace: parsed arguments
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=r"Fire2a-Cell2FireW Related algorithms CLI. Implemented here: Cell2FireW plain scars and statistic outputs into rasters and polygons.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    scars = parser.add_argument_group(
        "Scars: Transform C2FW scar outputs into rasters and polygons, i.e.: results/Grids/Grids*/ForestGrid*.csv"
    )
    scars.add_argument(
        "--scar-sample",
        help="Matching the pattern 'root/parent(+any digit)/children(+any digit).(any csv extension)' ",
    )
    scars.add_argument(
        "--scar-raster", default="", help="Output file name for the final scars raster, each band a simulation"
    )
    scars.add_argument(
        "--scar-poly",
        default="",
        help="Output file name for the evolution scars polygons, multiple features with simulation and period attributes",
    )
    scars.add_argument("--burn-prob", default="", help="Output file name for the burn probability raster, one band")

    stats = parser.add_argument_group(
        "Stats: Transform C2FW outputs into rasters, .e.g.: results/Intensity/Intensity1.asc, RateOfSpread/ROSFile2.asc, FlameLength/FL.asc, CrownFractionBurn/Cfb.asc, etc."
    )
    stats.add_argument(
        "--stat-sample",
        default="",
        help="Matching the pattern 'statistic_name(+any digit).asc' ",
    )
    stats.add_argument(
        "--stat-raster", default="", help="Output file name for the statistic raster, each band a simulation"
    )
    stats.add_argument(
        "--stat-summary", default="", help="Output file name for the raster, two bands: mean and std-deviation"
    )

    parser.add_argument("--verbose", "-v", action="count", default=0, help="WARNING:1, INFO:2, DEBUG:3")
    parser.add_argument(
        "--logfile",
        "-l",
        action="store_true",
        help="enable 5 log files named " + NAME + ".log (verbose must be enabled)",
        default=None,
    )
    parser.add_argument(
        "--base-raster", required=True, help="Raster to base the geotransform, width, heigth and authid"
    )
    parser.add_argument(
        "--authid", required=False, help="Auth id to override (or missing on the base raster (e.g., using .asc raster)"
    )
    args = parser.parse_args(argv)
    if args.logfile:
        args.logfile = NAME + ".log"
    return args


def main(argv=None):
    """

    args = arg_parser(['-vvv'])
    """

    if argv is sys.argv:
        argv = sys.argv[1:]
    args = arg_parser(argv)

    if args.verbose != 0:
        global logger
        from fire2a import setup_logger

        logger = setup_logger(verbosity=args.verbose, logfile=args.logfile)
        # set other modules logging level
        logging.getLogger("asyncio").setLevel(logging.INFO)

    logger.info("args %s", args)

    if not args.base_raster:
        logger.error("No base raster file name provided")
        return 1

    if not Path(args.base_raster).is_file():
        logger.error("Base raster %s is not a file", args.base_raster)
        return 1

    from fire2a.raster import read_raster

    _, raster_props = read_raster(args.base_raster, data=False, info=True)

    if not (authid := raster_props["Projection"]):
        if not (authid := args.authid):
            logger.error("No authid found on the base raster or provided")
            return 1
    logger.info("Read base raster, using authid: %s", authid)

    retval = {}
    if args.scar_sample and (args.scar_poly or args.scar_raster or args.burn_prob):
        retval["scar"] = build_scars(
            args.scar_raster,
            args.scar_poly,
            args.burn_prob,
            Path(args.scar_sample),
            raster_props["RasterXSize"],
            raster_props["RasterYSize"],
            raster_props["Transform"],
            authid,
        )
        logger.info("built scars return value: %s (0 means sucess)", retval["scar"])

    if args.stat_sample and args.stat_raster:
        retval["stat"] = build_stats(
            args.stat_raster,
            args.stat_summary,
            Path(args.stat_sample),
            raster_props["RasterXSize"],
            raster_props["RasterYSize"],
            raster_props["Transform"],
            authid,
        )
        logger.info("built statistic return value: %s (0 means success)", retval["stat"])

    print(retval)
    return sum(retval.values())


if __name__ == "__main__":
    sys.exit(main(sys.argv))
