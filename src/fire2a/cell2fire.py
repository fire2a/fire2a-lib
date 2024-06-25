#!python
# fmt: off
"""
Classes and methods to auxiliary to using Cell2Fire and its QGIS integration
Currently:
    - raster_layer_to_firebreak_csv
    - get_scars_files (all together but a bit slowen than the next two methods)
    - get_scars_indexed (part 1/2)
    - group_scars (part 2/2)
"""
# fmt: on
__author__ = "Fernando Badilla"
__revision__ = "$Format:%H$"

import logging
import sys
from pathlib import Path

from qgis.core import QgsRasterLayer

from .utils import fprint, loadtxt_nodata

logger = logging.getLogger(__name__)


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
        from  fire2a.firebreaks import raster_layer_to_firebreak_csv
        raster_layer_to_firebreak_csv(layer)
        import processing
        processing.run("fire2a:cell2firesimulator", ...
    ```
    See also: https://fire2a.github.io/docs/docs/qgis-toolbox/c2f_firebreaks.html
    """ # fmt: skip
    from numpy import array as np_array
    from numpy import where as np_where

    from .raster import get_rlayer_data, xy2id

    width = layer.width()
    data = get_rlayer_data(layer)

    # numpy is hh,ww indexing
    yy, xx = np_where(data == firebreak_val)
    ids = np_array([xy2id(x, y, width) for x, y in zip(xx, yy)])
    ids += 1

    with open(output_file, "w") as f:
        f.write("Year,Ncell\n")
        f.write(f"1,{','.join(map(str,ids))}\n")


def get_scars_files(sample_file):
    """Get sorted lists of (non-empty) files matching the pattern:

        root/parent(+any digit)/children(+any digit).(any extension)
    Normally used to read Cell2FireW scars 

        results/Grids/Grids*/ForestGrid*.csv

    Args:
        sample_file (Path): A sample file to extract the name and extension, parent and root directories

    Returns:

        * bool: True if successful, False otherwise.  
        * str: Error message, if any.  
        * Path: `root` - all other paths are relative to this and must be used as root/parent or root/child.  
        * list[Path]: `parents` - sorted list of parent directories.  
        * list[int]: `parents_ids` - corresponding simulation ids of these parents.  
        * list[list[Path]]: `children` - sorted list of lists of children files (grouped by simulation)\n
        * list[list[int]]: `children_ids` - list of lists of corresponding period ids of each simulation\n 

    ## Sample Usage:
    ```
        ret_val, msg, root, parent_dirs, parent_ids, children, children_ids = get_scars_files(Path(sample_file))
        if ret_val:
            final_scars = [ root / chl[-1] for chl in children ]
            ignitions_scars = [ root / chl[0] for chl in children ]
            length_of_each_simulation = [ len(chl) for chl in children ]
        else:
            print(msg)
    ```
    """ # fmt: skip
    from re import search as re_search

    ext = sample_file.suffix
    if match := re_search("(\d+)$", sample_file.stem):
        num = match.group()
    else:
        msg = f"sample_file: {sample_file} does not contain a number at the end"
        return False, msg, None, None, None, None, None
    file_name_wo_num = sample_file.stem[: -len(num)]
    parent = sample_file.absolute().parent
    root = sample_file.absolute().parent.parent
    parent = parent.relative_to(root)
    if match := re_search("(\d+)$", parent.name):
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
            parent_ids += [int(re_search("(\d+)$", par.name).group(0))]
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
                chl_ids += [int(re_search("(\d+)$", afile.stem).group(0))]
                chl_files += [afile]
        adict = dict(zip(chl_files, chl_ids))
        chl_files.sort(key=lambda x: adict[x])
        chl_ids.sort()
        children += [chl_files]
        children_ids += [chl_ids]

    # msg = f"Got {len(parent_dirs)} parent directories with {sum([len(chl) for chl in children])} children files"
    msg = ""
    return True, msg, root, parent_dirs, parent_ids, children, children_ids


def get_scars_indexed(sample_file):
    """Get a sorted list of files with the same pattern 'root/parent(+any digit)/children(+any digit).(any extension)'.
    Args:
        sample_file (Path): A sample file to extract the extension, children name (wo ending number),  parent (wo ending number) and root directory
    Returns:
        return_value (bool): True if successful, False otherwise
        return_message (str): Debug/Error message if any
        root (Path): all paths are relative to this and must be used as root/file
        parent_wo_num (str): parent name without the ending number
        child_wo_num (str): children name without the ending number
        extension (str): file extension
        files (list[Path]): sorted list of (relative paths) files
        indexes (list[Tuple[int,int]]]): list of tuples of simulation and period ids

    Sample:
        retval, retmsg, root, parent_wo_num, child_wo_num, ext, files, indexes = get_scars_indexed(sample_file)
    """
    from os import sep
    from re import findall as re_findall
    from re import search as re_search

    from numpy import array as np_array
    from numpy import fromiter as np_fromiter


    ext = sample_file.suffix
    if match := re_search("(\d+)$", sample_file.stem):
        num = match.group()
    else:
        msg = f"sample_file: {sample_file} does not contain a number at the end"
        return False, msg, None, None, None, ext, None, None
    child_wo_num = sample_file.stem[: -len(num)]
    parent = sample_file.absolute().parent
    root = sample_file.absolute().parent.parent
    parent = parent.relative_to(root)
    if match := re_search("(\d+)$", parent.name):
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

    if sep=="\\":
        sep = "\\\\"
    indexes = np_fromiter(
        # re_findall(parent_wo_num + "(\d+)" + sep + child_wo_num + "(\d+)" + ext, " ".join(map(str, files))),
        re_findall("(\d+)" + sep + child_wo_num + "(\d+)", " ".join(map(str, files))),
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
        files (list[Path]): list of files
        indexes (list[Tuple[int,int]]): list of tuples of simulation and period ids
    Returns:
        parent_ids (list[int]): list of simulation ids
        parent_dirs (list[Path]): list of parent directories
        children_ids (list[Tuple[int,int]]): list of tuples of simulation and period ids
        children_files (list[Path]): list of children files
        final_scars_ids (list[Tuple[int,int]]): list of tuples of simulation and period ids
        final_scars_files (list[Path]): list of final scars files
    Sample:
        parent_ids, parent_dirs, children_ids, children_files, final_scars_ids, final_scars_files = group_scars(files, indexes)
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
    """Build the final scars raster, evolution scars polygons and burn probability raster files
    Assummes (scar_raster or scar_poly or burn_prob) == True
    """
    from numpy import any as np_any
    from numpy import float32 as np_float32
    from numpy import int8 as np_int8
    from numpy import loadtxt as np_loadtxt
    from numpy import zeros as np_zeros
    from osgeo import gdal, ogr, osr

    from .processing_utils import (get_output_raster_format,
                                   get_vector_driver_from_filename)

    gdal.UseExceptions()

    retval, retmsg, root, parent_wo_num, child_wo_num, ext, files, indexes = get_scars_indexed(sample_file)
    if not retval:
        fprint(retmsg, level="error", feedback=feedback)
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
            band.SetUnitType("burned")
            if 0 != band.SetNoDataValue(0):
                fprint(f"Set NoData failed for Final Scar {i}: {afile}", level="warning", feedback=feedback)
            if 0 != band.WriteArray(data):
                fprint(f"WriteArray failed for Final Scar {i}: {afile}", level="warning", feedback=feedback)
            scar_raster_ds.FlushCache()
        if burn_prob:
            burn_prob_arr += data

    if scar_poly:
        # raster for each grid
        src_ds = gdal.GetDriverByName("MEM").Create("", W, H, len(files), gdal.GDT_Byte)
        src_ds.SetGeoTransform(geotransform)
        src_ds.SetProjection(authid)  # export coords to file
        # datasource for shadow geometry vector layer (polygonize output)
        ogr_ds = ogr.GetDriverByName("Memory").CreateDataSource("")
        # srs
        sp_ref = osr.SpatialReference()
        sp_ref.SetFromUserInput(authid)
        # otro
        # if scar_poly[-5:] != ".gpkg":
        #     scar_poly = scar_poly + ".gpkg"
        if scar_poly.startswith("memory:"):
            driver_name = "GPKG"
        else:
            driver_name = get_vector_driver_from_filename(scar_poly)
        otrods = ogr.GetDriverByName(driver_name).CreateDataSource(scar_poly)
        otrolyr = otrods.CreateLayer("propagation_scars", srs=sp_ref, geom_type=ogr.wkbPolygon)
        otrolyr.CreateField(ogr.FieldDefn("simulation", ogr.OFTInteger))
        otrolyr.CreateField(ogr.FieldDefn("time", ogr.OFTInteger))
        otrolyr.CreateField(ogr.FieldDefn("area", ogr.OFTInteger))
        otrolyr.CreateField(ogr.FieldDefn("perimeter", ogr.OFTInteger))
        # full loop
        count_evo = 0
        count_fin = 0
        for sim_id, ids, files in zip(parent_ids, children_ids, children_files):
            count_fin += 1
            for (_, per_id), afile in zip(ids, files):
                count_evo += 1

                # read data
                try:
                    data = np_loadtxt(root / afile, delimiter=",", dtype=np_int8)
                except:
                    fprint(f"Error reading {afile}, retrying with nodata = 0", level="error", feedback=feedback)
                    data = loadtxt_nodata(root / afile, delimiter=",", dtype=np_int8, no_data=0)
                if not np_any(data == 1):
                    fprint(f"no fire in {afile}, skipping propagation polygon", level="warning", feedback=feedback)
                else:
                    # SCARS POLY
                    # raster polygonize
                    src_band = src_ds.GetRasterBand(count_evo)
                    src_band.SetNoDataValue(0)
                    src_band.WriteArray(data)
                    ogr_layer = ogr_ds.CreateLayer("", srs=sp_ref)
                    gdal.Polygonize(src_band, src_band, ogr_layer, -1, ["8CONNECTED=8"])
                    # assumes only one feature : 8 neighbors provides that
                    feat = ogr_layer.GetNextFeature()
                    geom = feat.GetGeometryRef()
                    # create the feature and set values
                    featureDefn = otrolyr.GetLayerDefn()
                    feature = ogr.Feature(featureDefn)
                    feature.SetGeometry(geom)
                    feature.SetField("simulation", int(sim_id))
                    feature.SetField("time", int(per_id))
                    feature.SetField("area", int(geom.GetArea()))
                    feature.SetField("perimeter", int(geom.Boundary().Length()))
                    otrolyr.CreateFeature(feature)
                    otrods.FlushCache()

                if callback:
                    callback(count_evo / len(files) * 100, f"Processed Propagation-Scar {count_evo}/{len(indexes)}")
                else:
                    fprint(f"Processed Propagation-Scar {count_evo}/{len(indexes)}", level="info", feedback=feedback)

            if scar_raster or burn_prob:
                final_scar_step(
                    count_fin, data, afile, scar_raster, scar_raster_ds, burn_prob, burn_prob_arr, feedback=feedback
                )
                if callback:
                    callback(None, f"Processed +Final-Scar {count_evo}/{len(indexes)}")
                else:
                    fprint(f"Processed Final-Scar {count_fin}/{len(files)}", level="info", feedback=feedback)
        # clean up
        if scar_raster:
            scar_raster_ds.FlushCache()
            scar_raster_ds = None
        otrods.FlushCache()
        otrods = None
        # otrolyr.SyncToDisk() CRASHES QGIS
        # otrolyr = None
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
                fprint(f"Error reading {afile}, retrying with nodata = 0", level="error", feedback=feedback)
                data = loadtxt_nodata(root / afile, delimiter=",", dtype=np_int8, no_data=0)
            if not np_any(data == 1):
                fprint(f"no fire in Final-Scar {afile}", level="warning", feedback=feedback)
            final_scar_step(
                count_fin, data, afile, scar_raster, scar_raster_ds, burn_prob, burn_prob_arr, feedback=feedback
            )
            if callback and (scar_raster or burn_prob):
                callback(count_fin / len(final_scars_files) * 100, f"Processed Final-Scar {count_fin}/{len(files)}")

    if burn_prob:
        driver_name = get_output_raster_format(burn_prob, feedback=feedback)
        burn_prob_ds = gdal.GetDriverByName(driver_name).Create(burn_prob, W, H, 1, gdal.GDT_Float32)
        burn_prob_ds.SetGeoTransform(geotransform)
        burn_prob_ds.SetProjection(authid)
        band = burn_prob_ds.GetRasterBand(1)
        band.SetUnitType("probability")
        if 0 != band.SetNoDataValue(0):
            fprint(f"Set NoData failed for Burn Probability {burn_prob}", level="warning", feedback=feedback)
        if 0 != band.WriteArray(burn_prob_arr / len(final_scars_files)):
            fprint(f"WriteArray failed for Burn Probability {burn_prob}", level="warning", feedback=feedback)
        burn_prob_ds.FlushCache()
        burn_prob_ds = None

    return 0


def arg_parser(argv):
    import argparse

    parser = argparse.ArgumentParser(description="Fire2a-Cell2FireW Related algorithms CLI")
    parser.add_argument("--verbose", "-v", action="count", default=0, help="WARNING:v, INFO:vv, DEBUG:vvv")

    scars = parser.add_argument_group("Scars")
    scars.add_argument("--base-raster", required=True, help="Raster to base the geotransform, width, heigth and authid")
    scars.add_argument("--authid", required=False, help="Auth id to override (or missing on the base raster)")
    scars.add_argument(
        "--sample-file",
        required=True,
        help="Matching the pattern 'root/parent(+any digit)/children(+any digit).(any extension)',i.e., results/Grids/Grids*/ForestGrid*.csv",
    )
    scars.add_argument("--scar-raster", default="", help="The output file for the final scars raster")
    scars.add_argument("--scar-poly", default="", help="The output file for the evolution scars polygons")
    scars.add_argument("--burn-prob", default="", help="The output file for the burn probability raster")

    return parser.parse_args(argv)


def main(argv=None):
    """
    args = arg_parser(['-vvv'])
    """
    if argv is sys.argv:
        argv = sys.argv[1:]
    args = arg_parser(argv)

    if args.verbose != 0:
        from . import setup_logger

        global logger
        NAME = "fire2a"
        logger = setup_logger(NAME, verbosity=args.verbose, logfile=NAME + ".log")

    logger.info("args %s", args)

    if args.sample_file and args.base_raster:
        logger.info("building scars")
        from .raster import read_raster

        _, raster_props = read_raster(args.base_raster, data=False, info=True)
        if not (authid := raster_props["Projection"]):
            if not (authid := args.authid):
                logger.error("No authid found on the base raster or provided")
                return 1
        retval = build_scars(
            args.scar_raster,
            args.scar_poly,
            args.burn_prob,
            Path(args.sample_file),
            raster_props["RasterXSize"],
            raster_props["RasterYSize"],
            raster_props["Transform"],
            authid,
        )
        logger.info("built scars return value: %s", retval)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
