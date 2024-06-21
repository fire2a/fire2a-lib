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

from pathlib import Path

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
    See also: https://fire2a.github.io/docs/docs/qgis-toolbox/c2f_firebreaks.html
    """ # fmt: skip
    from numpy import array as np_array
    from numpy import where as np_where

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
    """Get sorted lists of (non-empty) files matching the pattern 'root/parent(+any digit)/children(+any digit).(any extension)'. Normally used to read Cell2FireW scars results/Grids/Grids*/ForestGrid*.csv
    Args:
        sample_file (Path): A sample file to extract the name and extension, parent and root directories
    Returns:
        bool: True if successful, False otherwise
        str: Error message if any
        root (Path): all other paths are relative to this and must be used as root/parent or root/child
        parents (list[Path]): sorted list of parent directories
        parents_ids (list[int]): corresponding simulation ids of these parents
        children (list[list[Path]]): sorted list of lists of children files
        children_ids (list[list[int]]): list of lists of corresponding period ids of each simulation
    Sample Usage:
        ret_val, msg, root, parent_dirs, parent_ids, children, children_ids = get_scars_files(sample_file)
        if ret_val:
            final_scars = [ root / chl[-1] for chl in children ]
            ignitions_scars = [ root / chl[0] for chl in children ]
            length_of_each_simulation = [ len(chl) for chl in children ]
        else:
            print(msg)
    """
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
    from numpy import array as np_array
    from numpy import in1d as np_in1d
    from numpy import unique as np_unique
    from numpy import where as np_where

    parent_ids = [sim_id for sim_id in np_unique(indexes["sim"])]
    children_ids = [[(sim_id, per_id) for per_id in indexes["per"][indexes["sim"] == sim_id]] for sim_id in parent_ids]

    parent_dirs = [root / (parent_wo_num + str(sim_id)) for sim_id in parent_ids]
    children_files = [
        [
            root / (parent_wo_num + str(sim_id)) / (child_wo_num + str(per_id) + ext)
            for per_id in indexes["per"][indexes["sim"] == sim_id]
        ]
        for sim_id in parent_ids
    ]
    final_scars_ids = np_array(
        # is sorted [(sim_id, indexes["per"][indexes["sim"] == sim_id].max()) for sim_id in parent_ids],
        [(sim_id, indexes["per"][indexes["sim"] == sim_id][-1]) for sim_id in parent_ids],
        dtype=[("sim", int), ("per", int)],
    )
    final_scars_indices = np_where(np_in1d(indexes.view(dtype="void"), final_scars_ids.view(dtype="void")))[0]
    final_scars_files = files[final_scars_indices]

    return parent_ids, parent_dirs, children_ids, children_files, final_scars_ids, final_scars_files


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
