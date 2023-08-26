#!python3
"""👋🌎 🌲🔥
This is the raster module docstring
"""
__author__ = "Fernando Badilla"
__version__ = '7937f19-dirty'

import logging as _logging

import numpy as _np
from osgeo import gdal as _gdal, ogr as _ogr
from pathlib import Path

_logger = _logging.getLogger(__name__)
_logging.basicConfig(level=_logging.INFO)
_logger.debug("Hello world!")


def id2xy(idx: int, w: int, h: int) -> (int, int):
    """Transform a pixel or cell index, into x,y coordinates.

    Args:

    param idx: index of the pixel or cell (0,..,w*h-1)  
    param w: width of the image or grid  
    param h: height of the image or grid

    Returns:

    tuple: (x, y) coordinates of the pixel or cell  

    In GIS, the origin is at the top-left corner, read from left to right, top to bottom.  
    If your're used to matplotlib, the y-axis is inverted.  
    Also as numpy array, the index of the pixel is [y, x].
    """  # fmt: skip
    return idx % w, idx // w


def xy2id(x: int, y: int, w: int, h: int) -> int:
    """Transform a x,y coordinates into a pixel or cell index.

    Args:

    param x: width or horizontal coordinate of the pixel or cell  
    param y: height or vertical coordinate of the pixel or cell  
    param w: width of the image or grid  
    param h: height of the image or grid

    Returns:

    int: index of the pixel or cell (0,..,w\*h-1)

    In GIS, the origin is at the top-left corner, read from left to right, top to bottom.  
    If your're used to matplotlib, the y-axis is inverted.  
    Also as numpy array, the index of the pixel is [y, x].
    """  # fmt: skip
    return y * w + x


def read_raster_band(filename: str, band: int = 1) -> (_np.ndarray, int, int):
    """Read a raster file and return the data as a numpy array, along width and height.

    Args:

    param filename: name of the raster file  
    param band: band number to read (default 1)

    Returns:

    tuple: (data, width, height)

    Raises:

    FileNotFoundError: if the file is not found
    """  # fmt: skip
    dataset = _gdal.Open(filename, _gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(filename)
    return dataset.GetRasterBand(1).ReadAsArray(), dataset.RasterXSize, dataset.RasterYSize


def read_raster(filename: str) -> (_np.ndarray, dict):
    """Read a raster file and return the data as a numpy array.  
    Along raster info: transform, projection, raster count, raster width, raster height.

    Args:
    param filename: name of the raster file

    Returns:
    tuple: (data np.array, info dict)
    info dict keys: "Transform", "Projection", "RasterCount", "RasterXSize", "RasterYSize"

    Raises:

    FileNotFoundError: if the file is not found
    """  # fmt: skip
    ds = _gdal.Open(filename, _gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(filename)
    data = ds.ReadAsArray()

    info = {
        "Transform": ds.GetGeoTransform(),
        "Projection": ds.GetProjection(),
        "RasterCount": ds.RasterCount,
        "RasterXSize": ds.RasterXSize,
        "RasterYSize": ds.RasterYSize,
    }
    return data, info


def get_cell_size(raster: _gdal.Dataset | str) -> float | tuple[float, float]:
    """
    Get the cell size(s) of a raster.

    Args:
        raster (gdal.Dataset | str): The GDAL dataset or path to the raster.

    Returns:
        float | tuple[float, float]: The cell size(s) as a single float or a tuple (x, y).
    """  # fmt: skip
    if isinstance(raster, str):
        ds = _gdal.Open(raster, _gdal.GA_ReadOnly)
    elif isinstance(raster, _gdal.Dataset):
        ds = raster
    else:
        raise ValueError("Invalid input type for raster")

    # Get the affine transformation parameters
    affine = ds.GetGeoTransform()

    if affine[1] != -affine[5]:
        # If x and y cell sizes are not equal
        cell_size = (affine[1], -affine[5])  # Return as a tuple
    else:
        cell_size = affine[1]  # Return as a single float

    return cell_size


def mask_raster(raster_ds: _gdal.Dataset, band: int, polygons: list[_ogr.Geometry]) -> _np.array:
    """
    Mask a raster with polygons using GDAL.

    Args:
        raster_ds (gdal.Dataset): GDAL dataset of the raster.
        band (int): Band index of the raster.
        polygons (list[ogr.Geometry]): List of OGR geometries representing polygons for masking.

    Returns:
        np.array: Masked raster data as a NumPy array.
    """  # fmt: skip

    # Get the mask as a NumPy boolean array
    mask_array = rasterize_polygons(polygons, raster_ds.RasterXSize, raster_ds.RasterYSize)

    # Read the original raster data
    original_data = band.ReadAsArray()

    # Apply the mask
    masked_data = _np.where(mask_array, original_data, _np.nan)

    return masked_data


def rasterize_polygons(polygons: list[_ogr.Geometry], width: int, height: int) -> _np.array:
    """
    Rasterize polygons to a boolean array.

    Args:
        polygons (list[ogr.Geometry]): List of OGR geometries representing polygons for rasterization.
        geo_transform (tuple): GeoTransform parameters for the raster.
        width (int): Width of the raster.
        height (int): Height of the raster.

    Returns:
        np.array: Rasterized mask as a boolean array.
    """  # fmt: skip

    mask_array = _np.zeros((height, width), dtype=bool)

    # Create an in-memory layer to hold the polygons
    mem_driver = _ogr.GetDriverByName("Memory")
    mem_ds = mem_driver.CreateDataSource("memData")
    mem_layer = mem_ds.CreateLayer("memLayer", srs=None, geom_type=_ogr.wkbPolygon)

    for geometry in polygons:
        mem_feature = _ogr.Feature(mem_layer.GetLayerDefn())
        mem_feature.SetGeometry(geometry.Clone())
        mem_layer.CreateFeature(mem_feature)

    # Rasterize the in-memory layer and update the mask array
    _gdal.RasterizeLayer(mask_array, [1], mem_layer, burn_values=[1])

    mem_ds = None  # Release the in-memory dataset

    return mask_array


def stack_rasters(file_list: list[Path], mask_polygon: list[_ogr.Geometry] = None) -> _np.array:
    """
    Stack raster files from a list into a 3D NumPy array.

    Args:
        file_list (list[Path]): List of paths to raster files.
        mask_polygon (list[ogr.Geometry], optional): List of OGR geometries for masking. Defaults to None.

    Returns:
        np.array: Stacked raster array.
        list: List of layer names corresponding to the rasters.
    """  # fmt: skip
    array_list = []
    cell_sizes = set()
    layer_names = []

    for raster_path in file_list:
        layer_name = raster_path.stem
        layer_names.append(layer_name)

        ds = _gdal.Open(str(raster_path))
        if ds is None:
            raise ValueError(f"Failed to open raster file: {raster_path}")

        band = ds.GetRasterBand(1)

        if mask_polygon:
            flatten_array = mask_raster(ds, band, mask_polygon)
        else:
            flatten_array = band.ReadAsArray()

        array_list.append(flatten_array)
        cell_sizes.add(get_cell_size(ds))

    assert len(cell_sizes) == 1, f"There are rasters with different cell sizes: {cell_sizes}"
    stacked_array = _np.stack(array_list, axis=0)
    return stacked_array, layer_names


if __name__ == "__main__":
    file_list = Path("/home/rodrigo/code/Cluster_Generator_C2FK/RASTERS PORTEZUELO").glob("*.asc")
    array = stack_rasters(file_list)
    print(array)
