#!python3
"""👋🌎 🌲🔥
This is the raster module docstring
"""
__author__ = "Fernando Badilla"
__version__ = '784d1a4-dirty'

import logging as _logging

import numpy as _np
from osgeo import gdal as _gdal

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
    Also as numpy array, the index of the pixel is [y, x]
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
    Also as numpy array, the index of the pixel is [y, x]
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
    tuple: (data, geotransform, projection)

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
