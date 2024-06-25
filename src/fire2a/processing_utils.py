#!python3
"""
This module is related to qgis processing algorithms, needing a special initialization
"""
__author__ = "Fernando Badilla"
__revision__ = "$Format:%H$"
import logging
import os
import sys

from qgis.core import QgsProcessingFeedback

from .utils import fprint

logger = logging.getLogger(__name__)

from platform import system as platform_system

# Append the path where processing plugin can be found
# TODO macos
if platform_system() == "Windows":
    sys.path.append("C:\\PROGRA~1\\QGIS33~1.1\\apps\\qgis\\python\\plugins")
else:
    sys.path.append("/usr/share/qgis/python/plugins")

# import processing
# from processing.core.Processing import Processing
# Processing.initialize()
from processing.algs.gdal.GdalUtils import GdalUtils


def get_vector_driver_from_filename(filename: str) -> str:
    return GdalUtils.getVectorDriverFromFileName(filename)


def get_output_raster_format(filename: str, feedback: QgsProcessingFeedback = None) -> str:
    """Gets a valid GDAL output raster driver name, warns if not found, defaults to GTiff.

    Args:
        filename (str): The name with extension of the raster. (Not implemented for suffixes with multiple dots, e.g. mpv.gz)
        feedback (QgsProcessingFeedback): The feedback object to push warnings to.

    Returns:
        str: The GDAL short format name for extension.

    Sample usage:
        driver_name = get_output_raster_format(filename, feedback)
        dst_ds = gdal.GetDriverByName(raster_format).Create(filename, W, H, 1, GDT_Float32)

    Based/copied from qgis.python.grassprovider.grass_utils.py GrassUtils
    """
    ext = os.path.splitext(filename)[1].lower()
    ext = ext.lstrip(".")
    if ext:
        supported = GdalUtils.getSupportedOutputRasters()
        for name in list(supported.keys()):
            exts = supported[name]
            if ext in exts:
                return name
    fprint(
        f"Using GTiff format! No supported GDAL raster format for {filename=} {ext=} found.",
        level="warning",
        feedback=feedback,
    )
    return "GTiff"


def check_gdal_readable_raster(filename):
    """Based/copied from qgis.python.grassprovider.grass_utils.py GrassUtils"""
    ext = os.path.splitext(filename)[1].lower()
    ext = ext.lstrip(".")
    if ext:
        supported = GdalUtils.getSupportedRasters()
        for name in list(supported.keys()):
            exts = supported[name]
            if ext in exts:
                return True
    return False
