#!python3
"""👋🌎
Miscellaneous utility functions that simplify common tasks.
"""
__author__ = "Fernando Badilla"
__revision__ = "$Format:%H$"
import logging

import numpy as np
from qgis.core import Qgis, QgsProcessingFeedback

logger = logging.getLogger(__name__)


def loadtxt_nodata(fname, no_data=-9999, dtype=np.float32, **kwargs) -> np.ndarray:
    """Load a text file into an array, casting safely to a specified data type, and replacing ValueError with a no_data value.
    Other arguments are passed to numpy.loadtxt. (delimiter=',' for example)

    Args:
        fname : file, str, pathlib.Path, list of str, generator
            File, filename, list, or generator to read.  If the filename
            extension is ``.gz`` or ``.bz2``, the file is first decompressed. Note
            that generators must return bytes or strings. The strings
            in a list or produced by a generator are treated as lines.
        dtype : data-type, optional
            Data-type of the resulting array; default: float32.  If this is a
            structured data-type, the resulting array will be 1-dimensional, and
            each row will be interpreted as an element of the array.  In this
            case, the number of columns used must match the number of fields in
            the data-type.
        no_data (numeric, optional): No data value. Defaults to -9999.
        **kwargs: Other arguments are passed to numpy.loadtxt. (delimiter=',' for example)

    Returns:
        out : numpy.ndarray: Data read from the text file.

    See Also:
        numpy: loadtxt, load, fromstring, fromregex
    """
    from functools import partial

    def conv(no_data, dtype, val):
        try:
            return dtype(val)
        except ValueError:
            return no_data

    conv = partial(conv, no_data, dtype)
    return np.loadtxt(fname, converters=conv, dtype=dtype, **kwargs)


def qgis2numpy_dtype(qgis_dtype: Qgis.DataType) -> np.dtype:
    """Conver QGIS data type to corresponding numpy data type
    https://raw.githubusercontent.com/PUTvision/qgis-plugin-deepness/fbc99f02f7f065b2f6157da485bef589f611ea60/src/deepness/processing/processing_utils.py
    This is modified and extended copy of GDALDataType.

    * ``UnknownDataType``: Unknown or unspecified type
    * ``Byte``: Eight bit unsigned integer (quint8)
    * ``Int8``: Eight bit signed integer (qint8) (added in QGIS 3.30)
    * ``UInt16``: Sixteen bit unsigned integer (quint16)
    * ``Int16``: Sixteen bit signed integer (qint16)
    * ``UInt32``: Thirty two bit unsigned integer (quint32)
    * ``Int32``: Thirty two bit signed integer (qint32)
    * ``Float32``: Thirty two bit floating point (float)
    * ``Float64``: Sixty four bit floating point (double)
    * ``CInt16``: Complex Int16
    * ``CInt32``: Complex Int32
    * ``CFloat32``: Complex Float32
    * ``CFloat64``: Complex Float64
    * ``ARGB32``: Color, alpha, red, green, blue, 4 bytes the same as QImage.Format_ARGB32
    * ``ARGB32_Premultiplied``: Color, alpha, red, green, blue, 4 bytes  the same as QImage.Format_ARGB32_Premultiplied
    """
    if qgis_dtype == Qgis.DataType.Byte or qgis_dtype == "Byte":
        return np.uint8
    if qgis_dtype == Qgis.DataType.UInt16 or qgis_dtype == "UInt16":
        return np.uint16
    if qgis_dtype == Qgis.DataType.Int16 or qgis_dtype == "Int16":
        return np.int16
    if qgis_dtype == Qgis.DataType.Float32 or qgis_dtype == "Float32":
        return np.float32
    if qgis_dtype == Qgis.DataType.Float64 or qgis_dtype == "Float64":
        return np.float64


def getGDALdrivers():
    from osgeo import gdal  # isort: skip # fmt: skip
    ret = []
    for i in range(gdal.GetDriverCount()):
        drv = {"ShortName": gdal.GetDriver(i).GetDescription()}
        meta = gdal.GetDriver(i).GetMetadata()
        assert "ShortName" not in meta
        drv.update(meta)
        ret += [drv]
    return ret


def getOGRdrivers():
    from osgeo import ogr  # isort: skip # fmt: skip
    ret = []
    for i in range(ogr.GetDriverCount()):
        drv = {"ShortName": ogr.GetDriver(i).GetDescription()}
        meta = ogr.GetDriver(i).GetMetadata()
        assert "ShortName" not in meta
        drv.update(meta)
        ret += [drv]
    return ret


def fprint(*args, sep=" ", end="", level="warning", feedback: QgsProcessingFeedback = None, **kwargs) -> None:
    """replacement for print into logger and QgsProcessingFeedback
    Args:
        *args: positional arguments
        sep (str, optional): separator between args. Defaults to " ".
        end (str, optional): end of line. Defaults to "".
        level (str, optional): logging level: debug, info, warning(default), error.
        feedback (QgsProcessingFeedback, optional): QgsProcessingFeedback object. Defaults to None.
        **kwargs: keyword arguments
    """
    msg = sep.join(map(str, args)) + sep
    msg += sep.join([f"{k}={v}" for k, v in kwargs.items()]) + end
    if level == "debug":
        if feedback:
            feedback.pushDebugInfo(msg)
        else:
            logger.debug(msg)
    elif level == "info":
        if feedback:
            feedback.pushInfo(msg)
        else:
            logger.info(msg)
    elif level == "warning":
        if feedback:
            feedback.pushWarning(msg)
        else:
            logger.warning(msg)
    elif level == "error":
        if feedback:
            feedback.reportError(msg)
        else:
            logger.error(msg)
