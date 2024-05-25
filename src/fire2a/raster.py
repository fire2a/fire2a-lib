#!python3
"""👋🌎 🌲🔥
This is the raster module docstring

List all gdal available drivers:
$ python -c "from osgeo import gdal;print('\n'.join(sorted([gdal.GetDriver(i).GetDescription() for i in range(gdal.GetDriverCount())])))"
"""
__author__ = "Fernando Badilla"
__revision__ = "$Format:%H$"

import logging
from pathlib import Path

import numpy as np
from osgeo import gdal, ogr
from qgis.core import QgsRasterLayer

from .utils import qgis2numpy_dtype

logger = logging.getLogger(__name__)


def id2xy(idx: int, w: int, h: int) -> tuple[int, int]:
    """Transform a pixel or cell index, into x,y coordinates.
    In GIS, the origin is at the top-left corner, read from left to right, top to bottom.  
    If your're used to matplotlib, the y-axis is inverted.  
    Also as numpy array, the index of the pixel is [y, x].

    Args:
        param idx: index of the pixel or cell (0,..,w*h-1)  
        param w: width of the image or grid  
        param h: height of the image or grid (not really used!)

    Returns:
        tuple: (x, y) coordinates of the pixel or cell  
    """  # fmt: skip
    return idx % w, idx // w


def xy2id(x: int, y: int, w: int) -> int:
    """Transform a x,y coordinates into a pixel or cell index.
    In GIS, the origin is at the top-left corner, read from left to right, top to bottom.  
    If your're used to matplotlib, the y-axis is inverted.  
    Also as numpy array, the index of the pixel is [y, x].

    Args:
        param x: width or horizontal coordinate of the pixel or cell  
        param y: height or vertical coordinate of the pixel or cell  
        param w: width of the image or grid  

    Returns:
        int: index of the pixel or cell (0,..,w\*h-1)
    """  # fmt: skip
    return y * w + x


def read_raster_band(filename: str, band: int = 1) -> tuple[np.ndarray, int, int]:
    """Read a raster file and return the data as a numpy array, along width and height.

    Args:
        param filename: name of the raster file  
        param band: band number to read (default 1)

    Returns:
        tuple: (data, width, height)

    Raises:
        FileNotFoundError: if the file is not found
    """  # fmt: skip
    dataset = gdal.Open(filename, gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(filename)
    return dataset.GetRasterBand(band).ReadAsArray(), dataset.RasterXSize, dataset.RasterYSize


def read_raster(filename: str, band: int = 1, data: bool = True, info: bool = True) -> tuple[np.ndarray, dict]:
    """Read a raster file and return the data as a numpy array.
    Along raster info: transform, projection, raster count, raster width, raster height.

    Args:
        param filename: name of the raster file
        param band: band number to read (default 1)
        param data: if True, return the data as a numpy array (default True)
        param info: if True, return the raster info (default True)

    Returns:
        tuple: (data, geotransform, projection)

    Raises:
        FileNotFoundError: if the file is not found
    """  # fmt: skip
    dataset = gdal.Open(filename, gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(filename)
    raster_band = dataset.GetRasterBand(band)
    data_output = raster_band.ReadAsArray() if data else None

    if info:
        rmin = raster_band.GetMinimum()
        rmax = raster_band.GetMaximum()
        if not rmin or not rmax:
            (rmin, rmax) = raster_band.ComputeRasterMinMax(True)

    info_output = (
        {
            "Transform": dataset.GetGeoTransform(),
            "Projection": dataset.GetProjection(),
            "RasterCount": dataset.RasterCount,
            "RasterXSize": dataset.RasterXSize,
            "RasterYSize": dataset.RasterYSize,
            "DataType": gdal.GetDataTypeName(raster_band.DataType),
            "NoDataValue": raster_band.GetNoDataValue(),
            "Minimum": rmin,
            "Maximum": rmax,
        }
        if info
        else None
    )
    return data_output, info_output


def get_geotransform(raster_filename: str) -> tuple[float, float, float, float, float, float]:
    """ Get geotransform from raster file.
    Args:
        raster_filename (str):

    Returns:
        tuple: geotransform
        GT[0] x-coordinate of the upper-left corner of the upper-left pixel.
        GT[1] w-e pixel resolution / pixel width.
        GT[2] row rotation (typically zero).
        GT[3] y-coordinate of the upper-left corner of the upper-left pixel.
        GT[4] column rotation (typically zero).
        GT[5] n-s pixel resolution / pixel height (negative value for a north-up image).

    reference: https://gdal.org/tutorials/geotransforms_tut.html
    """  # fmt: skip
    dataset = gdal.Open(raster_filename, gdal.GA_ReadOnly)
    if dataset is None:
        raise Exception(f"Data set is None, could not open {raster_filename}")
    return dataset.GetGeoTransform()


def transform_coords_to_georef(x_pixel: int, y_line: int, GT: tuple) -> tuple[float, float]:
    """ Transform pixel coordinates to georeferenced coordinates.
    Args:
        x_pixel (int): x pixel coordinate.
        y_line (int): y pixel coordinate.
        GT (tuple): geotransform, see get_geotransform(filename)

    Returns:
        tuple: x_geo, y_geo.

    reference: https://gdal.org/tutorials/geotransforms_tut.html
    """  # fmt: skip
    x_geo = GT[0] + x_pixel * GT[1] + y_line * GT[2]
    y_geo = GT[3] + x_pixel * GT[4] + y_line * GT[5]
    return x_geo, y_geo


def transform_georef_to_coords(x_geo: int, y_geo: int, GT: tuple) -> tuple[float, float]:
    """Inverse of transform_coords_to_georef.

    import sympy
    a, b, c, d, e, f, g, i, j, x, y = sympy.symbols('a, b, c, d, e, f, g, i, j, x, y', real=True)
    sympy.linsolve([a+i*b+j*c - x,d+i*e+j*f-y],(i,j))
    {((-a*f + c*d - c*y + f*x)/(b*f - c*e), (a*e - b*d + b*y - e*x)/(b*f - c*e))}

    Args:
        x_geo (int): x georeferenced coordinate.
        y_geo (int): y georeferenced coordinate.
        GT (tuple): geotransform, see get_geotransform(filename)

    Returns:
        tuple: x_pixel, y_line.

    TODO Raises:
        Exception: if x_pixel or y_line are not integer coordinates. by tolerance?

    reference: https://gdal.org/tutorials/geotransforms_tut.html
    """
    a, b, c, d, e, f = GT
    x, y = x_geo, y_geo
    i, j = (-a * f + c * d - c * y + f * x) / (b * f - c * e), (a * e - b * d + b * y - e * x) / (b * f - c * e)
    # if i % 1 != 0 or j % 1 != 0:
    #     raise Exception("Not integer coordinates!")
    return int(i), int(j)


def get_rlayer_info(layer: QgsRasterLayer):
    """Get raster layer info: width, height, extent, crs, cellsize_x, cellsize_y, nodata list, number of bands.

    Args:
        layer (QgsRasterLayer): A raster layer
    Returns:
        dict: raster layer info
    """
    provider = layer.dataProvider()
    ndv = []
    for band in range(1, layer.bandCount() + 1):
        ndv += [None]
        if provider.sourceHasNoDataValue(band):
            ndv[-1] = provider.sourceNoDataValue(band)
    return {
        "width": layer.width(),
        "height": layer.height(),
        "extent": layer.extent(),
        "crs": layer.crs(),
        "cellsize_x": layer.rasterUnitsPerPixelX(),
        "cellsize_y": layer.rasterUnitsPerPixelY(),
        "nodata": ndv,
        "bands": layer.bandCount(),
    }


def get_rlayer_data(layer: QgsRasterLayer):
    """Get raster layer data (EVERY BAND) as numpy array; Also returns nodata value, width and height
    The user should check the shape of the data to determine if it is a single band or multiband raster.
    len(data.shape) == 2 for single band, len(data.shape) == 3 for multiband.

    Args:
        layer (QgsRasterLayer): A raster layer

    Returns:
        data (np.array): Raster data as numpy array
        nodata (None | list): No data value
        width (int): Raster width
        height (int): Raster height

    FIXME? can a multiband raster have different nodata values and/or data types for each band?
    TODO: make a band list as input
    """
    provider = layer.dataProvider()
    if layer.bandCount() == 1:
        block = provider.block(1, layer.extent(), layer.width(), layer.height())
        nodata = None
        if block.hasNoDataValue():
            nodata = block.noDataValue()
        np_dtype = qgis2numpy_dtype(provider.dataType(1))
        data = np.frombuffer(block.data(), dtype=np_dtype).reshape(layer.height(), layer.width())
    else:
        data = []
        nodata = []
        np_dtype = []
        for i in range(layer.bandCount()):
            block = provider.block(i + 1, layer.extent(), layer.width(), layer.height())
            nodata += [None]
            if block.hasNoDataValue():
                nodata[-1] = block.noDataValue()
            np_dtype += [qgis2numpy_dtype(provider.dataType(i + 1))]
            data += [np.frombuffer(block.data(), dtype=np_dtype[-1]).reshape(layer.height(), layer.width())]
        # would different data types bug this next line?
        data = np.array(data)
    # return data, nodata, np_dtype
    return data


def get_cell_sizeV2(filename: str, band: int = 1) -> tuple[float, float]:
    _, info = read_raster(filename, band=band, data=False, info=True)
    return info["RasterXSize"], info["RasterYSize"]


def get_cell_size(raster: gdal.Dataset) -> tuple[float, float]:
    """Get the cell size(s) of a raster.
    PLANNED DEPRECATION

    Args:
        raster (gdal.Dataset | str): The GDAL dataset or path to the raster.

    Returns:
        float | tuple[float, float]: The cell size(s) as a single float or a tuple (x, y).
    """  # fmt: skip
    if isinstance(raster, str):
        ds = gdal.Open(raster, gdal.GA_ReadOnly)
    elif isinstance(raster, gdal.Dataset):
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


def mask_raster(raster_ds: gdal.Dataset, band: int, polygons: list[ogr.Geometry]) -> np.array:
    """Mask a raster with polygons using GDAL.

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
    original_data = band.ReadAsArray()  #  FIXME: wrong type hint : int has no attribute ReadAsArray

    # Apply the mask
    masked_data = np.where(mask_array, original_data, np.nan)

    return masked_data


def rasterize_polygons(polygons: list[ogr.Geometry], width: int, height: int) -> np.array:
    """Rasterize polygons to a boolean array.

    Args:
        polygons (list[ogr.Geometry]): List of OGR geometries representing polygons for rasterization.
        geo_transform (tuple): GeoTransform parameters for the raster.
        width (int): Width of the raster.
        height (int): Height of the raster.

    Returns:
        mask_array (np.array): Rasterized mask as a boolean array.
    """  # fmt: skip

    mask_array = np.zeros((height, width), dtype=bool)

    # Create an in-memory layer to hold the polygons
    mem_driver = ogr.GetDriverByName("Memory")
    mem_ds = mem_driver.CreateDataSource("memData")
    mem_layer = mem_ds.CreateLayer("memLayer", srs=None, geom_type=ogr.wkbPolygon)

    for geometry in polygons:
        mem_feature = ogr.Feature(mem_layer.GetLayerDefn())
        mem_feature.SetGeometry(geometry.Clone())
        mem_layer.CreateFeature(mem_feature)

    # Rasterize the in-memory layer and update the mask array
    gdal.RasterizeLayer(mask_array, [1], mem_layer, burn_values=[1])

    mem_ds = None  # Release the in-memory dataset

    return mask_array


def stack_rasters(file_list: list[Path], mask_polygon: list[ogr.Geometry] = None) -> tuple[np.ndarray, list[str]]:
    """Stack raster files from a list into a 3D NumPy array.

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

        ds = gdal.Open(str(raster_path))
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
    stacked_array = np.stack(array_list, axis=0)  #  type: np.array
    print(stacked_array.shape)
    return stacked_array, layer_names


if __name__ == "__main__":
    file_list = list(Path().cwd().glob("*.asc"))
    print(file_list)
    array = stack_rasters(file_list)
    print(array)
