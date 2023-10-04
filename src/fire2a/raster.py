#!python3
"""👋🌎 🌲🔥
This is the raster module docstring
"""
__author__ = "Fernando Badilla"
__version__ = 'e06c899-dirty'

import logging as _logging

import numpy as _np
from osgeo import gdal as _gdal, ogr as _ogr, osr as _osr
from osgeo.gdalconst import GDT_Int16
from pathlib import Path
from tempfile import NamedTemporaryFile, mkdtemp
from statistics import mode
from typing import Dict, List
from scipy import ndimage

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


def read_raster(filename: str, data: bool = True, info: bool = True, band: int = 1) -> (_np.ndarray, dict):
    """Read a raster file and return the data as a numpy array.  
    Along raster info: transform, projection, raster count, raster width, raster height.

    Args:
    param filename: name of the raster file

    Returns:
    tuple: (data, geotransform, projection)

    Raises:
    FileNotFoundError: if the file is not found
    """  # fmt: skip
    dataset = _gdal.Open(filename, _gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(filename)
    raster_band = dataset.GetRasterBand(band)
    data_output = raster_band.ReadAsArray() if data else None

    info_output = {
        "Transform": dataset.GetGeoTransform(),
        "Projection": dataset.GetProjection(),
        "RasterCount": dataset.RasterCount,
        "RasterXSize": dataset.RasterXSize,
        "RasterYSize": dataset.RasterYSize,
        "DataType": _gdal.GetDataTypeName(raster_band.DataType),
    } if info else None
    return data_output, info_output


def mask_raster(raster_ds: _gdal.Dataset, polygons: list[_ogr.Geometry]) -> _np.array:
    """
    Mask a raster with polygons using GDAL.

    Parameters:
        raster_ds (gdal.Dataset): GDAL dataset of the raster.
        band (int): Band index of the raster.
        polygons (list[ogr.Geometry]): List of OGR geometries representing polygons for masking.

    Returns:
        np.array: Masked raster data as a NumPy array.
    """  # fmt: skip

    # Get the mask as a NumPy boolean array
    mask_array = rasterize_polygons(polygons, raster_ds.RasterXSize, raster_ds.RasterYSize)

    # Read the original raster data
    original_data = raster_ds.ReadAsArray()

    # Apply the mask
    masked_data = _np.where(mask_array, original_data, _np.nan)

    return masked_data


def rasterize_polygons(polygons: list[_ogr.Geometry], width: int, height: int) -> _np.array:
    """
    Rasterize polygons to a boolean array.

    Parameters:
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


def stack_bands_to_raster(raster_list: [], layer_names=[], output_file: str = None):
    if not output_file:
        output_file = NamedTemporaryFile(delete=False).name

    options = _gdal.BuildVRTOptions(separate=True, bandList=layer_names)
    vrt_ds = _gdal.BuildVRT(output_file + ".vrt", raster_list, options=options)

    # Create a new dataset with multiple layers
    _gdal.Translate(output_file, vrt_ds, format="GTiff")

    # Close all the source files and the VRT dataset
    for src_ds in raster_list:
        src_ds = None
    vrt_ds = None


def array2raster(array, metadata, band_name: str = None, output_filename: str = None) -> _gdal.Dataset:
    # array = array[::-1]  # reverse array so the tif looks like the array
    cols = array.shape[1]
    rows = array.shape[0]
    geotransform = metadata["geotransform"]
    crs = metadata["crs"]
    dtype = _gdal.GDT_Float32
    crs = crs if crs else 4326
    band_name = band_name if band_name else "cluster"
    nbands = 1

    if output_filename is None:
        # If output_filename is not provided, create a temporary file
        output_filename = NamedTemporaryFile(suffix=".tif", delete=False).name

    driver = _gdal.GetDriverByName("GTiff")
    outRaster = driver.Create(output_filename, cols, rows, nbands, dtype)
    outRaster.SetGeoTransform(*geotransform)
    outband = outRaster.GetRasterBand(1)
    outband.SetDescription(band_name)
    outband.WriteArray(array)
    outRasterSRS = _osr.SpatialReference()
    outRasterSRS.ImportFromEPSG(crs)
    outRaster.SetProjection(outRasterSRS.ExportToWkt())
    outband.SetNoDataValue(-9999.9)
    outband.FlushCache()
    outRaster = None
    return _gdal.Open(output_filename)


def stack_rasters_to_ndarray(file_list: list[str], mask_polygon: list[_ogr.Geometry] = None) -> _np.array:
    """
    Stack raster files from a list into a 3D NumPy array.

    Parameters:
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

        band, metadata = read_raster(raster_path)

        if mask_polygon:
            flatten_array = mask_raster(band, mask_polygon)
        else:
            flatten_array = band.ReadAsArray()

        array_list.append(flatten_array)
        raster_cell_size = (metadata["RasterXSize"],metadata["RasterYSize"])
        cell_sizes.add(raster_cell_size)

    assert len(cell_sizes) == 1, f"There are rasters with different cell sizes: {cell_sizes}"
    stacked_array = _np.stack(array_list, axis=0)
    return stacked_array, layer_names


def polygonize(raster, output_filename) -> _ogr.DataSource:
    # Read the input raster's band (assuming a single-band raster)
    band = raster.GetRasterBand(1)
    try:
        crs = band.GetProjection()
    except:
        crs = "EPSG:4326"

    dst_layer_name = "cluster"
    driver = _ogr.GetDriverByName("ESRI Shapefile")
    output_ds = driver.CreateDataSource(output_filename)

    spacial_reference = _osr.SpatialReference()
    spacial_reference.SetFromUserInput(crs)

    shape_layer = output_ds.CreateLayer(dst_layer_name, srs=spacial_reference)
    field = _ogr.FieldDefn("Cluster", _ogr.OFTInteger)
    shape_layer.CreateField(field)
    dst_field = shape_layer.GetLayerDefn().GetFieldIndex("Cluster")

    _gdal.Polygonize(band, None, shape_layer, dst_field, [], callback=None)

    return driver.Open(output_filename, 1)

    # # Create a new vector layer to store the polygons
    # output_layer = output_ds.CreateLayer("cluster", srs=None)

    # # Use gdal.Polygonize to convert the raster to polygons
    # # fd = ogr.FieldDefn("Cluster", ogr.OFTInteger)
    # # dst_layer.CreateField(fd)
    # # dst_field = dst_layer.GetLayerDefn().GetFieldIndex("Cluster")
    # _gdal.Polygonize(band, None, output_layer, -1, [], callback=None)

    # # Close the output vector dataset
    # output_ds = None

    # # Close the input raster dataset
    # input_raster = None
    # return output_filename


def plot_array(array: _np.array) -> None:
    import matplotlib.pyplot as plt

    plt.imshow(array, cmap="viridis")  # You can change the colormap as needed
    plt.colorbar()  # Add a colorbar to the plot
    plt.title("2D NumPy Array")
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.show()


def add_features(shape: _ogr.DataSource, features_dct: Dict[str, List[float or int]], output_path) -> None:
    layer = shape.GetLayer()
    for feature in features_dct.keys():
        feature_field = _ogr.FieldDefn(feature, _ogr.OFSTFloat32)
        layer.CreateField(feature_field)

    for feature in layer:
        print(feature)


if __name__ == "__main__":
    file_list = Path("/home/rodrigo/code/Cluster_Generator_C2FK/RASTERS PORTEZUELO").glob("*.asc")
    file_list = list(file_list)
    sample_file = file_list[0].__str__()
    _, metadata = read_raster(sample_file, data = False)
    array = stack_rasters_to_ndarray(file_list)
    # (array)
    # Example usage:
    output_raster = "output_shapefile.tif"
    array = array[0][0]
    rst = array2raster(array, metadata)

    shape = polygonize(rst, "output_polygons.shp")

    # driver = _ogr.GetDriverByName("ESRI Shapefile")
    # dataSource = driver.Open(shapefile_path, 1)  # 1 para abrir en modo de escritura
    # if dataSource is None:
    #     print("No se pudo abrir el archivo shapefile.")
    #     exit()
    # layer = dataSource.GetLayer()

    # for feature in layer:
    #     key = feature.GetField("FID")  # Reemplaza "tu_llave" con el nombre real del campo de la llave en tu shapefile
    # if key in dct:
    #     valor = dct[key]
    #     feature.SetField(
    #         "nuevo_campo", valor
    #     )  # Reemplaza "nuevo_campo" con el nombre del campo donde deseas agregar el valor
    #     layer.SetFeature(feature)
