#!python3
"""
/usr/bin/gdal_calc.py
/usr/lib/python3/dist-packages/osgeo_utils/gdal_calc.py

function osgeo_utils.gdal_calc.Calc(
    calc: Union[str, Sequence[str]],
    outfile: Union[str, os.PathLike, NoneType] = None,
    NoDataValue: Optional[numbers.Number] = None,
    type: Union[int, str, NoneType] = None,
    format: Optional[str] = None,
    creation_options: Optional[Sequence[str]] = None,
    allBands: str = '',
    overwrite: bool = False,
    hideNoData: bool = False,
    projectionCheck: bool = False,
    color_table: Union[osgeo.gdal.ColorTable,
    ForwardRef('ColorPalette'), str, os.PathLike, Sequence[str], NoneType] = None,
    extent: Optional[osgeo_utils.auxiliary.extent_util.Extent] = None,
    projwin: Union[Tuple, osgeo_utils.auxiliary.rectangle.GeoRectangle, NoneType] = None,
    user_namespace: Optional[Dict] = None,
    debug: bool = False,
    quiet: bool = False, **input_files)

from IPython.terminal.embed import InteractiveShellEmbed
InteractiveShellEmbed()()
"""
import string
import sys
from pathlib import Path

from osgeo.gdal import Dataset
from osgeo_utils.gdal_calc import Calc

from fire2a.raster import get_projwin, read_raster


def calc(
    outfile="outfile.tif",
    infiles=["infile.tif", "infile2.tif"],
    weights=None,
    NoDataValue=None,
    overwrite=True,
    type="Float32",
    format="GTiff",
    projwin=None,
    **kwargs,
):
    for i, infile in enumerate(infiles):
        if isinstance(infile, Path):
            infiles[i] = str(infile)
    if isinstance(outfile, Path):
        outfile = str(outfile)
    # if not projwin:
    #     info = locals().get("info", read_raster(infiles[0], data=False)[1])
    #     projwin, _ = get_projwin(info["Transform"], info["RasterXSize"], info["RasterYSize"])
    #     print(f"{projwin=}")

    letter_file = {}
    letter_calc = ""

    AlphaList = list(string.ascii_letters)
    if not weights:
        for alpha, afile in zip(AlphaList, infiles):
            letter_file[alpha] = afile
            letter_calc += alpha + "+"
    else:
        for alpha, afile, weight in zip(AlphaList, infiles, weights):
            letter_file[alpha] = afile
            letter_calc += str(weight) + "*" + alpha + "+"

    letter_calc = letter_calc[:-1]

    dataset = Calc(
        calc=letter_calc,
        outfile=outfile,
        NoDataValue=NoDataValue,
        overwrite=overwrite,
        type=type,
        format=format,
        projwin=projwin,
        **kwargs,
        **letter_file,
    )
    dataset.FlushCache()

    return dataset


def arg_parser(argv=None):
    """Parse command line arguments."""
    from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

    parser = ArgumentParser(
        description="Raster (weighted) summation utility, wrapping osgeo_utils.gdal_calc for sum(weights*rasters). Run `gdal_calc.py --help` for more information.",
        formatter_class=ArgumentDefaultsHelpFormatter,
        epilog="Full documentation at https://fire2a.github.io/fire2a-lib/fire2a/raster/gdal_calc_sum.html",
    )
    parser.add_argument(
        "infiles",
        nargs="+",
        type=Path,
        help="List of rasters to sum up to 52",
    )
    parser.add_argument("-o", "--outfile", help="Output file", type=Path, default="outfile.tif")
    parser.add_argument(
        "-w",
        "--weights",
        nargs="*",
        type=float,
        help="An optional list of weights to ponder the summation (else 1's)",
    )
    parser.add_argument(
        "-p",
        "--projwin",
        nargs=4,
        type=float,
        metavar=("min_x", "max_y", "max_x", "min_y"),
        help="An optional list of 4 coordinates defining the projection window, if not provided the 1st raster projwin is used",
    )
    parser.add_argument(
        "-n",
        "--NoDataValue",
        help="output nodata value (send empty for default datatype specific, see `from osgeo_utils.gdal_calc import DefaultNDVLookup`)",
        type=float,
        nargs="?",
        default=-9999,
    )
    parser.add_argument(
        "-r",
        "--return_dataset",
        help="Return dataset (for scripting -additional keyword arguments are passed to gdal_calc.Calc) instead of return code",
        action="store_true",
    )
    args = parser.parse_args(argv)
    args.projwin = tuple(args.projwin) if args.projwin else None
    if len(args.infiles) > 52:
        parser.error("Number of input rasters must be less than 53")
    if args.weights:
        if len(args.weights) != len(args.infiles):
            parser.error("Number of weights must match the number of input rasters")
    return args


def main(argv=None):
    """
    args = arg_parser(["/tmp/fuels.tif"])
    args = arg_parser(["-p","326555.21700972149847075","4562711.76232888735830784", "334617.04166780092054978", "4566058.28178922645747662","/tmp/fuels.tif"])
    args = arg_parser(["fuels.tif"])
    args = arg_parser(["-i","fuels.tif", "-m", "minmax"])
    args = arg_parser(["-i","cbh.tif", "-m", "minmax", "30"])

    _, info = read_raster("tests/assets_gdal_calc/fuels.tif", data=False)
    args = arg_parser(["tests/assets_gdal_calc/fuels.tif","tests/assets_gdal_calc/otro.tif"])
    args = arg_parser(["-w","1.0","2.0","--","tests/assets_gdal_calc/fuels.tif","tests/assets_gdal_calc/otro.tif"])
    """
    if argv is sys.argv:
        argv = sys.argv[1:]
    args = arg_parser(argv)

    print(f"{args=}")

    ds = calc(**vars(args))

    if args.return_dataset:
        return ds

    if isinstance(ds, Dataset):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
