#!python3
"""👋🌎  
Miscellaneous utility functions that simplify common tasks.
"""
__author__ = "Fernando Badilla"
__version__ = 'v0.0.1-41-g664244e-dirty'
__revision__ = "$Format:%H$"
from numpy import loadtxt, float32
from functools import partial

def loadtxt_nodata(fname, no_data=-9999, dtype=float32, **kwargs):
    """ Load a text file into an array, casting safely to a specified data type, and replacing ValueError with a no_data value.
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
    def conv(no_data, dtype, val):
        try:
            return dtype(val)
        except ValueError:
            return no_data
    conv = partial(conv, no_data, dtype)
    return loadtxt(fname, converters=conv, dtype=dtype, **kwargs)

