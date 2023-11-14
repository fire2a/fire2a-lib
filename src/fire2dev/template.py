#!python3
"""👋🌎 This docstring is called a module docstring. Describes the general purpose of the module.

In this case being a sample module to serve as a pattern for creating new modules.

It has docstrings for:  
- module  
- global variable  
- method  
- class  

Implements:  
- skipping black formating of docstrings using # fmt: skip/on/off  
- logging  
- module cli using main & argparse:  
```
$ python -m fdolib.template --help
```
ipython  
%autoindent  
from fire2a.template import a_method
a_method((1, 'a'), 'b', 'c', an_optional_argument=2, d='e', f='g')
👋🌎
""" # fmt: skip
__author__ = "Fernando Badilla"
__version__ = 'v0.0.1-38-gd1276ea-dirty'
__revision__ = '$Format:%H$'

MODULE_VARIABLE = 'very important and global variable'
""" this docstring describes a global variable   
has 0 indent """ # fmt: skip

import logging as _logging
import sys

import numpy as _np
from osgeo import gdal as _gdal, ogr as _ogr
from pathlib import Path

def a_method( a_required_argument : tuple[int, str], *args, an_optional_argument : int = 0, **kwargs ) -> dict:
    """ this is a method docstring that describes a method """ # fmt: skip
    a, b = a_required_argument
    
    for arg in args:
        _logger.debug('log *args %s', arg)

    for key, value in kwargs.items():
        _logger.debug('log **kwargs key:%s value:%s', key, value)

    _logger.info('info log %s', MODULE_VARIABLE[::-1])
    """ this is not a docstring, just a comment """

    return {'a': a, 'b': b}

def main(argv):
    """ this is a function docstring that describes a function """
    _logger = _logging.getLogger('fire2a:templatemodule')
    _logger.info("Hello world!")
    _logger.info(f"argv:{argv}")
    returns = a_method((1, 'a'), 'b', 'c', an_optional_argument=2, d='e', f='g')
    _logging.debug('a_method returns %s', returns)

if __name__ == '__main__':
    _logger = _logging.getLogger('fire2a:templatemodule')
    sys.exit(main(sys.argv))
else:
    _logger = _logging.getLogger(__name__)
    _logging.basicConfig(level=_logging.INFO)
    _logger.debug("Hello world!")

