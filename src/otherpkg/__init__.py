#!python3
"""
This is the OTHER PKG package docstring
"""
__author__ = "Fernando Badilla"
__version__ = 'v0.0.1-0-gecfa54c-dirty'

import logging as _logging

_logger = _logging.getLogger(__name__)
_logging.basicConfig(level=_logging.INFO)
_logger.debug("Hello world! from OTHER PKG")

PACKAGE_WIDE_CONSTANT = 66
""" this variable is available to all modules in this other package """
