#!python3
""" 👋🌎🔥
This is the package docstring
"""
__author__ = "Fernando Badilla"
__version__ = 'v0.0.1-0-gecfa54c-dirty'

import logging as _logging

_logger = _logging.getLogger(__name__)
_logging.basicConfig(level=_logging.INFO)
_logger.info("Hello world! version %s", __version__)

PACKAGE_WIDE_CONSTANT = 42
""" this is a variable docstring """
