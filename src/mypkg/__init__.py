#!python3
"""
This is the package docstring
"""
__author__ = 'Fernando Badilla'
__version__ = '40ee0ea-dirty'

import logging as _logging
_logger = _logging.getLogger(__name__)
_logging.basicConfig(level=_logging.INFO)
_logger.debug('Hello world!')

PACKAGE_WIDE_CONSTANT = 42
""" this variable is available to all modules in this package """
