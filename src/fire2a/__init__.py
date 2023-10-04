#!python3
"""👋🌎 🌲🔥
This is fire2a-lib distribution, fire2a package docstring
"""
__author__ = "Fernando Badilla"
__version__ = 'e06c899-dirty'

import logging as _logging

_logger = _logging.getLogger(__name__)
_logging.basicConfig(level=_logging.INFO)
_logger.info("Hello world! version %s", __version__)
