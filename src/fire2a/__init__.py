#!python3
"""👋🌎 🌲🔥
Hello World Forest Fire!
This is fire2a-lib distribution, fire2a package docstring
More info on:

    fire2a.github.io/fire2a-lib  
    fire2a.github.io/docs  
    www.fire2a.com  
""" # fmt: skip
__author__ = "Fernando Badilla"
__version__ = 'v0.0.1-38-gd1276ea-dirty'
__revision__ = '$Format:%H$'

import logging as _logging

_logger = _logging.getLogger(__name__)
_logging.basicConfig(level=_logging.DEBUG)
_logger.info("Hello world! version %s", __version__)
