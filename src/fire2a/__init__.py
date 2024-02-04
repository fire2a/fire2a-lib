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
__version__ = 'v0.0.1-42-g648b7fd-dirty'
__revision__ = '$Format:%H$'

import logging as logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)
logger.info("Hello world! version %s", __version__)
