#!python3
"""Hello World 👋🌎
This is Forest Fire Analytics 🌲🔥🧠

When developing fire2a, AVOID writing code like this: 
    
    from fire2a.module import function 

Use this instead:

    from .module import function

UNTIL WE CAN DISTRIBUTE ON PYPI!!
 
This is fire2a-lib distribution, fire2a package docstring
More info on:

    fire2a.github.io/fire2a-lib  
    fire2a.github.io/docs  
    www.fire2a.com  
"""  # fmt: skip
__author__ = "Fernando Badilla"
__revision__ = "$Format:%H$"

import logging
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

logger = logging.getLogger(__name__)

try:
    __version__ = distribution("fire2a-lib").version
    version_from = "importlib"
except PackageNotFoundError:
    if (Path(__file__).parent / "_version.py").exists():
        from ._version import __version__

        version_from = "_version.py"
    else:
        __version__ = "0.0.0"
        version_from = "fallback"

# logger.warning("%s Package version: %s, from %s", __name__, __version__, version_from)



def setup_logger(name: str = __name__, verbosity: int = 0, logfile: Path = None):
    r"""Capture the logger and setup name, verbosity, stream handler & rotating logfile if provided.
    Args:
        name (str, optional): Name of the logger. Defaults to \__name __ 
        verbosity (str | int): Verbosity level, implemented, WARNING:1, INFO:2 (default), or DEBUG:3
        logfile (Path, optional): Create a -rotated- logfile (5 files, 25MB each).
    Returns:
        logger (Logger):  All code in this pkg uses logger.info("..."), logger.debug, etc.

    ## Developers implementing their own logger
        * All fire2a modules uses `logger = logging.getLogger(__name__)`

    # Regular Usage Guideline  
    logging.critical("Something went wrong, exception info?", exc_info=True)  
    logging.error("Something went wrong, but we keep going?")  
    logging.warning("Default message level")  
    logging.info("Something planned happened")  
    logging.debug("Details of the planned thing that happened")  
    print("Normal program output, not logged")
    """  # fmt: skip
    # Capture the logger
    if name:
        # specific logger
        logger = logging.getLogger(name)
    else:
        # root logger
        logger = logging.getLogger()

    # Create a stream handler
    import sys

    stream_handler = logging.StreamHandler(sys.stdout)

    # Create a rotating file handler
    if logfile:
        from logging.handlers import RotatingFileHandler

        rf_handler = RotatingFileHandler(logfile, maxBytes=25 * 1024, backupCount=5)

    # Set the logs level
    if verbosity in ["CRITICAL", "FATAL"] or verbosity == -1:
        level = logging.CRITICAL
    elif verbosity == "ERROR" or verbosity == 0:
        level = logging.WARNING
    elif verbosity == "WARNING" or verbosity == 1:
        level = logging.WARNING
    elif verbosity == "INFO" or verbosity == 2:
        level = logging.INFO
    elif verbosity == "DEBUG" or verbosity == 3:
        level = logging.DEBUG
    else:
        level = logging.DEBUG
    logger.setLevel(level)
    stream_handler.setLevel(level)
    if logfile:
        rf_handler.setLevel(level)

    # formatter
    # "%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    stream_handler.setFormatter(formatter)
    if logfile:
        rf_handler.setFormatter(formatter)

    # Add the handlers to the logger
    logger.addHandler(stream_handler)
    if logfile:
        logger.addHandler(rf_handler)
    logger.debug("Logger initialized @level %s", logging.getLevelName(level))

    return logger


def setup_file(name="unknown", filepath=Path().cwd()):
    """Setups the NAME and FILEPATH variables for modules
    Tries getting them from __main__.__file__ and __main__.__package__
    If it's the __main__ entry point, tries to use the package name
    Args:
        main: __main__ from the calling script
        name: if fails, returns name (default "unknown")
        here: if fails, returns here Path (default "cwd")
    """
    import __main__

    if filestr := getattr(__main__, "__file__", None):
        file = Path(filestr)
        NAME = file.stem
        if NAME == "__main__":
            if package := getattr(__main__, "__package__", None):
                NAME = package + "_main"
            else:
                NAME = name
        FILEPATH = file.parent
    else:
        NAME = name
        FILEPATH = filepath
    logger.warning("setup_file(%s, %s) __main__:%s", NAME, FILEPATH, __main__)
    return NAME, FILEPATH
