#!python3
"""This is Wildfires/Forest Fire Advanced Analytics (fire2a) algorithms library 🌲🔥🧠📚 api documentation.

Get it from [PyPI](https://pypi.org/project/fire2a-lib/): `pip install fire2a-lib`

Important links:

End user documentation : https://fire2a.github.io/docs <font color="red">Use our tools with zero coding!</font>

Source code and development tips : https://github.com/fire2a/fire2a-lib Get help or contribute!

Public contact : <a href="mailto:[fire2a@fire2a.com]">e-mail</a>

Public website : www.fire2a.com

Please browse or search using the sidebar to the left!
"""
__author__ = "Fernando Badilla"
__revision__ = "$Format:%H$"
import logging
import sys
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

logger = logging.getLogger(__name__)
"""@private"""

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

    # Find or create StreamHandler
    stream_handler = None
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler):
            stream_handler = h
            break
    if stream_handler is None:
        stream_handler = logging.StreamHandler(sys.stdout)
        logger.addHandler(stream_handler)

    # Find or create RotatingFileHandler
    rf_handler = None
    if logfile:
        from logging.handlers import RotatingFileHandler

        for h in logger.handlers:
            if isinstance(h, RotatingFileHandler):
                rf_handler = h
                break
        if rf_handler is None:
            rf_handler = RotatingFileHandler(logfile, maxBytes=25 * 1024, backupCount=5)
            logger.addHandler(rf_handler)

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
    if stream_handler:
        stream_handler.setLevel(level)
    if logfile and rf_handler:
        rf_handler.setLevel(level)

    # formatter
    # "%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    if stream_handler:
        stream_handler.setFormatter(formatter)
    if logfile and rf_handler:
        rf_handler.setFormatter(formatter)

    # Add the handlers to the logger
    if stream_handler:
        logger.addHandler(stream_handler)
    if logfile and rf_handler:
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
