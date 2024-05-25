#!python3
"""Hello World 👋🌎
This is Forest Fire Analytics 🌲🔥🧠
 
This is fire2a-lib distribution, fire2a package docstring
More info on:

    fire2a.github.io/fire2a-lib  
    fire2a.github.io/docs  
    www.fire2a.com  
"""  # fmt: skip
__author__ = "Fernando Badilla"
__revision__ = "$Format:%H$"

import logging
from pathlib import Path

from importlib_metadata import PackageNotFoundError, distribution

logger = logging.getLogger(__name__)

try:
    __version__ = distribution("fire2a").version
    version_from = "importlib_metadata"
except PackageNotFoundError:
    if (Path(__file__).parent / "_version.py").exists():
        from ._version import __version__

        version_from = "_version.py"
    else:
        __version__ = "0.0.0"
        version_from = "fallback"

logger.warning("%s Package version: %s, from %s", __name__, __version__, version_from)


def setup_logger(name: str = __name__, verbosity: int = 0, logfile: Path = None):
    """Capture the logger and setup name, verbosity, stream handler & rotating logfile if provided.
    Args:
        name (str, optional): Name of the logger. Defaults to \__name __. Don't change unless you know what you are doing!
        verbosity (int, optional): Verbosity level. Defaults to 0 (warning). 1 info, >=2 debug
        logfile (Path | None, optional): Logfile path. Defaults to None.
    Returns:
        logger (modified Logger object)  

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
    import sys

    # Capture the logger
    logger = logging.getLogger(name)
    # Create a stream handler
    stream_handler = logging.StreamHandler(sys.stdout)
    # Create a rotating file handler
    if logfile:
        from logging.handlers import RotatingFileHandler

        rf_handler = RotatingFileHandler(logfile, maxBytes=25 * 1024, backupCount=5)
    # Set the log level
    if verbosity == 0:
        level = logging.WARNING
    elif verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG
    logger.setLevel(level)
    stream_handler.setLevel(level)
    if logfile:
        rf_handler.setLevel(level)
    # formatter
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(filename)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler.setFormatter(formatter)
    if logfile:
        rf_handler.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(stream_handler)
    if logfile:
        logger.addHandler(rf_handler)
    logger.warning("Logger initialized @level %s", logging.getLevelName(level))
    return logger
