#!python3
"""👋🌎 🌲🔥
This is fire2a-lib distribution, fire2a package docstring
More info on:

    fire2a.github.io/fire2a-lib  
    fire2a.github.io/docs  
    www.fire2a.com  
"""  # fmt: skip
__author__ = "Fernando Badilla"
__version__ = 'v0.0.1+0-gf866f08'
__revision__ = "$Format:%H$"

from pathlib import Path


def setup_logger(name: str = __name__, verbosity: int = 0, logfile: Path | None = None):
    """capture the logger and set the verbosity, stream handler, rotating logfile if provided.
    logging.critical("Something went wrong, exception info?", exc_info=True)
    logging.error("Something went wrong, but we keep going?")
    logging.warning("Default message level")
    logging.info("Something planned happened")
    logging.debug("Details of the planned thing that happened")
    global logger
    """
    import logging
    import sys

    logger = logging.getLogger(name)
    # Create a stream handler
    stream_handler = logging.StreamHandler(sys.stdout)
    # Create a rotating file handler
    if logfile:
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
        "%(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    stream_handler.setFormatter(formatter)
    if logfile:
        rf_handler.setFormatter(formatter)
    # Add the handlers to the logger
    logger.addHandler(stream_handler)
    if logfile:
        logger.addHandler(rf_handler)
    return logger
