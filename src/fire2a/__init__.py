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
__version__ = 'v0.0.1+0-gf866f08'
__revision__ = "$Format:%H$"

from pathlib import Path


def setup_logger(name: str = __name__, verbosity: int = 0, logfile: Path | None = None):
    """Capture the logger and setup name, verbosity, stream handler & rotating logfile if provided.
    Args:
        name (str, optional): Name of the logger. Defaults to \__name __. Don't change unless you know what you are doing!
        verbosity (int, optional): Verbosity level. Defaults to 0 (warning). 1 info, >=2 debug
        logfile (Path | None, optional): Logfile path. Defaults to None.
    Returns:
        logger (modified Logger object)  

    ## Developers implementing their own logger
        * All fire2a modules uses `logger = logging.getLogger(__name__)`

        * Reuse this logging setup as shown in `usage_template/template.ipynb`
    ## typical usage
    logging.critical("Something went wrong, exception info?", exc_info=True)  
    logging.error("Something went wrong, but we keep going?")  
    logging.warning("Default message level")  
    logging.info("Something planned happened")  
    logging.debug("Details of the planned thing that happened")  
    """  # fmt: skip
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
