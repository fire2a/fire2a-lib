#!python3
"""👋🌎 

This docstring is called a module docstring. Describes the general purpose of the module.

In this case being a sample module to serve as a pattern for creating new modules.

It has docstrings for:  
- module  
- global variable  
- method  
- class  

Implements:  
- skipping black formating of docstrings using # fmt: skip/on/off  
- logging  
- module cli using main & argparse:  
```
$ python -m fdolib.template --help
```
ipython  
%autoindent  
from fire2a.template import a_method
a_method((1, 'a'), 'b', 'c', an_optional_argument=2, d='e', f='g')
👋🌎
"""  # fmt: skip
__author__ = "Fernando Badilla"
__revision__ = "$Format:%H$"


import logging
import sys
from pathlib import Path

from fire2template import setup_file

logger = logging.getLogger(__name__)
"""capture the logger"""
# adjust harcoded values for ipython (or jupyter)
NAME, FILEPATH = setup_file(name="template", filepath=Path("/home/fdo/source/fire2a-lib/src/fire2template"))
""" global variables available in python and ipython(setup manually)"""
# REPO = FILEPATH.parent.parent
# DEPOT = FILEPATH / "depot"

MODULE_VARIABLE = "very important and global variable"
""" this docstring describes a global variable has 0 indent """


def cast(numbers):
    """cast a list of strings to a list of floats
    Args:
        numbers (list): list of strings
    Returns:
        list: list of floats
    Raises:
        SystemExit: if a string cannot be casted to a float
    """
    logger.debug(f"cast: before {numbers=}")
    try:
        resp = list(map(float, numbers))
    except ValueError as e:
        logger.fatal("%s", e)
        sys.exit(1)
    logger.debug(f"cast: after {resp=}")
    return resp


def calc(operation, numbers):
    """mock calculator that performs a simple operation on a list of numbers
    Args:
        operation (str): operation to perform
        numbers (list): list of numbers
    Returns:
        float: result of the operation
    """
    from functools import reduce

    from numpy import prod as np_prod
    from numpy import sum as np_sum

    logger.debug(f"calc: {operation=}, {numbers=}")
    if operation == "+":
        logger.info("attempting summation...")
        return np_sum(numbers)
    elif operation == "-":
        logger.info("attempting substraction...")
        return reduce(lambda x, y: x - y, numbers)
    elif operation == "*":
        logger.info("attempting multiplication...")
        return np_prod(numbers)
    elif operation == "/":
        logger.info("attempting division, replacing /0s by 1")
        return reduce(lambda x, y: x / y if y != 0 else 1, numbers)


def arg_parser(argv):
    """parse arguments: operation, numbers, verbosity, logfile
    Args:
        argv (list): list of strings
    Returns:
        argparse.Namespace: parsed arguments
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Simplest module to serve as a template. It's function is to perform a simple operation on a list of numbers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-op",
        "--operation",
        help="specify operation to perform (escape * to work: \*)",
        default="+",
        type=str,
        choices=["+", "-", "*", "/"],
    )
    parser.add_argument(nargs="+", dest="numbers", help="numbers to perform operation on")
    parser.add_argument("--verbose", "-v", action="count", default=0, help="WARNING:1, INFO:2, DEBUG:3")
    parser.add_argument(
        "--logfile",
        "-l",
        action="store_true",
        help="enable 5 log files named " + NAME + ".log (verbose must be enabled)",
        default=None,
    )
    args = parser.parse_args(argv)
    if args.logfile:
        args.logfile = NAME + ".log"
    return args


def main(argv=None):
    """main function to be called by the cli

    args = arg_parser(["-vvv"])
    """

    if argv is sys.argv:
        argv = sys.argv[1:]
    args = arg_parser(argv)

    if args.verbose != 0:
        global logger
        from fire2template import setup_logger

        logger = setup_logger(verbosity=args.verbose, logfile=args.logfile)
        # set other modules logging level
        logging.getLogger("asyncio").setLevel(logging.INFO)

    logger.info("args %s", args)

    if not args.numbers:
        logger.error("No numbers provided")
        return 1
    logger.info("attempting casting to float...")
    numbers = cast(args.numbers)
    logger.info("attempting mock calculator...")
    result = calc(args.operation, numbers)
    logger.info(f"{result=}")
    print(f"{result=}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
