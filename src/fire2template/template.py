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
__version__ = 'v0.0.1+0-gf866f08'
__revision__ = "$Format:%H$"


import logging
import sys
from pathlib import Path

import numpy as np

from fire2template import setup_logger

logger = logging.getLogger(__name__)
MODULE_VARIABLE = "very important and global variable"
""" this docstring describes a global variable has 0 indent """


def cast(numbers):
    logger.debug(f"cast: before {numbers=}")
    resp = list(map(float, numbers))
    logger.debug(f"cast: after {resp=}")
    return resp


def calc(operation, numbers):
    """mock calculator"""
    logger.debug(f"calc: {operation=}, {numbers=}")
    if operation == "+":
        logger.info("attempting summation...")
        return np.sum(numbers)
    elif operation == "-":
        logger.info("attempting substraction...")
        return np.subtract(*numbers)
    elif operation == "*":
        logger.info("attempting multiplication...")
        return np.prod(numbers)
    elif operation == "/":
        logger.info("attempting division by the last non zero in the list...")
        for i, dividend in enumerate(numbers[::-1]):
            logger.debug(f"try {dividend=}")
            if dividend == 0:
                continue
            break
        logger.debug(f"got {dividend=}")
        if dividend == 0:
            logger.error("Avoiding division by zero")
            return
        return np.divide(numbers[:i] + numbers[i + 1 :], dividend)


def argument_parser(argv):
    """parse arguments
    - logger: verbosity, logfile
    - mock calculator: operation and numbers"""
    # fmt: off
    import argparse
    parser = argparse.ArgumentParser(
        description="Simplest module to serve as a template. It's function is to perform a simple operation on a list of numbers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # fmt: on
    parser.add_argument(
        "-v",
        "--verbosity",
        help="increase output verbosity (0: warning(default), 1: info, 2>=: debug)",
        action="count",
        default=0,
    )
    parser.add_argument("-l", "--logfile", help="log file", type=Path)
    parser.add_argument(
        "-op", "--operation", help="specify operation to perform", default="+", type=str, choices=["+", "-", "*", "/"]
    )
    parser.add_argument(nargs="+", dest="numbers", help="numbers to perform operation on")
    return parser.parse_args(argv)


def main(argv=None):
    """this is a function docstring that describes a function"""
    if argv is None:
        argv = sys.argv[1:]
    args = argument_parser(argv)
    logger = setup_logger(__name__, args.verbosity, args.logfile)
    logger.info(f"{args=}")
    logger.debug(f"debugging...")
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
