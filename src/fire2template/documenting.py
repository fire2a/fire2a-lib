#!python3
"""👋🌎 This docstring is called a module docstring. Describes the general purpose of the module.

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

MODULE_VARIABLE = "very important and global variable"
""" this docstring describes a global variable   
has 0 indent """  # fmt: skip

import logging
import sys

import numpy as _np

logger = logging.getLogger(__name__)


def latex_how_to():
    r"""raw docstring with $\frac{x}{y}$.

    By prefixing r to the string (see python raw strings) you can use latex math expressions witout escaping \ twice

    See more https://github.com/pdoc3/pdoc/issues/410
    """
    pass


def a_method(
    a_required_argument: tuple[int, str, float], *args, an_optional_argument: dict[str, float] = {"key": 0}, **kwargs
) -> tuple[bool, dict]:
    """ a_method implementing type hinting and checking

    Args:
        a_required_argument (tuple[int, str, float]): representing blabla
        an_optional_argument (dict[str, float], optional): configuring the algorithm, defaults to {'key':0}.

    Returns:
        dict: The output being ...

    Raises:  
        TypeError: if args are not of the expected type
    """  # fmt: skip
    a, b, c = a_required_argument
    if not isinstance(a, int):
        raise TypeError("a is not an int")
    if not isinstance(b, str):
        raise TypeError("b is not a str")
    if not isinstance(c, float):
        raise TypeError("c is not a float")

    if an_optional_argument:
        if not isinstance(an_optional_argument, dict):
            raise TypeError("an_optional_argument is not a dict")
        for key, value in an_optional_argument.items():
            if not isinstance(key, str):
                raise TypeError("an_optional_argument key is not a str")
            if not isinstance(value, float):
                raise TypeError("an_optional_argument value is not a float")
            if not isinstance(value, float):
                raise TypeError("an_optional_argument value is not a float")


def b_method(a_required_argument: tuple[int, str], *args, an_optional_argument: int = 0, **kwargs) -> dict:
    """ this is a method docstring that describes a method """  # fmt: skip
    a, b = a_required_argument

    for arg in args:
        logger.debug("log *args %s", arg)

    for key, value in kwargs.items():
        logger.debug("log **kwargs key:%s value:%s", key, value)

    logger.info("info log %s", MODULE_VARIABLE[::-1])
    """ this is not a docstring, just a comment """

    return {"a": a, "b": b}


class AClass:
    """this is a class docstring"""

    class_variable = 1
    """ this docstring describes a class variable"""

    def __init__(self, arg1, arg2):
        self.arg1 = arg1

    def do_something(self, *args, **kwargs):
        """does add arguments + class_variable
        Description of the function and its arguments.

        Args:
            param1 (type): Description of the first parameter
            param2 (type): Description of the second parameter

        Returns:
            return_type: Description of the return value

        Args:
           *args (int): arguments to be added to class_variable
           **kwargs (int): keyword arguments to be added to class_variable

        Returns:
            int: The sum of args and class_variable

        Raises:
            TypeError: if args are not of the expected type
        """
        return arg1 + arg2 + class_variable


def main(argv):
    """This is a function docstring that describes a function"""
    logger = setup_logger(__name__, 2, None)
    logger.info("Hello world!")
    logger.info(f"argv:{argv}")
    returns = b_method((1, "a"), "b", "c", an_optional_argument=2, d="e", f="g")
    print(f"{returns=}")
    logging.debug("b_method returns %s", returns)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
