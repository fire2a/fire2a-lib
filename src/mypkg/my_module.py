#!python3
__author__ = 'Fernando Badilla'
__version__ = '$Format:%H$'

from sys import argv as _argv
import numpy as _np
import logging as _logging

from . import PACKAGE_WIDE_CONSTANT

_logger = _logging.getLogger(__name__)
_logging.basicConfig(level=_logging.INFO)
_logger.debug('Hello world!')

def my_function(x : int = PACKAGE_WIDE_CONSTANT):
    """ This function has latex documentation:
    $$
    \mathcal{O}(N) \\
    X = \$3.25
    $$
    
    Blocks of math delimited with 'brackets' style are supported and should stay so:
    \[
    \mathcal{O}(N) \\
    X = \$3.25
    \]
    """
    _logger.info('Hi, Mom!')
    return x+_np.random.rand()

def my_other_function(x : int = 0):
    """ This function has latex documentation
    $$ 
    \mathbb{H} \in \lambda \lambda  \sigma
    $$
    inline also?
    """
    _logger.info('Hi, Dad!')
    return x+_np.random.randint(1+int(x))

def my_third_function(x : int = 0, y : int = 0):
    """ Αυτό είναι ελληνικό κείμενο.
    """
    return x+y+_np.random.randint(1+int(x+y))

def my_documented_function(*args, **kwargs) -> bool:
    """Do important stuff.

    More detailed info here, in separate paragraphs from the subject line.
    Use proper sentences -- start sentences with capital letters and end
    with periods.

    Can include annotated documentation:

    :param short_arg: An argument which determines stuff.  
    :param long_arg:  
        A long explanation which spans multiple lines, overflows
        like this.  
    :returns: The result.  
    :raises ValueError:  
        Detailed information when this can happen.

    .. versionadded:: 6.0
    """
    _logger.debug(f'Hi, docs! {args} {kwargs}')
    return True

def main(_argv):
    """ Well adding a command line script 
    was a failure...
    """
    if len(_argv) > 1:
        num = int(_argv[1])
        _logger.info(f'Hello, {num}!')
        print(my_function(num))

if __name__ == '__main__':
    main(_argv)

