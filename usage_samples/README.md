**Using fire2a template sample**

* 4 ways
* restart the kernel to make sure each runs independently
* fire2a-lib install: https://github.com/fire2a/fire2a-lib/#quickstart

## 1. Use the functions

```python
from fire2template.template import calc, cast

numbers = cast(["1", "2", "3"])
result = calc("*", numbers)
print(f"{result=}")
```

## 2. + also watch logs
### a. basic logging config
Basic config modifies all loggers bluntly

```python
from fire2template.template import calc, cast
import logging
logging.basicConfig(level=logging.DEBUG, format="[%(asctime)s] %(levelname)s [%(name)s:%(lineno)s] %(message)s")

numbers = cast(["4", "5", "6"])
result = calc("*", numbers)
print(f"{result=}")
```

### b. fire2a logger setup
Allow more options and integration

```python
from fire2template.template import calc, cast
from fire2template import setup_logger
logger = setup_logger(verbosity=3)
# DONT DO THIS: naming sets a new logger
# logger = setup_logger('jupy',3,None)

logger.warning("hello world!")

numbers = cast(["4", "5", "6"])
result = calc("*", numbers)
print(f"{result=}")

logger.warning("bye world!")
```

## 3. run it from main

```python
from fire2template.template import main

if 0==main(["-vv","--operation", "/","7.7", "8.8", "9.9"]):
    print("success")
```

## 4. run it from command line
'*' globs in command line so must be quoted

```python
! python3 -m fire2template.template -v --operation '*' 7.7 8.8 9.9
```

```python

```
