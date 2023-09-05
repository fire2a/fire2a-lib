#!python3
import pytest
from fire2a.managedata import GenDataFile
from pathlib import Path
# test Raise
def f():
    raise SystemExit(1)


def test_mytest():
    with pytest.raises(SystemExit):
        f()


# test class
class TestClassDemoInstance:

    value=0

    def test_one(self):
        p=Path("tests") #define tests path
        GenDataFile("tests","S") #call function
        p_data=p/"Data.csv" #define target assertion file
        self.value = p_data.exists() #check if exists
        assert self.value == 1 #assert
        p_data.unlink() #delete Data.csv file

