#!python3
"""
managedata test
"""
__author__ = "David Palacios Meneses"
__version__ = 'v0.0.1+0-gf866f08'
import pytest
from fire2a.treatmentpreproccessing import bin_to_nod
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
        vector_test=[1,1,0,0,0,0,0,0,1,1,0,1] #define set of firebreak plan
        filename="test_treatment.csv" #define filename
        file_path=p/filename #define path
        bin_to_nod(vector_test,str(file_path)) #call method
        self.value = file_path.exists() #check if exists file_path
        assert self.value == 1 #assert
        file_path.unlink() #delete Data.csv file

