#!python3
"""
managedata test
"""
__author__ = "David Palacios Meneses"
__version__ = 'v0.0.1+0-gf866f08'
import pytest
from fire2a.clusteroptimization import run_model
from pathlib import Path

pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")

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
        lookupTable=p/"spain_lookup_table.csv" #lookuptable path
        forestFile=p/"fuels_2.asc" #forest path
        dpvFile=p/"dpv.asc" #dpv path
        clusterFile=p/"clusters.asc" #clusters path
        percentage=0.01 #percentage of treatment
        p_treatment=p/"treatment.csv"
        run_model(str(lookupTable),str(forestFile),str(dpvFile),str(clusterFile),str(p_treatment),percentage) #run model
        self.value = p_treatment.exists() #check if exists
        assert self.value == 1 #assert
        p_treatment.unlink() #delete Data.csv file

    
    
    
    
    

