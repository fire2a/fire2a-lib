#!python3
"""
managedata test
"""
__author__ = "David Palacios Meneses"
__version__ = 'v0.0.1+0-gf866f08'

from pytest import mark
pytestmark = mark.filterwarnings("ignore::FutureWarning")

def test_treatment_csv_exists(request,tmp_path):
    from fire2a.clusteroptimization import run_model

    assets_path = request.config.rootdir / "tests" / "assets"

    lookupTable=assets_path/"spain_lookup_table.csv" #lookuptable path
    forestFile=assets_path/"fuels_2.asc" #forest path
    dpvFile=assets_path/"dpv.asc" #dpv path
    clusterFile=assets_path/"clusters.asc" #clusters path
    percentage=0.01 #percentage of treatment
    p_treatment=tmp_path/"treatment.csv"
    run_model(str(lookupTable),str(forestFile),str(dpvFile),str(clusterFile),str(p_treatment),percentage) #run model
    value = p_treatment.exists() #check if exists
    assert value == 1 #assert
    p_treatment.unlink() #delete Data.csv file

    
    
    
    
    

