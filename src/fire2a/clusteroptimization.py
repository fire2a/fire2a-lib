#!python3
"""
Optimization module
"""
__author__ = "David Palacios Meneses"
__version__ = 'v0.0.1+0-gf866f08'

from pyomo.environ import Var, value, ConcreteModel, Constraint, linear_expression, Set, Binary, Param, Objective, maximize, SolverFactory
from collections import defaultdict
from time import time
from fire2a.managedata import Lookupdict, ForestGrid
from fire2a.raster import read_raster
from fire2a.treatmentpreproccessing import bin_to_nod
from pathlib import Path
def values_per_cluster(dpv:dict)->dict:
  """
  Gets dpv values per each cell and returns the average dpv of each cluster

  Args:
    dpv (dict [int, list [tuple [float, int]]]): DPV values of each cell as a dictionary, where each cell is represented as key,\
    and the value corresponds to a list containing both dpv value of the cell and id cluster to which said cell belongs

  Returns:
    dict [int, float] Dictionary with the mean dpv of each cluster
  """ #fmt: skip

  all_values=list(dpv.values())
  out_dict={}
  aux_counter={}
  for value,rodal in all_values:
    if rodal not in out_dict:
      out_dict[rodal]=value
      aux_counter[rodal]=1
    else:
      out_dict[rodal]+=value
      aux_counter[rodal]+=1
  for key in out_dict:
    out_dict[key]=out_dict[key]/aux_counter[key]
  return out_dict
  



def obj_rule(model:ConcreteModel) -> linear_expression:
    r"""
    Pyomo model objective function:
         
    $$
    \sum_{c \in C} x_{c} \cdot DPV_{c}
    $$   
    where C corresponds to a set of clusters, x corresponds to the decision of selecting fuel treatment and DPV corresponds
    to the dpv of the respective cluster

    Args:
        model (ConcreteModel): A pyomo model

    Returns:

        linear_expression:  Objective function
    """
        
    return sum(model.fpv[i]*model.x[i] for i in model.I)

def optimization_model(treatment_percentage:float,AvailCells:set,cluster_size:dict,cluster_dpv:dict) -> Var:
    """
    Pyomo optimization model

    Args:
        treatment_percentage (float [0,1]): Max percentage of AvailCells to treat
        AvailCells (set[int]): Set of Available Cells (do not includes cells with no data, no fuel or firebreak)
        cluster_size (dict[int,int]): Dictionary with clusters as key and number of cells of each cluster as value
        cluster_dpv (dict[int,float]): Dictionary with clusters as key and average DPV of each cluster as value 

    Returns:

        Var:  Decision variable that indicates objective clusters to treat
    """#fmt: skip
     
    max_percentage=int(treatment_percentage*len(AvailCells))

    clusters=cluster_size.keys()

    model= ConcreteModel()

    model.I=Set(initialize=list(clusters))
    model.x = Var(model.I, 
                    domain=Binary, 
                    initialize = 0)
    model.fpv = Param(model.I, 
                        initialize = cluster_dpv ,
                        default = 0)
    model.csize=Param(model.I, initialize=cluster_size)
    
    model.z = Objective(rule=obj_rule, sense=maximize)

    model.cons= Constraint(expr=sum(model.x[i]*model.csize[i] for i in model.I) <= max_percentage)
    solver = SolverFactory('glpk',executable="C://Users//david//anaconda3//Library//bin//glpsol.exe")
    t0 = time()
    print("solving...")
    results = solver.solve(model,logfile="lg.log")
    tf = time()
    t = tf-t0
    print(f'Resolution Time: {t}')
    print("Optimization Model Results:")
    model.z.display()

    return model.x

def manage_cluster_data(dpvFile:str,clusterFile:str):
    """
    Manages DPV and clusters data for optimization model format
    
    Args:
        dpvFile (str): Downstream Protection Value ASCII file
        clusterFile (str): Cluster ASCII file, where each cell has the cluster identifier to which it belongs

    Returns:
        dpv_values (dict[int,float]): Dictionary with clusters as key and average DPV as value
        cluster_size (dict[int,int]): Dictionary with clusters as key and number of cells of each cluster as value
        cells_in_clusters (dict[int,[list[int]]]): Dictionary with clusters as key and a list containing cells id \
        inside that cluster as value
        
    """

    all_dpv_values,dpvmetadata=read_raster(dpvFile)  #array of dpv values
    cluster_array,dpvmetadata=read_raster(clusterFile) #array
    dpv_values={}
    cluster_keys={}
    c=1
    for _, row in enumerate(all_dpv_values):
        for _,value in enumerate(row):
            dpv_values[c]=[value]
            cluster_keys[c]=int(value)
            c+=1
    c=1
    for _, row in enumerate(cluster_array):
        for _,value in enumerate(row):
            cluster_keys[c]=int(value)
            c+=1
    cells_in_clusters=defaultdict(list)
    for key, value in sorted(cluster_keys.items()):
        cells_in_clusters[value].append(key) #dict with cluster id as key and list of cells id as value
    cluster_size={}
    for key in cells_in_clusters:
        cluster_size[key]=len(cells_in_clusters[key]) #dict with cluster id as key and number of cells inside that cluster as value
    for k, v in list(dpv_values.items()):
        if v[0] < 0:
            del dpv_values[k]
        else:
            dpv_values[k].append(cluster_keys[k])

    return dpv_values,cluster_size,cells_in_clusters


def run_model(lookupTable:str,ForestFile:str,dpvFile:str,clusterFile:str,treatment_file_name:str,treatment_percentage:float) -> None:
    """
    main function call for run model optimization for treatment to cluster
    
    Args:
        lookupTable (str): location of lookuptable .csv
        ForestFile (str): location of landscape data; a matrix that stores the fuel code of each cell
        dpvFile (str): location of dpv file; a matrix that stores the dpv of each cell
        clusterFile (str): location of cluster file; a matrix that stores the cluster id of each cell
        treatment_file_name (str): target name of the treatment file to be written, it is mandatory a .csv extension
        treatment_percentage (float[0,1]): target percentage of area of the landscape to treat; varies from 0 to 1
    Returns:
        None

    Raises:

        ValueError: If the extension of the files are not correct, or treatment percentage is not between 0 and 1
    
    """
    if lookupTable[-4:]!=".csv":
        raise ValueError("Extension must be .csv")
    elif ForestFile[-4:]!=".asc":
        raise ValueError("Extension must be .asc")
    elif dpvFile[-4:]!=".asc":
        raise ValueError("Extension must be .asc")
    elif clusterFile[-4:]!=".asc":
        raise ValueError("Extension must be .asc")
    elif treatment_file_name[-4:]!=".csv":
        raise ValueError("Extension must be .csv")
    elif treatment_percentage<0 or treatment_percentage>1:
        raise ValueError("Treatment percentage must be between 0 and 1")
    

    print("generating landscape data")
    FBPDict, _ = Lookupdict(lookupTable)
    _, _, Rows, Cols, AdjCells, _, _ = ForestGrid(ForestFile,FBPDict)
    NCells = Rows * Cols

    AvailSet = set()
    AvailCells=set()

    setDir = ['S', 'SE', 'E', 'NE', 'N', 'NW', 'W', 'SW']
    aux = set([])
    Adjacents = {} # dict of adjacency of availCells
    for k in range(NCells):
        aux = set([])
        for i in setDir:
            if AdjCells[k][i] != None:
                if AdjCells[k][i][0] in AvailSet :
                    aux = aux | set(AdjCells[k][i])
        Adjacents[k + 1] = aux & AvailSet
    print("calling cluster data")
    dpv_values,cluster_size,cells_in_clusters=manage_cluster_data(dpvFile,clusterFile)
    print("arranging cluster dpv")
    cluster_dpv=values_per_cluster(dpv_values)
    print("calling model")
    optimization_plan=optimization_model(treatment_percentage,AvailCells,cluster_size,cluster_dpv)

    print("arranging firebreak plan")
    S = set()  

    for i in cluster_size.keys():
        if value(optimization_plan[i]) > 0: 
        # print 'x[' + str(i) + '] = ', value(model.x[i])
            S.add(i)
    firebreaks=[]
    for k in S:
        for element in cells_in_clusters[k]:
            firebreaks.append(element)
    bin_to_nod(firebreaks,treatment_file_name)
    print("success")

    ###for version 2.0. generate raster version of the firebreak treatment
    return None




