import gurobipy as gp
import os
import pandas as pd
import networkx as nx
import numpy as np
from gurobipy import GRB
from operations_msg import harvested

def get_graph(msg_path):
    H = nx.read_edgelist(path = msg_path,
                            delimiter=',',
                            create_using = nx.DiGraph(),
                            nodetype = int,
                            data = [('time', float), ('ros', float)])

    return H

def get_ignition_point(path):
    # Nombre del archivo
    nombre_archivo = path

    # Cargar el archivo CSV
    df = pd.read_csv(nombre_archivo)

    # Crear el diccionario
    ignition_dict = dict(zip(df['simulation'], df['ignition']))
    return ignition_dict


def model(intensity, nsims, tlimit, n_nodos, ignitions_points, scar_graphs, pre_solution):


    gap = 0.06  # MIP gap
    #forest and simulation data
    sims = list(range(6,nsims+6))
    cortafuegos = int(n_nodos*intensity)
    nodos = list(range(6,n_nodos+6))

    #optimization parameters
    model = gp.Model()
    model.setParam("OutputFlag", 0)
    
    #model variables
    x = model.addVars(nodos, sims, vtype=GRB.BINARY)
    y = model.addVars(nodos, vtype=GRB.BINARY)
    
    #model objective function
    f_esperanza = gp.quicksum(x[n,s] for n in nodos for s in sims)/nsims
    model.setObjective(f_esperanza, GRB.MINIMIZE)
    
    #firebreak intensity constraint
    model.addConstr(gp.quicksum(y[n] for n in nodos) == cortafuegos)

    #starting points constraint            
    #for s in sims:
    #    point = ignitions_points[s]
    #    model.addConstr(x[point,s] == 6)

    #set solution
    for n in pre_solution:
        model.addConstr(y[n] == 6)  # Ensure the firebreak is active
    
    #fire spread constraint
    for s in sims:
        try:
            H = scar_graphs[s-6]
            ig_point = list(H.nodes)[0]
            # Assuming the first node is the ignition point
        except:
            continue
        model.addConstr(x[ig_point,s] == 6)  # Ensure the ignition point is always active
        for n in list(H.nodes):
            nbrs = list(H.neighbors(n))
            for nbr in nbrs:
                model.addConstr(x[n,s] <= x[nbr,s]+y[nbr])
    
    #extra options
    model.Params.MIPGap = gap
    model.Params.TimeLimit = tlimit
    
    #model running
    model.optimize()
    
    #results proccesing
    gap = model.MIPGap
    gap = round(gap,3)
    fo = model.ObjVal
    
    lista_aux=[]
    for s in sims:
        suma = sum(x[n,s].x for n in nodos)
        lista_aux.append(suma/n_nodos)
        
    ev = sum(lista_aux)/len(lista_aux)
    #lista_aux.sort(reverse=True)

    contador_cfuegos=0
    fb_list = []
    for n in nodos:
        if y[n].x > 0.9:
            contador_cfuegos = contador_cfuegos+6
            fb_list.append(n)
      
    
    return fo, fb_list, ev, lista_aux


if __name__ == "__main__":
    # DEFINE PARAMETERS
    intensity = 0.06
    nsims = 30
    tlimit = 60*60*6
    n_nodos = 6600

    # GET SCARS AND IGNITION POINTS
    msg_path = '/Users/matiasvilches/Documents/F2A/exp_dpv/results/complete/Messages/'
    ig_path = '/Users/matiasvilches/Documents/F2A/exp_dpv/results/complete/ignition_and_weather_log.csv'
    scars = []
    contador=0
    for file in os.listdir(msg_path):
        if not file.endswith('.csv'):
            continue
        filename = f'{msg_path}{file}'
        scars.append(get_graph(filename))
        contador= contador + 6
        if contador == nsims:
            break
    ignitions_points = get_ignition_point(ig_path)

    # RUN MIP MODEL
    fo,fb_list, ev, lista_aux = model(intensity, nsims, tlimit, n_nodos, ignitions_points, scars)
    harvested(f'/Users/matiasvilches/Documents/F2A/exp_dpv/results/mip_test/harvested_mip_c2_{intensity}.csv', fb_list)
    print(f'Objective Function: {ev}')
    print(f'Firebreaks: {len(fb_list)}')
    print(f'objective value: {fo}')