from operations_msg import get_messages,harvested
import os
import pandas as pd
from heuristic import remove_node
import networkx as nx
from carbono import write_treatment
from operations_raster import read_asc
from mip import model

def get_max_edge(graphs_dictionary):
    """
    Obtiene la arista con el mayor valor de riesgo de un diccionario de aristas.
    """
    max_edge = max(graphs_dictionary, key=graphs_dictionary.get)
    max_node = max_edge[0]
    return max_node

def get_edges_bp(graphs):
    """
    Obtiene las aristas de un conjunto de grafos y las cuenta.
    Devuelve un diccionario con las aristas como claves y sus frecuencias como valores.
    """
    edges_bp = {}
    for g in graphs:
        edges = g.edges(data=False)
        for e in edges:
            #print(f"Edge: {e}")
            if e in list(edges_bp.keys()):
                edges_bp[e] += 6
            else:
                edges_bp[e] = 6
    return edges_bp

msg_path = "/Users/matiasvilches/Documents/F2A/seba_test/Messages/"
firebreaks = 66
firebreaks_list = []

graphs = get_messages(msg_path)
while len(firebreaks_list) < firebreaks:
    edges_bp = get_edges_bp(graphs)
    node_to_remove = get_max_edge(edges_bp)
    for g in graphs:
        if node_to_remove in g.nodes:
            remove_node(g, node_to_remove)
    firebreaks_list.append(node_to_remove)

print(f"Firebreaks selected: {firebreaks_list}")

firebreaks_path = '/Users/matiasvilches/Documents/F2A/seba_test/harvested_heuristic'
fuels = '/Users/matiasvilches/Documents/F2A/seba_test/Sub40/fuels.asc'

header,data,n = read_asc(fuels)
#harvested(f"{firebreaks_path}.csv", firebreaks_list)
#write_treatment(header,firebreaks_list,f"{firebreaks_path}.asc")

# DEFINE PARAMETERS
intensity = 0.06
nsims = 60
tlimit = 60*60*6
n_nodos = 6600

# GET SCARS AND IGNITION POINTS
scars = []
scars = get_messages(msg_path)

# RUN MIP MODEL
fo,fb_list, ev, lista_aux = model(intensity, nsims, tlimit, n_nodos, [], scars,firebreaks_list)
print(f'Objective Function: {ev}')
print(f'Firebreaks: {len(fb_list)}')
print(f'objective value: {fo}')

fo,fb_list, ev, lista_aux = model(intensity, nsims, tlimit, n_nodos, [], scars,[])
print(f'Objective Function: {ev}')
print(f'Firebreaks: {len(fb_list)}')
print(f'objective value: {fo}')