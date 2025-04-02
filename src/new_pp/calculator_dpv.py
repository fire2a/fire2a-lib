import pandas as pd
import numpy as np 
import networkx as nx
from concurrent.futures import ProcessPoolExecutor
from operations_raster import read_asc, write_asc

def split_into_chunks(data, n_chunks):
    """Splits a list into `n_chunks` roughly equal parts."""
    chunk_size = len(data) // n_chunks
    return [data[i * chunk_size:(i + 1) * chunk_size] for i in range(n_chunks)]

def calculate_value(graphs, values_risk,ncells):
    """Processes a batch of graphs and computes DPV values efficiently."""

    final = np.zeros(ncells, dtype=np.float32)
    num_graphs = len(graphs)

    for graph in graphs:

        for u, v, attrs in graph.edges(data=True):
            attrs['weight'] = attrs['ros']  # Asigna el peso a partir de 'ros'

        if len(graph.nodes()) == 0:
            continue
        
        root = [n for n,d in graph.in_degree() if d==0][0]
        shortest_paths = nx.single_source_dijkstra_path(graph, root, weight='weight')

        # Create a new graph containing only shortest paths
        new_graph = nx.DiGraph()
        for destino, camino in shortest_paths.items():
            for i in range(len(camino) - 1):
                u, v = camino[i], camino[i + 1]
                new_graph.add_edge(u, v)
        
        graph = new_graph
        
        nodes = np.array(graph.nodes, dtype=np.int32)  # Convertimos directamente a array numpy
        values = np.zeros(ncells, dtype=np.float32)

        descendants_dict = {node: np.array(list(nx.descendants(graph, node)), dtype=np.int32) - 1 for node in nodes}
        
        # DIVIDE BY NUMBER OF PARENTS (IDEA DEL PROFE)
        #for n in graph.nodes():
        #    parents = graph.in_degree(n)
        #    if parents > 0:
        #        values_risk[n-1] = values_risk[n-1]/int(parents)
        
        # Calculo del DPV por cada nodo
        dpv_values = np.array([values_risk[descendants].sum() if len(descendants) > 0 else 0 for node, descendants in descendants_dict.items()], dtype=np.float32)

        # Sumar el riesgo propio del nodo
        values[nodes - 1] = values_risk[nodes - 1] + dpv_values
        final += values / num_graphs

    return final

def process_dpv(graphs, values_risk_file, n_threads,dpv_output):
    """
    Processes DPV calculations in parallel using multiple threads.

    Parameters:
        graphs (list): List of NetworkX graphs.
        values_risk_file (str): Path to the ASC file with risk values.
        dpv_output (str): Output file path for the final DPV result.
        n_threads (int): Number of parallel threads to use.
    """

    # Load values risk ASC file
    header, values_risk = read_asc(values_risk_file)
    shape = values_risk.shape
    ncells = values_risk.size
    values_risk = values_risk.reshape([-1])

    # Split graphs into blocks for parallel processing
    blocks = split_into_chunks(graphs, n_threads)

    # Parallel execution
    with ProcessPoolExecutor(max_workers=n_threads) as executor: 
        resultados = executor.map(calculate_value, blocks, [values_risk] * n_threads, [ncells] * n_threads)

    # Aggregate DPV results
    dpv_final = np.zeros(ncells)
    for dpv_partial in resultados:
        dpv_final += dpv_partial
    #dpv_final = dpv_final.reshape(shape)

    # Save DPV output
    write_asc(dpv_output, header, dpv_final)
    print("DPV calculation complete.")
    return (dpv_final)


