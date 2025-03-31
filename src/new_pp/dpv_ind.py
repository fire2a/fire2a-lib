import pandas as pd
import numpy as np 
import networkx as nx
from concurrent.futures import ProcessPoolExecutor
import os
from operations_raster import read_asc, write_asc


def calculate_dpv_for_message(archive,var_file,var_file2):

    header, values_risk = read_asc(var_file)
    header2, values_risk2 = read_asc(var_file2)
    shape = values_risk.shape
    ncells = values_risk.size
    values_risk = values_risk.reshape([-1])
    values_risk2 = values_risk2.reshape([-1])
    values_risk_total = values_risk+values_risk2
    
    try:
        message = pd.read_csv(archive, header=None)
        graph = nx.DiGraph()

        for _, row in message.iterrows():
            graph.add_edge(int(row[0]) - 1, int(row[1]) - 1, weight=row[2])

        root = int(message.iloc[0, 0]) - 1
        shortest_paths = nx.single_source_dijkstra_path(graph, root, weight='weight')

        # Create a new graph containing only shortest paths
        new_graph = nx.DiGraph()
        for destino, camino in shortest_paths.items():
            for i in range(len(camino) - 1):
                u, v = camino[i], camino[i + 1]
                new_graph.add_edge(u, v)

        nodes = list(new_graph.nodes)
        dpv_values = np.zeros(ncells)

        # Compute DPV for each node
        for node in nodes:
            descendants = list(nx.descendants(new_graph, node))
            dpv_values[node] = values_risk_total[node] + values_risk_total[descendants].sum()

        return dpv_values

    except Exception as e:
        print(f"Error processing {archive}: {e}")
        return np.zeros(ncells)

forest = "sub40"
forest_layer = f"/home/matias/Documents/Emisiones/{forest}/forest/fuels.asc"
values_risk_folder = f"/home/matias/Documents/Emisiones/{forest}/results/preset/SurfFractionBurn/"
values_risk_folder2 = f"/home/matias/Documents/Emisiones/{forest}/results/preset/CrownFire/"
directory = f"/home/matias/Documents/Emisiones/{forest}/results/preset/Messages/"
dpv_output = f"/home/matias/Documents/Emisiones/{forest}/results/preset/sub40_dpv.asc"
n_threads = 25

# Get list of message files
var_files = [os.path.join(values_risk_folder, f) for f in os.listdir(values_risk_folder)]
var_files2 = [os.path.join(values_risk_folder2, f) for f in os.listdir(values_risk_folder2)]
message_files = [os.path.join(directory, f) for f in os.listdir(directory)]

# Process messages in parallel
with ProcessPoolExecutor(max_workers=n_threads) as executor:
    dpv_results = list(executor.map(calculate_dpv_for_message, message_files,var_files,var_files2))

# Compute the mean DPV across all messages
header, forest_raster = read_asc(forest_layer)
shape = forest_raster.shape
dpv_final = np.mean(dpv_results, axis=0).reshape(shape)

# Write the final DPV raster
write_asc(dpv_output, header, dpv_final)
print("DPV calculation complete.")
