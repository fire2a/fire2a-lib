import networkx as nx
import os
from calculator_dpv import *
import rasterio
import gc
import time

def get_graph(msg_path):
    H = nx.read_edgelist(path = msg_path,
                            delimiter=',',
                            create_using = nx.DiGraph(),
                            nodetype = int,
                            data = [('time', float), ('ros', float)])

    #nodos = list(H.nodes())
    return H

def read_asc(file_path):
    with open(file_path, 'r') as file:
        # Leer las primeras seis líneas y almacenarlas
        header = [next(file) for _ in range(6)]
        # Leer los datos numéricos
        data = np.loadtxt(file, dtype=np.float32)
    return header, data

def get_max_dpv(stats):
    key = max(stats, key=stats.get)
    return key

def raster_to_dict(raster_data):

    rows, cols = raster_data.shape
    # Crear el diccionario con el ID de la celda como clave
    raster_dict = {}
    cell_id = 1

    for row in range(rows):
        for col in range(cols):
            raster_dict[cell_id] = raster_data[row, col]
            cell_id += 1

    return raster_dict

def write_treatment(header_file,firebreaks,output_path):

    header = {}
    for line in header_file:
        key, value = line.strip().split(maxsplit=1)
        header[key] = float(value) if '.' in value or 'e' in value.lower() else str(value)

    ncols = int(header['ncols'])
    nrows = int(header['nrows'])

    data = np.zeros((nrows, ncols))

    for n in firebreaks:

        row = n // ncols  # Calculate row index
        col = n % ncols   # Calculate column index

        data[row][col-1] = 1

    with open(output_path, 'w') as file:
        # Escribir el encabezado
        file.writelines(header_file)
        # Escribir los datos numéricos
        np.savetxt(file, data, fmt='%.6f')

def remove_node(G,node):
    outgoing_edges = list(G.out_edges(node_to_remove))
    G.remove_edges_from(outgoing_edges)
    # Get root
    lista_aux = [n for n,d in G.in_degree() if d==0]
    root = lista_aux[0]
    # Eliminar el nodo
    G.remove_node(node_to_remove)
    componentes_conexos = list(nx.weakly_connected_components(G))
    g_aux = [sub for sub in componentes_conexos if root in sub]
    if len(g_aux) > 0:
        final_graph = g_aux [0]
        final_graph = G.subgraph(final_graph).copy()
        return final_graph
    else:
        return None

def update_dpv(graph,node,values_risk):
    descendants = list(nx.descendants(graph, node))    
    dpv_values = values_risk[np.array(descendants) - 1].sum()
    new_value = values_risk[node - 1] + dpv_values
    return new_value
    
def update_var(graph,node):
    parents = graph.in_degree(node)
    if parents > 0:
        value = values_risk[node-1]/int(parents)
    else:
        value = values_risk[node-1]
        
    return value

if __name__ == "__main__":

    folder = "/home/matias/Documents/test_msg/full_msg"
    forest = "/home/matias/Documents/Emisiones/sub40/forest/fuels.asc"

    #INITIALIZATION
    graphs_init = []
    firebreaks = []
    firebreak_limit = 48

    header,data = read_asc(forest)

    #PRINT GRAPH INFO
    contador = 1
    msg_folder = "/home/matias/Documents/test_msg/full_msg/Messages/"
    for file in os.listdir(msg_folder):
        print(contador)
        filename = msg_folder+file
        g = get_graph(filename)
        graphs_init.append(g)
        if contador == 1000:
            break
        contador = contador + 1
    
    #CALCULATE DPV
    print("Gettin First DPV")
    values_risk = "/home/matias/Documents/Emisiones/sub40/forest/uniraster.asc"
    n_threads=25
    dpv = process_dpv(graphs_init,values_risk, n_threads)
    dpv_dic = raster_to_dict(dpv)
    max_key = get_max_dpv(dpv_dic)
    firebreaks.append(max_key)
    contador = 1
    print(f"cortafuego {contador}")
    print(max_key)
    
    graphs = graphs_init.copy()
    del graphs_init
    gc.collect()

    header, values_risk = read_asc(values_risk)
    shape = values_risk.shape
    values_risk = values_risk.reshape([-1])

    t1 = time.time()
    while len(firebreaks) < firebreak_limit:

        #PARAMETERS
        contador = contador+1
        graphs_cut = []
        node_to_remove = firebreaks[-1]
        
        #CUTTING GRAPHS
        print("ENTERING GRAPH CUT ZONE")
        descendants_dictionary = []
        for G in graphs:
            # Obtener y eliminar las aristas salientes del nodo
            if G.has_node(node_to_remove):
                descendants_dictionary.append(list(nx.descendants(G, node_to_remove)))
                cutted_tree = remove_node(G,node_to_remove)
                if cutted_tree:
                    graphs_cut.append(cutted_tree)
                else:
                    continue
            else:
                graphs_cut.append(G)

        
        
        #CALCULATING DPV
        print("UPDATING DPV ZONE")
        indice = 0
        for g in graphs_cut:
            
            descendants_list = descendants_dictionary[indice]
            for desc in descendants_list:
                if desc in g:
                    value = update_var(g,desc)
                    values_risk[desc-1] = value
            
            actualized_dpvs = []
            for desc in descendants_list:    
                #UPDATE DPV
                if desc in g:
                    ancestros = nx.ancestors(g,desc)
                    for an in ancestros:
                        if an not in actualized_dpvs:
                            dpv_dic[an] = update_dpv(g,an,values_risk)
                            actualized_dpvs.append(an)
            indice = indice+1

        
        #SAVING FIREBREAK
        max_key = get_max_dpv(dpv_dic)
        firebreaks.append(max_key)
        
        #PRINTING STATUS
        print(f"cortafuego {contador}")
        print(max_key)

        #UPDATE GRAPH
        del graphs
        graphs = graphs_cut.copy()
        del graphs_cut
        gc.collect()
        
    t2 = time.time()
    print("SEGUNDOS DE EJECUCION:",t2-t1)
    print(firebreaks)
    output = "ff_firebreaks_1k_faster.asc"
    write_treatment(header,firebreaks,output)