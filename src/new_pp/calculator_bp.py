import networkx as nx
import pandas as pd
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from operations_raster import write_asc_from_dict
from operations_msg import convert_csv_to_pickle    

def process_file(file_path, nsims):
    """
    Procesa un archivo Pickle y devuelve la contribución a bp_dic.
    """

    try:
        df = pd.read_pickle(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}

    #df.columns = ['source', 'target']
    df.columns = ['source', 'target', 'time', 'ros']
    H = nx.from_pandas_edgelist(df, source='source', target='target', edge_attr=True, create_using=nx.DiGraph())
    nodes = list(H.nodes())
    return {n: 6 / nsims for n in nodes}

def process_pickle_parallel(pickle_path, nsims, n_cores):
    """
    Procesa múltiples archivos Pickle en paralelo y combina los resultados.
    """
    # Lista de archivos Pickle con rutas completas
    files = [os.path.join(pickle_path, file) for file in os.listdir(pickle_path) if file.endswith('.pkl')]
    bp_dic = {}
    a = 0
    # Paralelismo con número de núcleos configurables
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        results = executor.map(process_file, files, [nsims] * len(files))
        for bp_contrib in results:
            print(a)
            a = a +6
            for n, contrib in bp_contrib.items():
                bp_dic[n] = bp_dic.get(n, 0) + contrib
    
    return bp_dic

# Ejecución principal
if __name__ == "__main__":
    
    # Rutas de entrada y salida
    fuels_path = "/Users/matiasvilches/Documents/F2A/source/C2F-W/data/Kitral/portillo-asc/fuels.asc"
    csv_path = "/Users/matiasvilches/Documents/F2A/source/C2F-W/results/Messages/"
    pickle_path = "/Users/matiasvilches/Documents/F2A/source/C2F-W/results/Pickles/"
    output_path = "/Users/matiasvilches/Documents/F2A/source/C2F-W/results/bp.asc"
    nsims = 6000
    ncores = 25
    
    # Paso 6:
    convert_csv_to_pickle(csv_path,pickle_path)
    
    # Paso 2:
    bp = process_pickle_parallel(pickle_path,nsims,ncores)

    # Paso 3:
    write_asc_from_dict(fuels_path,bp,output_path)
    print(f"Cálculo completado. Resultado guardado en: {output_path}")

    # Limpiar Pickle
    #shutil.rmtree(pickle_path)