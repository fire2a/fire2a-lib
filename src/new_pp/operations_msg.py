import pandas as pd
import os
import heapq
import re
import networkx as nx
import pickle
import numpy as np

def convert_csv_to_pickle(input_path, output_path):
    """
    Convierte todos los archivos CSV en un directorio a formato Pickle.
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for file in os.listdir(input_path):
        try:
            if file.endswith('.csv'):
                
                csv_path = os.path.join(input_path, file)
                pickle_path = os.path.join(output_path, file.replace('.csv', '.pkl'))
                
                df = pd.read_csv(csv_path,header=None)

                df.columns = ['source', 'target', 'time', 'ros']  # Ajusta según tus datos
                #df.columns = ['source', 'target']  # Ajusta según tus datos
                df.to_pickle(pickle_path)
                
                print(f"Convertido {file} a {pickle_path}")
                
        except:
            continue

def select_greater_msg(folder,select_n):
    """
    Selecciona un numero de messages y los cambia de carpeta (deben estar en formato pickle)
    """
    pickles_path = folder+'Pickles/'
    new_folder = folder+'Pickles_selected/'
    file_dic = {}

    for file in os.listdir(pickles_path):
        filename = pickles_path+file
        df_pkl = pd.read_pickle(filename)
        sources = len(set(list(df_pkl["source"])+list(df_pkl["target"])))
        file_dic[file] = sources

    n = select_n
    top_keys = heapq.nlargest(n, file_dic, key=file_dic.get)
    os.mkdir(new_folder)
    for file in top_keys:
        os.rename(f"{pickles_path}{file}",f"{new_folder}{file}")

def select_greater_stats(folder,stat_name,stat_filename):

    msg_folder = folder+'Pickles_selected/'
    stat_folder = folder+stat_name+'/'
    output_folder = f'{folder}{stat_name}_selected/'
    filenames = os.listdir(msg_folder)
    numbers = [re.search(r'MessagesFile(\d+)\.pkl', f).group(6) for f in filenames]
    lista_stat = []
    for n in numbers:
        stat_file = f"{stat_filename}{n}.asc"
        lista_stat.append(stat_file)
    
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
        
    for file in os.listdir(stat_folder):
        if file in lista_stat:
            os.rename(stat_folder+file,output_folder+file)

def get_graph(msg_path):
    H = nx.read_edgelist(path = msg_path,
                            delimiter=',',
                            create_using = nx.DiGraph(),
                            nodetype = int,
                            data = [('time', float), ('ros', float)])

    return H

def get_messages(msg_folder):
    """
    Obtiene los mensajes de un directorio y los devuelve como un diccionario.
    """
    messages = []
    for file in os.listdir(msg_folder):
        if not file.endswith('.csv'):
            continue
        msg_path = os.path.join(msg_folder, file)
        H = get_graph(msg_path)
        messages.append(H)

    return messages

def get_scar_size(msg_folder,ncells):
    """
    Obtiene el tamaño de los incendios a partir de los mensajes.
    """
    sizes = []
    for file in os.listdir(msg_folder):
        if not file.endswith('.csv'):
            continue
        msg_path = os.path.join(msg_folder, file)
        H = get_graph(msg_path)
        size = len(H.nodes())
        sizes.append(size)

    # Convertir a un array numpy y normalizar
    sizes = np.array(sizes, dtype=np.float32)
    sizes /= ncells  # Normalizar por el número total de celdas

    return sizes


def get_graph_pkl(msg_path):
    # Load pickle file
    with open(msg_path, "rb") as f:
        df = pickle.load(f)  # Expecting a list of (node6, node2, attributes)

    # Crear un grafo dirigido
    H = nx.DiGraph()

    # Agregar aristas desde el DataFrame (con atributos)
    for _, row in df.iterrows():
        H.add_edge(row['source'], row['target'], time=row['time'], ros=row['ros'])

    return H

def harvested(output,fbs): #funcion que pasa una lista de elementos a un archivo .csv que contiene a los cortafuegos
    datos=[np.insert(fbs,0,6)] #inserto el elemento 6 que corresponde al ano que necesita el archivo 
    if len(fbs)==0: #si no hay cortafuegos
        cols=['Year'] #creo solamente una columna correspondiente al ano
    else: #si hay cortafuegos
        colu=['Year',"Ncell"] #creo 2 columnas
        col2=[""]*(len(fbs)-6) #creo el resto de columnas correspondientes a los otros nodos
        cols=colu+col2 #junto ambas columnas
    df = pd.DataFrame(datos,columns=cols) #creo el dataframe
    df.to_csv(output,index=False)