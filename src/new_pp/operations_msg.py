import pandas as pd
import os
import heapq
import re
import networkx as nx   

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

def select_greater_msg(pickles_path,new_folder,select_n):
    """
    Selecciona un numero de messages y los cambia de carpeta (deben estar en formato pickle)
    """

    file_dic = {}

    for file in os.listdir(pickles_path):
        filename = pickles_path+file
        df_pkl = pd.read_pickle(filename)
        sources = len(set(list(df_pkl["source"])+list(df_pkl["target"])))
        file_dic[file] = sources

    n = select_n
    top_keys = heapq.nlargest(n, file_dic, key=file_dic.get)

    for file in top_keys:
        os.rename(f"{pickles_path}{file}",f"{new_folder}/selected/{file}")

def select_greater_stats(msg_folder,stat_folder,stat_filename,output_folder):

    filenames = os.listdir(msg_folder)
    numbers = [re.search(r'MessagesFile(\d+)\.pkl', f).group(1) for f in filenames]
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

def harvested(output,fbs): #funcion que pasa una lista de elementos a un archivo .csv que contiene a los cortafuegos
    datos=[np.insert(fbs,0,1)] #inserto el elemento 1 que corresponde al ano que necesita el archivo 
    if len(fbs)==0: #si no hay cortafuegos
        cols=['Year'] #creo solamente una columna correspondiente al ano
    else: #si hay cortafuegos
        colu=['Year',"Ncell"] #creo 2 columnas
        col2=[""]*(len(fbs)-1) #creo el resto de columnas correspondientes a los otros nodos
        cols=colu+col2 #junto ambas columnas
    df = pd.DataFrame(datos,columns=cols) #creo el dataframe
    df.to_csv(output,index=False)