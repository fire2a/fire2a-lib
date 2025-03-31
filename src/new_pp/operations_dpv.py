import numpy as np
from operations_raster import read_asc
from operations_raster import write_asc
from heapq import nlargest

def write_treatment(fuels,firebreaks,output_path):

    #fuels: .asc file of forest
    #firebreaks: list of firebreaks
    #output_path = path in which will be stored dpv.asc file

    header_file,data = read_asc(fuels)
    
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

def select_dpv(file_path,ratio):
    with open(file_path, 'r') as file:
        # Leer las primeras seis líneas y almacenarlas
        header = [next(file) for _ in range(6)]
        # Leer los datos numéricos
        data = np.loadtxt(file, dtype=np.float32)

    xdim,ydim = np.shape(data)
    nodos = int(xdim*ydim)
    node_list = list(range(1,nodos+1))
    dpv = dict.fromkeys(node_list,0)

    nodo = 1
    for i in range(0,len(data)):
        for j in range(0,len(data[i])):
            dpv[nodo] = data[i][j]
            nodo = nodo+1
            

    n_fb = int(ratio*nodos)
    selection = nlargest(n_fb, dpv.items(), key=lambda i: i[1])

    firebreaks = []
    for n in selection:
        firebreaks.append(n[0])

    return firebreaks

def ponderacion_dpv(dic,output_path):
    
    data_box = []
    for key in dic:
        header, data = read_asc(key)
        data_box.append(data)

    for i in range(0,len(data)):
        for j in range(0,len(data[i])):
            valor_ponderado = 0
            k = 0
            for key in dic:
                k +=1
                valor_ponderado += dic[key]*data_box[k][i][j]
            data[i][j] = valor_ponderado

    write_asc(output_path,header,data)

