import os
import numpy as np
import rasterio

def read_asc(file_path):
    with open(file_path, 'r') as file:
        # Leer las primeras seis líneas y almacenarlas
        header = [next(file) for _ in range(6)]
        # Leer los datos numéricos
        data = np.loadtxt(file, dtype=np.float32)
    return header, data

def write_asc(file_path, header, array):
    with open(file_path, 'w') as file:
        # Escribir el encabezado
        file.writelines(header)
        # Escribir los datos numéricos
        np.savetxt(file, array, fmt='%.6f')

def write_asc_from_dict(fuels,dic,output_path):

    header_file, data = read_asc(fuels)
    
    header = {}
    for line in header_file:
        key, value = line.strip().split(maxsplit=1)
        header[key] = float(value) if '.' in value or 'e' in value.lower() else str(value)

    nrows = int(header['nrows'])
    ncols = int(header['ncols'])

    data = np.zeros((nrows, ncols))
    
    for n in dic.keys():

        row = (n-1) // ncols
        col = n % ncols

        data[row][col-1] = dic[n]

    with open(output_path, 'w') as file:
        # Escribir el encabezado
        file.writelines(header_file)
        # Escribir los datos numéricos
        np.savetxt(file, data, fmt='%.6f')

def average_asc_files(input_folder, output_file):
    files = [f for f in os.listdir(input_folder) if f.endswith('.asc')]
    if not files:
        raise FileNotFoundError("No .asc files found in the specified folder.")
    
    # Leer el primer archivo para obtener el encabezado y la dimensión de los datos
    first_file = os.path.join(input_folder, files[0])
    header, data = read_asc(first_file)
    
    sum_array = np.zeros_like(data)
    count = 0
    
    # Acumular los datos de todos los archivos
    for file in files:

        print(count)

        file_path = os.path.join(input_folder, file)
        _, data = read_asc(file_path)

        minimo = data.min()

        if minimo < 0:
            for i in range(0,len(data)):
                for j in range(0,len(data[i])):
                    if data[i][j] < 0:
                        data[i][j] = 0
        
        sum_array += data

        count += 1        
    
    # Calcular el promedio
    avg_array = sum_array / count
    
    # Escribir el archivo de salida
    write_asc(output_file, header, avg_array)

def sum_raster_values(raster_path):
    with rasterio.open(raster_path) as src:
        data = src.read(1)  # Read the first band (assuming single-band raster)
        total_sum = np.nansum(data)  # Sum all values, ignoring NaNs (if any)
    return float(total_sum)