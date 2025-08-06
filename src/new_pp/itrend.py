import subprocess as subp
import random as rd
import os
import rasterio
import numpy as np
import pandas as pd
import multiprocessing
import shutil
import networkx as nx
import itertools
from collections import defaultdict
from sim_temp import simulate_season_hyperparameters
from multiprocessing import Pool
import traceback

# 1. OBTENER EL NUMERO DE INCENDIOS Y AREA TOTAL QUEMADA
# 2. SIMULAR UNA TEMPORADA
#   a. SIMULAR UN INCENDIO
#   b. ACTUALIZAR EL MAPA DE COMBUSTIBLES
#   c. SIMULAR DE NUEVO
# 3. SIMULAR N TEMPORADAS
# 4. CONTAR LAS VECES QUE EL LARGO DE LLAMA SUPERA UN UMBRAL
# 5. LA TASA DE EXCEDENCIA PASA SER

def get_graph(msg_path):
    H = nx.read_edgelist(path = msg_path,
                            delimiter=',',
                            create_using = nx.DiGraph(),
                            nodetype = int,
                            data = [('time', float), ('ros', float)])

    return H

def create_dir(directory):
    """
    Creates a directory if it does not exist.
    
    Parameters:
        directory (str): Path to the directory to be created.
    """
    
    if not os.path.exists(directory):
        os.makedirs(directory)  

def c2f_simulation(
    fuels_path,
    output_path,
    n_threads,
    nsims,
    sim,
    seed,
    ruta_base
):
    """
    Simulates wildfires using the C2F model and processes the results.
    
    Parameters:
        fuels_path (str): Path to the fuels raster file.
        output_path (str): Path to save the simulation results.
        n_threads (int): Number of threads for parallel processing.
        burn_area_threshold (float): Threshold for burn area percentage to stop simulation.
        min_fire_size (float): Minimum fire size in hectares to consider a valid fire.
    """
    # Placeholder for actual C2F simulation command
    command = " ".join([
        f'{ruta_base}/source/C2F-W/Cell2Fire/Cell2Fire',  # Replace with actual command
        '--input-instance-folder', str(fuels_path), #Input folder
        '--output-folder', str(output_path), #Output folder
        #'--fuels', str(fuels), #specific fuel file
        '--nthreads', str(n_threads), #Number of threads
        '--nsims', str(nsims),  #Number of simulations
        '--sim',str(sim), #Fuel model
        '--seed',str(seed), #Random seed
        '--output-messages',
        '--weather random',
        '--nweathers 500',
        '--statistics'  # Enable statistics output
    ])
    subp.call(command, shell=True, stdout=subp.DEVNULL)  # Run the C2F simulation command
    
    return command

def update_fuelmap(
    fuels_path,
    burned_cells
):
    """
    Updates the fuel map with the burned area as non fuel.
    Parameters:
        fuels_path (str): Path to the original fuels raster file.
        burn_area_path (str): Path to the burned area raster file.
        output_path (str): Path to save the updated fuel map.
    """

    burned_cells = [i - 1 for i in burned_cells]
    with rasterio.open(fuels_path) as src:
        profile = src.profile.copy()
        data = src.read(1)  # Leer banda 1
        nrows, ncols = data.shape

        # Convertir índices planos a (fila, columna)
        rows, cols = np.unravel_index(burned_cells, (nrows, ncols))

        # Cambiar valores a 101
        data[rows, cols] = 101

        # Guardar nueva copia del raster
    with rasterio.open(fuels_path, 'w', **profile) as dst:
        dst.write(data, 1)

def recover_fuels(original_fuels, new_fuels):
    # Abrir el raster original
    with rasterio.open(original_fuels) as src:
        profile = src.profile
        data = src.read()  # Leer todas las bandas

    # Escribir la copia exacta
    with rasterio.open(new_fuels, 'w', **profile) as dst:
        dst.write(data)

def stop_criteria(
    burned_cells,
    scar_sizes
):
    """
    Placeholder for stop criteria logic.
    This function should implement the logic to determine when to stop the simulation.
    For example, it could check if the burn area percentage exceeds a threshold or if the number of fires exceeds a limit.
    """

    stop = False
    burned_cells = len(set(burned_cells))
    sizes = sum(1 for x in scar_sizes if x >= 5)
    # Implement your stop criteria logic here
    if burned_cells > 1000:  # Example condition, replace with actual logic
        stop = True
    elif sizes > 10:  # Example condition, replace with actual logic
        stop = True

    return stop

def run_season(temporadas,forest_path,fuel_wip_folder,database_path,ruta_base):
        
    total_incendios = []
    for t in range(1, temporadas + 1):
        n_threads = 1
        nsims = 1
        sim = 'K'
        
        original_fuels = f'{forest_path}fuels.asc'
        fuel_wip_path = f'{fuel_wip_folder}fuels.asc'

        print(10*'-',f'Starting season {t}',10*'-')
        forest = 'Biobio'
        scar_sizes = []
        stop = False
        n_simulacion = 1
        
        # Reset the working fuel map at the start of each season
        recover_fuels(original_fuels, fuel_wip_path)
        
        # Simulate the season to get hyperparameters
        incendios, area = simulate_season_hyperparameters(database_path)
        
        while stop == False:
            
            # 1. Simular incendios individuales
            seed = rd.randint(1,999)
            output_path = f'{ruta_base}/ITREND/results/{forest}/seasons/t{t}/sim{n_simulacion}'
            create_dir(output_path)
            c2f_simulation(fuel_wip_folder, output_path, 'fuels.asc', n_threads, nsims, sim, seed, ruta_base)

            # 2. Verificar criterios de parada
            message_graph = get_graph(f'{output_path}/Messages/MessagesFile1.csv')
            burned_cells = list(message_graph.nodes())
            scar_size = len(burned_cells)

            if scar_size < 5:
                continue

            scar_sizes.append(scar_size)
            season_size = sum(scar_sizes)

            if season_size >= area:
                total_incendios.append(n_simulacion)
                print(f'Season size: {season_size}, Area: {area}, Total Incendios: {total_incendios}')
                stop = True
                print(10*'_',f'Finalizing season {t}',10*'_')
                continue
            else:
                print(f'Season size: {season_size}, Area: {area}')
                pass

            # 3. Actualizar mapa de combustibles con la superficie quemada
            update_fuelmap(fuel_wip_path, burned_cells)
            n_simulacion +=1

    # Reset the working fuel map at the start of each season
    recover_fuels(original_fuels, fuel_wip_path)
    return total_incendios

def calculate_excedance(umbral,seasons_folder,original_fuels,excedance_folder_output):

    excedance_ratio_output = f"{excedance_folder_output}excedance_map.asc"

    # este diccionario me guarda las veces que un 
    # nodo se quema y las veces que sobrepasa el umbral
    excedance = defaultdict(lambda: [0, 0])

    for t_folder in os.listdir(seasons_folder):
        t_path = os.path.join(seasons_folder, t_folder)
        if os.path.isdir(t_path) and t_folder.startswith("t"):
            
            for sim_folder in os.listdir(t_path):
                    sim_path = os.path.join(t_path, sim_folder)
                    if os.path.isdir(sim_path) and sim_folder.startswith("sim"):
                    
                        msg_file_path = os.path.join(sim_path, "Messages", "MessagesFile1.csv")
                        message_graph = get_graph(msg_file_path)
                        burned_cells = list(message_graph.nodes())
                        
                        statistics_file_path = os.path.join(sim_path,"Statistics","statisticsPerSim.csv")
                        flame_length = pd.read_csv(statistics_file_path, sep=',')
                        flame_length = flame_length.loc[0,'surfaceFlameLengthMean']

                        for cell in burned_cells:
                            if flame_length > umbral:
                                excedance[cell][0] += 1
                                excedance[cell][1] += 1
                            else:
                                excedance[cell][1] += 1


    with rasterio.open(original_fuels) as src:
            profile = src.profile
            data = src.read(1)
            nrows, ncols = data.shape
            
    for i in range(nrows):
            for j in range(ncols):
                cell = np.ravel_multi_index((i, j), dims=(nrows, ncols)) + 1
                cell_n = excedance[cell][0]
                cell_m = excedance[cell][1]
                try:
                    data[i, j] =  float(cell_n / cell_m)
                except:
                    data[i, j] = 0

    #write the excedence ratio map
    create_dir(excedance_folder_output)
    profile.update(dtype='float32', count=1)
    with rasterio.open(excedance_ratio_output, 'w', **profile) as dst:
        dst.write(data, 1)
    print("Excedance ratio map calculated")

def prepare_fuel_pool(original_fuels_path, wip_folder, n_cores):
    """Crea copias de trabajo del archivo de combustibles.
    
    Args:
        original_fuels_path: Ruta al archivo original de combustibles
        wip_folder: Directorio donde crear las copias
        n_cores: Número de copias a crear
        
    Returns:
        Lista con rutas a los archivos creados
    """
    os.makedirs(wip_folder, exist_ok=True)
    return [
        shutil.copy2(original_fuels_path, os.path.join(wip_folder, f'fuel_wip_{i}.asc'))
        for i in range(n_cores)
    ]


def run_season_mt(args):
    t, forest, wip_folder, ruta_base, database_path,temporal_wip = args

    base_season_path = os.path.join(ruta_base, 'ITREND/results', forest, f'seasons/t{t}')
    #os.makedirs(wip_folder, exist_ok=True)
    wip_folder_t = f'{temporal_wip}/t{t}'
    fuel_file = os.path.join(wip_folder_t, 'fuels.asc')
    #shutil.copy(original_fuels, fuel_file)
    shutil.copytree(wip_folder, wip_folder_t, dirs_exist_ok=True)
    #print(f"{wip_folder_t} created")

    try:
        #print(f'Season {t} started | Fuel file: {os.path.basename(fuel_file)}')
        scar_sizes = []
        area = simulate_season_hyperparameters(database_path)[1]
        n_simulacion = 1

        while True:  # Bucle de incendios (no de temporadas)
            seed = rd.randint(1, 999)
            output_dir = os.path.join(base_season_path, f'sim{n_simulacion}')
            #print(f'Preparing output directory: {output_dir}')
            os.makedirs(output_dir, exist_ok=True)

            command = c2f_simulation(wip_folder_t, output_dir, 1, 1, "K", seed, ruta_base)
            #print(f'Running command: {command}',"\n")

            burned_cells = list(get_graph(f'{output_dir}/Messages/MessagesFile1.csv').nodes())
            scar_size = len(burned_cells)

            if scar_size < 5:
                continue
                #print(f"Fire {n_simulacion} in t{t} skipped (size {scar_size})")
            else:
                scar_sizes.append(scar_size)
                current_total = sum(scar_sizes)

                if current_total >= area * 0.95:
                    print(f'Season {t} completed | Area: {current_total:.2f}/{area:.2f}')
                    finish = True
                    break  # Sale del bucle de incendios (temporada completada)
                    
                update_fuelmap(fuel_file, burned_cells)
                #print(f'Season {t} progress: {current_total:.2f}/{area:.2f}')

            n_simulacion += 1

    except Exception as fire_error:
        print(f'Error in fire {n_simulacion} of Season {t} | Fuel file: {os.path.basename(fuel_file)}')
        #print(traceback.format_exc())
        if os.path.exists(base_season_path):
            pass
            #shutil.rmtree(base_season_path)
        raise  # Opcional: relanza el error si quieres manejar el fallo fuera de la función.

    finally:
        if os.path.exists(wip_folder_t):
            os.remove(wip_folder_t)

def run_all_seasons_parallel(temporadas, n_cores, forest, temporal_wip, wip_folder, ruta_base, database_path):
    """Controlador principal"""
    #os.makedirs(wip_folder, exist_ok=True)
    
    # Preparar argumentos (ya no necesitamos fuel_files)
    tasks = [
        (t, forest, wip_folder, ruta_base, database_path,temporal_wip)
        for t in range(1, temporadas + 1)
    ]
    
    with Pool(n_cores) as pool:
        pool.map(run_season_mt, tasks)


if __name__ == '__main__':
    
    ruta_base = '/Users/matiasvilches/Documents/F2A'
    forest = 'Biobio'

    forest_path = f'{ruta_base}/ITREND/forest/{forest}/'
    original_fuels = f'{forest_path}fuels.asc'
    excedance_folder_output = f"{ruta_base}/ITREND/results/{forest}/excedance_maps/"
    fuel_wip_folder = f'{ruta_base}/ITREND/results/{forest}/{forest}_wip/'
    database_path = f'{ruta_base}/ITREND/data/BD_Incendios.csv'
    season_folder = f'{ruta_base}/ITREND/results/{forest}/seasons/'
    temporal_wip = f'{ruta_base}/ITREND/results/{forest}/wips/'
    
    temporadas = 20
    umbral_excedencia = 3
    n_cores = 5
    
    #run_season(temporadas,forest_path,fuel_wip_folder,database_path,ruta_base)
    run_all_seasons_parallel(temporadas,n_cores,forest,temporal_wip,fuel_wip_folder,ruta_base,database_path)
    #calculate_excedance(umbral_excedencia,season_folder,original_fuels,excedance_folder_output)