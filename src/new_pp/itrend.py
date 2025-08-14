import subprocess as subp
import random as rd
import os
import rasterio
import numpy as np
import pandas as pd
import shutil
import networkx as nx
from collections import defaultdict
from sim_temp import simulate_season_hyperparameters
from multiprocessing import Pool
import time

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
        '--statistics',  # Enable statistics output,
        '--cros'
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

def calculate_excedance(seasons_folder,original_fuels,umbrales):

    #excedance_ratio_output = f"{excedance_folder_output}excedance_map.asc"

    # este diccionario me guarda las veces que un 
    # nodo se quema y las veces que sobrepasa un cierto umbral
    # nodo[umbral[ veces que se sobrepasa, veces que se quema]]
    excedance = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    temporadas = len(season_folder)

    for t_folder in os.listdir(seasons_folder):
        t_path = os.path.join(seasons_folder, t_folder)
        if os.path.isdir(t_path) and t_folder.startswith("t"):
            
            for sim_folder in os.listdir(t_path):
                    sim_path = os.path.join(t_path, sim_folder)
                    if os.path.isdir(sim_path) and sim_folder.startswith("sim"):
                        
                        try:
                            msg_file_path = os.path.join(sim_path, "Messages", "MessagesFile1.csv")
                            message_graph = get_graph(msg_file_path)
                            burned_cells = list(message_graph.nodes())
                            
                            statistics_file_path = os.path.join(sim_path,"Statistics","statisticsPerSim.csv")
                            flame_length = pd.read_csv(statistics_file_path, sep=',')
                            flame_length = flame_length.loc[0,'crownFlameLengthMean']
                            #print(flame_length)

                            for cell in burned_cells:
                                for umbral in umbrales:
                                    if flame_length > umbral:
                                        excedance[cell][umbral][0] += 1
                                        excedance[cell][umbral][1] += 1
                                    else:
                                        excedance[cell][umbral][1] += 1
                                    
                        except FileNotFoundError:
                            print(f"File not found: {msg_file_path} or {statistics_file_path}")
                            continue


    with rasterio.open(original_fuels) as src:
            profile = src.profile
            data = src.read(1)
            nrows, ncols = data.shape
            
    
    for umbral in umbrales:
        for i in range(nrows):
                for j in range(ncols):
                    cell = np.ravel_multi_index((i, j), dims=(nrows, ncols)) + 1
                    cell_n = excedance[cell][umbral][0]
                    cell_m = excedance[cell][umbral][1]
                    try:
                        # veces que pasa el umbral / veces que se quema
                        data[i, j] =  float(cell_n / cell_m)
                        # veces que pasa el umbral / temporadas
                        # descomentar si se quiere usar esta métrica (tiene en cuenta la prob de quema)
                        #data[i, j] = float(cell_n / temporadas)

                    except:
                        data[i, j] = 0

        #write the excedence ratio map
        create_dir(excedance_folder_output)
        profile.update(dtype='float32', count=1)
        excedance_ratio_output = os.path.join(excedance_folder_output, f'excedance_map_{umbral}m.asc')
        with rasterio.open(excedance_ratio_output, 'w', **profile) as dst:
            dst.write(data, 1)
        print("Excedance ratio map calculated")

def run_season_mt(args):
    t, forest, forest_path, ruta_base, database_path,temporal_wip,pond = args

    base_season_path = os.path.join(ruta_base, 'ITREND/results', forest, f'seasons/t{t}')
    #os.makedirs(wip_folder, exist_ok=True)
    wip_folder_t = f'{temporal_wip}/t{t}'
    fuel_file = os.path.join(wip_folder_t, 'fuels.asc')
    #shutil.copy(original_fuels, fuel_file)
    shutil.copytree(forest_path, wip_folder_t, dirs_exist_ok=True)
    #print(f'Working folder for season {t} created at: {wip_folder_t}')
    #print(f"{wip_folder_t} created")

    try:
        #print(f'Season {t} started | Fuel file: {os.path.basename(fuel_file)}')
        scar_sizes = []
        area = simulate_season_hyperparameters(database_path)[1]
        n_simulacion = 1
 
        while True:  # Bucle de incendios (no de temporadas)
            seed = rd.randint(1, 999)
            #seed = 123
            output_dir = os.path.join(base_season_path, f'sim{n_simulacion}')
            #print(f'Preparing output directory: {output_dir}')
            os.makedirs(output_dir, exist_ok=True)

            #print(f'Running fire simulation {n_simulacion} for season {t}')
            command = c2f_simulation(wip_folder_t, output_dir, 1, 1, "K", seed, ruta_base)
            
            burned_cells = list(get_graph(f'{output_dir}/Messages/MessagesFile1.csv').nodes())
            conversor_tamaño = 1
            scar_size = len(burned_cells) / conversor_tamaño
            #area = 1000
            if scar_size < 5:
                continue
                #print(f"Fire {n_simulacion} in t{t} skipped (size {scar_size})")
            else:
                scar_sizes.append(scar_size)
                current_total = sum(scar_sizes)

                if current_total >= area * 0.98 * pond:
                    #print(f'Season {t} completed | Area: {current_total:.2f}/{area:.2f}')
                    finish = True
                    break  # Sale del bucle de incendios (temporada completada)
                    
                update_fuelmap(fuel_file, burned_cells)
                #print(f'Season {t} progress: {current_total:.2f}/{area:.2f}')

            n_simulacion += 1

        # remove the working folder after the season is done
        if os.path.exists(wip_folder_t):
            shutil.rmtree(wip_folder_t)

    except Exception as fire_error:
        print(f'Error in fire {n_simulacion} of Season {t} | Fuel file: {os.path.basename(fuel_file)}')
        #print(traceback.format_exc())
        if os.path.exists(base_season_path):
            pass
            #shutil.rmtree(base_season_path)
        raise  # Opcional: relanza el error si quieres manejar el fallo fuera de la función.

def run_all_seasons_parallel(temporadas, n_cores, forest, temporal_wip, forest_path, ruta_base, database_path,pond):
    """Controlador principal"""
    #os.makedirs(wip_folder, exist_ok=True)
    
    # Preparar argumentos (ya no necesitamos fuel_files)
    tasks = [
        (t, forest, forest_path, ruta_base, database_path,temporal_wip,pond)
        for t in range(1, temporadas + 1)
    ]
    
    with Pool(n_cores) as pool:
        pool.map(run_season_mt, tasks)


if __name__ == '__main__':
    
    # SIMULATION BLOCK
    for zone in ["sur"]:

        pond = 0
        if zone == 'norte':
            pond = 0.47
        elif zone == 'sur':
            pond = 0.53
        
        # calculate proccesing time
        start_time = time.time()
        
        # Define paths and parameters
        ruta_base = '/Users/matiasvilches/Documents/F2A'
        #ruta_base = '/home/matias/Documents/'
        forest = f'Biobio_{zone}_100'
        forest = "Biobio"

        forest_path = f'{ruta_base}/ITREND/forest/{forest}/'
        database_path = f'{ruta_base}/ITREND/data/BD_Incendios.csv'
        temporal_wip = f'{ruta_base}/ITREND/results/{forest}/wips'
        
        temporadas = 3
        n_cores = 1
        
        # run all seasons in parallel
        #run_all_seasons_parallel(temporadas,n_cores,forest,temporal_wip,forest_path,ruta_base,database_path,pond)
        
        # calculate processing time in hours
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Processing time for {n_cores} cores: {elapsed_time / 3600:.2f} hours = {elapsed_time / 60:.2f} minutes")

    # EXCEEDANCE MAPS BLOCK
    umbrales = [3,4,5,6]
    for zone in ["sur"]:
        forest = f'Biobio_{zone}_100'
        forest = "Biobio"
        season_folder = f'{ruta_base}/ITREND/results/{forest}/seasons/'
        original_fuels = f'{forest_path}fuels.asc'
        excedance_folder_output = f"{ruta_base}/ITREND/results/{forest}/excedance_maps/"
        calculate_excedance(season_folder,original_fuels,umbrales)
