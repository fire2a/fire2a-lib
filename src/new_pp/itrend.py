import subprocess as subp
import random as rd
import sys
import os
import rasterio
import numpy as np
import pandas as pd
from sim_temp import simulate_season_hyperparameters

sys.path.append('/home/matias/Documents/source/tools/post-processing/')
#sys.path.append('/Users/matiasvilches/Documents/F2A/source/tools/post-processing/')
from operations_raster import read_asc,average_asc_files
from operations_raster import write_asc
from operations_msg import get_graph
from calculator_bp import bp_calculation


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
        '--input-instance-folder', fuels_path, #Input folder
        '--output-folder', output_path, #Output folder
        '--nthreads', str(n_threads), #Number of threads
        '--nsims', str(nsims),  #Number of simulations
        '--sim',sim, #Fuel model
        '--seed',str(seed), #Random seed
        '--output-messages',
        '--weather random',
        '--nweathers 30',
        '--statistics'  # Enable statistics output
    ])
    subp.call(command, shell=True, stdout=subp.DEVNULL)  # Run the C2F simulation command

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

if __name__ == '__main__':
    
    
    ruta_base = '/home/matias/Documents'
    
    temporadas = 10
    forest = 'Biobio'
    forest_path = f'{ruta_base}/ITREND/forest/{forest}/fuels.asc'
    original_fuels = f'{ruta_base}/ITREND/forest/{forest}/fuels.asc'
    fuel_wip_path = f'{ruta_base}/ITREND/results/{forest}_wip/fuels.asc'
    total_incendios = []

    #"""
    
    for t in range(1, temporadas + 1):
        print(10*'-',f'Starting season {t}',10*'-')
        forest = 'Biobio'
        scar_sizes = []
        stop = False
        n_simulacion = 1
        forest_path = f'{ruta_base}/ITREND/forest/{forest}/'
        path = f'{ruta_base}/ITREND/data/BD_Incendios.csv'

        # Reset the working fuel map at the start of each season
        recover_fuels(original_fuels, fuel_wip_path)
        
        incendios, area = simulate_season_hyperparameters(path)
        #area = area*8
        #print(f'Incendios: {incendios}, Area: {area}')
        
        while stop == False:
            
            # 1. Simular incendios individuales
            output_path = f'{ruta_base}/ITREND/results/{forest}/t{t}/sim{n_simulacion}'
            #print(output_path)
            create_dir(output_path)
            n_threads = 1
            nsims = 1
            sim = 'K'
            seed = rd.randint(1,999)
            c2f_simulation(forest_path, output_path, n_threads, nsims, sim, seed, ruta_base)

            # 2. Verificar criterios de parada
            message_graph = get_graph(f'{output_path}/Messages/MessagesFile1.csv')
            burned_cells = list(message_graph.nodes())
            scar_size = len(burned_cells)
            #print("scar size: ",scar_size)
            if scar_size < 5:
                continue
            scar_sizes.append(scar_size)
            season_size = sum(scar_sizes)
            if season_size >= area:
                #print(f'Season size: {season_size}, Area: {area}')
                total_incendios.append(n_simulacion)
                print(f'Season size: {season_size}, Area: {area}, Total Incendios: {total_incendios}')
                stop = True
                continue
            else:
                print(f'Season size: {season_size}, Area: {area}')
                #pass

            # 3. Actualizar mapa de combustibles con la superficie quemada
            update_fuelmap(fuel_wip_path, burned_cells)
            n_simulacion +=1
            #if n_simulacion == incendios_temporada:
                #recover_fuels(original_fuels,fuel_wip_path)
                #stop = True
     #"""       
    # 4. Calculate excedence ratio maps
    # for every season and simulation, calculate the times a cell exceded 3 meters of flame lenght
    #total_incendios = [20,14,2,26,1,10,31,16,15,22]
    season_excedence = {}
    season_messages = {}
    for t in range(1, temporadas + 1):
        cells_excedence = []
        for n_simulacion in range(1, total_incendios[t-1] + 1):
            output_path = f'{ruta_base}/ITREND/results/{forest}/t{t}/sim{n_simulacion}'
            flame_length_map = f'{output_path}/Statistics/statisticsPerSim.csv'
            flame_length = pd.read_csv(flame_length_map, sep=',')
            flame_length = flame_length.loc[0,'surfaceFlameLengthMean']
            print(f'Flame length for simulation {n_simulacion} in season {t}: {flame_length}')

            if flame_length < 3:
                continue
            else:
                message_file = f'{output_path}/Messages/MessagesFile1.csv'
                message_graph = get_graph(message_file)
                season_messages[t] = season_messages.get(t, []) + [message_file]
                burned_cells = list(message_graph.nodes())
                #for i in burned_cells:
                #    cells_excedence[i] = (cells_excedence.get(i, 0) + 1)/temporadas#/ total_incendios[t-1]
                cell_excedence.extend(burned_cells)
        season_excedence[t] = cells_excedence

    #print(f'season messages: {season_messages}')

    # 5. Calculate Burn probability map
    # bp_calculation function: (fuels,files_list,pickle_path,output_path,nsims,ncores)
    """
    ncores = 6
    bp_output = f"{ruta_base}/ITREND/results/{forest}/BurnProbability"
    create_dir(bp_output)
    for t in range(1, temporadas + 1):
        bp_file_output = f'{bp_output}/bp_t{t}.asc'
        pickle_path = f'{ruta_base}/ITREND/results/{forest}/t{t}/Pickles/'
        try:
            bp_calculation(original_fuels, season_messages[t], pickle_path, bp_file_output, len(season_messages[t]), ncores)
        except KeyError as e:
            print(f"KeyError: {e}. No messages found for season {t}. Skipping Burn Probability calculation.")
            continue
    """

    # 6. Estimate the excedence ratio map using burn probability
    # if there is no burn probability map for specific season, skip it
    
    with rasterio.open(original_fuels) as src:
            profile = src.profile
            data = src.read(1)
            nrows, ncols = data.shape
            
    for i in range(nrows):
            for j in range(ncols):
                data[i, j] = 0
    
    for t in range(1, temporadas + 1):
        excedence_by_cell = season_excedence[t]

        for node in excedence_by_cell:
                    # transform coordinates (x,y) to cell id, considerate 1-based indexing
                    i, j = np.unravel_index(node-1, (nrows, ncols))
                    data[i, j] += 1/temporadas
    
    #write the excedence ratio map
    excedence_folder_output = f"{ruta_base}/ITREND/results/{forest}/excedance_maps/"
    create_dir(excedence_folder_output)
    excedence_ratio_output = f"{excedence_folder_output}excedance_map.asc"
    with rasterio.open(excedence_ratio_output, 'w', **profile) as dst:
        dst.write(data, 1)

    print("Excedance ratio map calculated")

    #excedence_folder_output = f"{ruta_base}/ITREND/results/{forest}/excedance_maps/"
    #input = excedence_folder_output
    #output = f"{excedence_folder_output}average_excedance.asc"
    #average_asc_files(input, output)
    