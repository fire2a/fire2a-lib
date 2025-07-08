from operations_raster import read_asc
from operations_raster import write_asc
import subprocess as subp
from operations_msg import get_graph
import random as rd

import os

from sim_temp import simulate_season_hyperparameters

"""
# 1. SIMULAR INCENDIOS INDIVIDUALES
# 2. ACTUALIZAR MAPA DE COMBUSTIBLES CON LA SUPERFICIE QUEMADA
# 3. SE SIMULA HASTA CUMPLIR ALGUN CRITERIO DE PARADA (PORCENTAJE DE AREA QUEMADA, NUMERO DE INCENDIOS SUPERIORES A 5HA)
# 4. SE OBTIENEN LOS MAPAS DE FLAME LENGTH, FIRELINE INTENSITY, ETC. DE CADA INCENDIO
# 5. SE SIMULAN LAS TEMPORADAS NECESARIAS
# 6. SE PROMEDIAN LOS RESULTADOS PARA TODAS LAS TEMPORADAS
# 7. SE OBTIENEN LOS MAPAS DE RIESGO DE INCENDIO
# 8. SE OBTIENEN LOS MAPAS DE RIESGO DE INCENDIO CON LA SUPERFICIE QUEMADA PARA LAS CELDAS QUE SUPEREN EL UMBRAL DE RIESGO
"""

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
    seed
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
        '/Users/matiasvilches/Documents/F2A/source/C2F-W/Cell2Fire/Cell2Fire',  # Replace with actual command
        '--input-instance-folder', fuels_path, #Input folder
        '--output-folder', output_path, #Output folder
        '--nthreads', str(n_threads), #Number of threads
        '--nsims', str(nsims),  #Number of simulations
        '--sim',sim, #Fuel model
        '--seed',str(seed), #Random seed
        '--output-messages',
        '--weather random',
        '--nweathers 129'
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

    header, data, nodos = read_asc(fuels_path)
    
    for node in burned_cells:
        row = (node-1) // 20
        col = node % 20
        data[row][col-1] = 101

    write_asc(fuels_path, header, data)

def recover_fuels(original_fuels, new_fuels):
    header,data,nodos = read_asc(original_fuels)
    write_asc(new_fuels,header,data)

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
    
    temporadas = 2
    incendios_temporada = 1
    forest_path = '/Users/matiasvilches/Documents/F2A/ITREND/sub20/'

    for t in range(1, temporadas + 1):
        scar_sizes = []
        stop = False
        n_simulacion = 0
        original_fuels = '/Users/matiasvilches/Documents/F2A/ITREND/sub20/fuels.asc'
        fuel_wip_path = '/Users/matiasvilches/Documents/F2A/ITREND/results/sub20_wip/fuels.asc'
        forest_path = '/Users/matiasvilches/Documents/F2A/ITREND/sub20/'
        path = '/Users/matiasvilches/Documents/F2A/ITREND/data/BD_Incendios.csv'

        # Reset the working fuel map at the start of each season
        recover_fuels(original_fuels, fuel_wip_path)
        
        incendios, area = simulate_season_hyperparameters(path)
        print(f'Incendios: {incendios}, Area: {area}')
        incendios, area = simulate_season_hyperparameters(path)
        
        while stop == False:
            
            # 1. Simular incendios individuales
            output_path = f'/Users/matiasvilches/Documents/F2A/ITREND/sub20/results/t{t}/sim{n_simulacion}'
            create_dir(output_path)
            n_threads = 1
            nsims = 1
            sim = 'C'
            seed = rd.randint(1,999)
            #seed = 2
            c2f_simulation(forest_path, output_path, n_threads, nsims, sim, seed)

            # 2. Verificar criterios de parada
            message_graph = get_graph(f'{output_path}/Messages/MessagesFile1.csv')
            burned_cells = list(message_graph.nodes())
            scar_size = len(burned_cells)
            print("scar size: ",scar_size)
            if scar_size < 5:
                continue
            scar_sizes.append(scar_size)
            season_size = sum(scar_sizes)
            if season_size < area:
                print(f'Season size: {season_size}, Area: {area}')
                stop = True
                continue

            # 3. Actualizar mapa de combustibles con la superficie quemada
            update_fuelmap(fuel_wip_path, burned_cells)
            n_simulacion +=1
            forest_path = '/Users/matiasvilches/Documents/F2A/ITREND/results/sub20_wip/'
            #if n_simulacion == incendios_temporada:
                #recover_fuels(original_fuels,fuel_wip_path)
                #stop = True