#%% imports
import os
import numpy as np
import networkx as nx
import pandas as pd
import gurobipy as gp
import re
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import rasterio
import subprocess as subp
from statistics import mean
from concurrent.futures import ProcessPoolExecutor
from gurobipy import GRB
from operations_raster import read_csv,raster_to_dict,sum_raster_values

def get_top(dictionary,k):
    sorted_items = sorted(dictionary.items(), key=lambda x: x[1], reverse=True)
    top_k_items = sorted_items[:k]
    top_k_keys = [item[0] for item in top_k_items]
    return top_k_keys

def model_opt(alpha,dpv,fuel_emissions):
    
    header_file,data,nodos = read_asc(fuel_emissions)

    header = {}
    for line in header_file:
        key, value = line.strip().split(maxsplit=1)
        header[key] = float(value) if '.' in value or 'e' in value.lower() else str(value)

    ncols = int(header['ncols'])
    nrows = int(header['nrows'])
    availset = []

    for n in range(1,nodos+1):
        row = (n-1) // ncols  # Calculate row index
        col = n % ncols   # Calculate column index
        if data[row][col-1] > 0:
            availset.append(n)

    cortafuegos = int(len(availset)*alpha)
    model = gp.Model()
    model.setParam("OutputFlag", 0)
    
    dpv_dic = raster_to_dict(dpv)
    fuel_load_dic = raster_to_dict(fuel_emissions)

    x = model.addVars(availset, vtype=GRB.BINARY)
    z = gp.quicksum((dpv_dic[n]-fuel_load_dic[n])*x[n] for n in availset)
    obj = model.setObjective(z, GRB.MAXIMIZE)
    
    c_fblimit = model.addConstr(gp.quicksum(x[n] for n in availset) == cortafuegos)
    model.optimize()
    
    fb_list = []
    for n in availset:
        if x[n].x >= 0.9:
            fb_list.append(n)

    return fb_list
    


def percentage_formatter(x, pos):
    return f"{x*100:.1f}%"

def sum_raster_values(raster_path):
    with rasterio.open(raster_path) as src:
        data = src.read(1)  # Read the first band (assuming single-band raster)
        total_sum = np.nansum(data)  # Sum all values, ignoring NaNs (if any)
    return float(total_sum)

def sum_data_values(data):
    total_sum = np.nansum(data)  # Sum all values, ignoring NaNs (if any)
    return float(total_sum)

def plot(confidence,intensities,means):
    confidence_low = [i[0] for i in confidence]
    confidence_up = [i[1] for i in confidence]

    fig, ax = plt.subplots()
    x = intensities

    ax.plot(x, means,linestyle='dashed')

    for xi, yi in zip(x, means):
        plt.scatter(xi, yi, color = "#1a1a1a",s=30, zorder=5)

    ax.fill_between(x, 
                    confidence_low, 
                    confidence_up, 
                    color='b', 
                    alpha=.15)

    #ax.set_ylim(50000,260000)

    ax.set_title('CO2E Emissions')
    ax.xaxis.set_major_formatter(mtick.FuncFormatter(percentage_formatter))

def remove_lowest_30_percent(values):
    if not values:
        return [], []
    
    # Sort values while keeping track of original indices
    sorted_indices = sorted(range(len(values)), key=lambda i: values[i])
    
    # Determine how many elements to remove (30% of total length)
    num_to_remove = int(len(values) * 0.3)
    
    # Get indices of removed elements
    removed_indices = sorted_indices[:num_to_remove]
    
    # Get the filtered list by keeping elements not in removed_indices
    filtered_values = [values[i] for i in sorted_indices[num_to_remove:]]
    
    return filtered_values, removed_indices

def remove_indices_from_list(values, indices_to_remove):
    return [value for i, value in enumerate(values) if i not in indices_to_remove]

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

def multiply_rasters(loads_raster_path, fraction_raster_path, output_raster_path):
    # 1. Read the fuel loads raster
    with rasterio.open(loads_raster_path) as src_load:
        load_data = src_load.read(1)  # Read band 1
        profile = src_load.profile.copy()  # Copy metadata

    # 2. Read the surface fraction raster
    with rasterio.open(fraction_raster_path) as src_fraction:
        fraction_data = src_fraction.read(1)

    # Optional: Check that both rasters have the same dimensions
    if load_data.shape != fraction_data.shape:
        raise ValueError("The two rasters must have the same shape/dimensions.")
    
    # 3. Multiply each pixel: fuel_load * surface_fraction
    multiplied_data = load_data * fraction_data
    
    # 4. Update the profile for ASCII Grid output
    #    - 'AAIGrid' driver creates an Arc/Info ASCII Grid
    #    - Use float32 for decimal values
    #    - Set nodata to something appropriate (here, 0)
    profile.update(
        driver="AAIGrid",
        dtype=rasterio.float32,
        nodata=0
    )
    
    # 5. Write the output ASCII
    with rasterio.open(output_raster_path, "w", **profile) as dst:
        dst.write(multiplied_data.astype(np.float32), 1)

    #print(f'Raster generado de emisiones generado')

def multiply_rasters_woo(loads_raster_path, fraction_raster_path):
    # 1. Read the fuel loads raster
    with rasterio.open(loads_raster_path) as src_load:
        load_data = src_load.read(1)  # Read band 1
        profile = src_load.profile.copy()  # Copy metadata

    # 2. Read the surface fraction raster
    with rasterio.open(fraction_raster_path) as src_fraction:
        fraction_data = src_fraction.read(1)

    # Optional: Check that both rasters have the same dimensions
    if load_data.shape != fraction_data.shape:
        raise ValueError("The two rasters must have the same shape/dimensions.")
    
    # 3. Multiply each pixel: fuel_load * surface_fraction
    multiplied_data = load_data * fraction_data
    
    return multiplied_data

def multiply_raster_data(raster1, data):
    # 1. Read the fuel loads raster
    with rasterio.open(raster1) as src_load:
        load_data = src_load.read(1)  # Read band 1
        profile = src_load.profile.copy()  # Copy metadata

    # Optional: Check that both rasters have the same dimensions
    if load_data.shape != data.shape:
        raise ValueError("The two rasters must have the same shape/dimensions.")
    
    # 3. Multiply each pixel: fuel_load * surface_fraction
    multiplied_data = load_data * data
    
    return multiplied_data

def c2f_validation(args,nsims):

    print("simulating")

    #INITIALIZATION
    forest,fuels,sfb_directory,dpv_path,fe_path,gf_path,fe_file_total,percentages = args
    
    #PARAMETERS
    input_folder = f'--input-instance-folder /home/matias/Documents/Emisiones/{forest}/forest' 
    output_folder = f'--output-folder /home/matias/Documents/Emisiones/{forest}/'
    nsims = f'--nsims {nsims}'
    nthreads = '--nthreads 30'
    weather_opt = '--nweathers 86 --weather random'
    seed = '--seed 333'
    extra = '--cros --ignitionsLog'
    outputs = '--out-sfb --output-messages --out-crown'
    ignitions = '--ignitions-random'

    #CELL2FIRE EXECUTION
    for perc in percentages:
        print(perc)
        #ROUTES
        results_path = f"/home/matias/Documents/Emisiones/{forest}/results"
        harvest_file_csv = f"{results_path}/harvest/harvest{perc}.csv"
        harvest_file_asc = f"{results_path}/harvest/harvest{perc}.asc"
        val_output_folder = f"{output_folder}/results/validation/{perc}/"
        harvest_emissions_asc = f"{results_path}/harvest/harvest_emissions{perc}.asc"
        emissions_path = f"{results_path}/validation/emissions/{perc}"

        #SOLVE OPTIMIZATION MODEL
        fb_sol = model_opt(perc,dpv_path,fe_file_total)

        #SAVE FIREBREAK SOLUTIONS
        header,data,nodos = read_asc(fuels)
        write_treatment(header,fb_sol,harvest_file_asc)
        harvest_csv = harvested(harvest_file_csv,fb_sol)
        
        #CALL C2F
        firebreaks_opt = f'--FirebreakCells {harvest_file_csv}'
        options = " ".join([input_folder,val_output_folder,nsims,nthreads,seed,extra,outputs,weather_opt,firebreaks_opt,ignitions])
        c2f_call = '/home/matias/source/C2F-W/Cell2Fire/Cell2Fire --sim C  '+options
        subp.call(c2f_call, shell=True, stdout=subp.DEVNULL)

def c2f_validation2(args,nsims):

    print("simulating")

    #INITIALIZATION
    forest,fuels,sfb_directory,dpv_path,fe_path,gf_path,fe_file_total,percentages = args
    
    #PARAMETERS
    input_folder = f'--input-instance-folder /home/matias/Documents/Emisiones/{forest}/forest' 
    output_folder = f'--output-folder /home/matias/Documents/Emisiones/{forest}/'
    nsims = f'--nsims {nsims}'
    nthreads = '--nthreads 30'
    weather_opt = '--nweathers 86 --weather replication'
    seed = '--seed 333'
    extra = '--cros --ignitionsLog --out-ros'
    outputs = '--out-sfb --output-messages --out-crown'
    ignitions = '--ignitions'

    #CELL2FIRE EXECUTION
    for perc in percentages:
        print(perc)
        #ROUTES
        results_path = f"/home/matias/Documents/Emisiones/{forest}/results"
        harvest_file_csv = f"{results_path}/harvest/harvest{perc}.csv"
        harvest_file_asc = f"{results_path}/harvest/harvest{perc}.asc"
        val_output_folder = f"{output_folder}/results/validation/{perc}/"
        harvest_emissions_asc = f"{results_path}/harvest/harvest_emissions{perc}.asc"
        emissions_path = f"{results_path}/validation/emissions/{perc}"

        #SOLVE OPTIMIZATION MODEL
        fb_sol = model_opt(perc,dpv_path,fe_file_total)

        #SAVE FIREBREAK SOLUTIONS
        header,data,nodos = read_asc(fuels)
        write_treatment(header,fb_sol,harvest_file_asc)
        harvest_csv = harvested(harvest_file_csv,fb_sol)
        
        #CALL C2F
        firebreaks_opt = f'--FirebreakCells {harvest_file_csv}'
        options = " ".join([input_folder,val_output_folder,nsims,nthreads,seed,extra,outputs,weather_opt,firebreaks_opt,ignitions])
        c2f_call = '/home/matias/source/C2F-W/Cell2Fire/Cell2Fire --sim C  '+options
        subp.call(c2f_call, shell=True)#, stdout=subp.DEVNULL)

def c2f_preset(forest,nsims):    
    #PARAMETERS
    input_folder = f'--input-instance-folder /home/matias/Documents/Emisiones/{forest}/forest' 
    output_folder = f'--output-folder /home/matias/Documents/Emisiones/{forest}/results/preset/'
    nsims_opt = f'--nsims {nsims}'
    nthreads = '--nthreads 30'
    weather_opt = '--nweathers 86 --weather random'
    seed = '--seed 999'
    extra = '--cros'
    outputs = '--out-sfb --output-messages --out-crown'

    #CALL C2F
    options = " ".join([input_folder,output_folder,nsims_opt,nthreads,seed,extra,outputs,weather_opt])
    c2f_call = '/home/matias/source/C2F-W/Cell2Fire/Cell2Fire --sim C --ignitions-random '+options
    subp.call(c2f_call, shell=True, stdout=subp.DEVNULL)

def process_results_uni(args):
    print("processing results")
    
    #INITIALIZATION
    forest,fuels,sfb_directory,dpv_path,fe_path,gf_path,fe_file_total,percentages = args
    mean_list = []
    sem_list = []
    confidence = []
    zero_emissions_list = []
    #RESULTS PROCESSING
    for perc in percentages:
        
        #INITIALIZATION
        print(perc)
        emissions = []
        results_path = f"/home/matias/Documents/Emisiones/{forest}/results"
        harvest_file_csv = f"{results_path}/harvest/harvest{perc}.csv"
        harvest_file_asc = f"{results_path}/harvest/harvest{perc}.asc"
        harvest_emissions_asc = f"{results_path}/harvest/harvest_emissions{perc}.asc"
        val_output_folder = f"{results_path}/validation/{perc}/"
        val_sfb_folder = f"{val_output_folder}SurfFractionBurn_selected/"
        val_crown_folder = f"{val_output_folder}CrownFire_selected/"
        crown_fuel_load = f"/home/matias/Documents/Emisiones/{forest}/forest/{forest}_fl_copa.asc"
        
        #GENERATE EMISSIONS RASTER DATA
        harvest_emissions_data = multiply_rasters_woo(fe_file_total,harvest_file_asc)
        harvest_emissions = sum_data_values(harvest_emissions_data)
        
        #CALCULATE AND SAVE TOTAL EMISSIONS
        for filename in os.listdir(val_sfb_folder):

            #multiply sfb (it is sfb x fl in C2F) x gfe raster
            sfb_file = val_sfb_folder+filename
            wildfire_emissions_data = multiply_rasters_woo(gf_path,sfb_file)
            
            #multiply burned crown x crown fuel load
            file_number = re.findall(r'\d+',filename)[0]
            crown_file = f'{val_crown_folder}Crown{file_number}.asc'
            crown_sfb_data = multiply_rasters_woo(crown_fuel_load,crown_file)
            wildfire_crown_emissions_data = multiply_raster_data(gf_path,crown_sfb_data)
            
            #sum all emissions
            wildfire_emissions = sum_data_values(wildfire_emissions_data)
            wildfire_crown_emissions = sum_data_values(wildfire_crown_emissions_data)
            total_emissions = wildfire_emissions + wildfire_crown_emissions + harvest_emissions
            
            if wildfire_emissions == 0 and perc == 0:
                zero_emissions_list.append(filename)
                continue
            else:
                emissions.append(total_emissions)

        
        #STATS GENERATING
        
        #calculate mean and sem
        mean_co2 = np.mean(emissions)
        sem_co2 = st.sem(emissions)
        
        #append to final results lists
        mean_list.append(mean_co2)
        sem_list.append (sem_co2)
        confidence.append(st.norm.interval(0.95, loc=mean_co2, scale=sem_co2))
    
    print(mean_list)
    return mean_list, confidence, zero_emissions_list

def process_single_percentage(perc, args, zero_filenames):
    """
    Process a single percentage level in parallel.
    """
    forest, fuels, sfb_directory, dpv_path, fe_path, gf_path, fe_file_total, _ = args
    print(f"Processing percentage: {perc}")

    emissions = []
    results_path = f"/home/matias/Documents/Emisiones/{forest}/results"
    harvest_file_asc = f"{results_path}/harvest/harvest{perc}.asc"
    val_output_folder = f"{results_path}/validation/{perc}/"
    val_sfb_folder = f"{val_output_folder}SurfFractionBurn_selected/"
    val_crown_folder = f"{val_output_folder}CrownFire_selected/"
    crown_fuel_load = f"/home/matias/Documents/Emisiones/{forest}/forest/{forest}_fl_copa.asc"

    # Generate emissions raster data
    harvest_emissions_data = multiply_rasters_woo(fe_file_total, harvest_file_asc)
    harvest_emissions = sum_data_values(harvest_emissions_data)
 
    # Process each wildfire simulation
    for filename in os.listdir(val_sfb_folder):
        
        if filename in zero_filenames:
            print(f"skiped file {filename}")
            continue
        
        sfb_file = os.path.join(val_sfb_folder, filename)
        wildfire_emissions_data = multiply_rasters_woo(gf_path, sfb_file)

        # Extract file number
        file_number = re.findall(r'\d+', filename)[0]
        crown_file = os.path.join(val_crown_folder, f'Crown{file_number}.asc')

        crown_sfb_data = multiply_rasters_woo(crown_fuel_load, crown_file)
        wildfire_crown_emissions_data = multiply_raster_data(gf_path, crown_sfb_data)

        # Sum emissions
        wildfire_emissions = sum_data_values(wildfire_emissions_data)
        wildfire_crown_emissions = sum_data_values(wildfire_crown_emissions_data)
        total_emissions = wildfire_emissions + wildfire_crown_emissions + harvest_emissions
        emissions.append(total_emissions)

    # Compute statistics
    mean_co2 = np.mean(emissions)
    sem_co2 = st.sem(emissions)
    confidence = st.norm.interval(0.95, loc=mean_co2, scale=sem_co2)

    return perc, mean_co2, sem_co2, confidence

def process_results(args,zero_emissions_list):
    """
    Parallelized version of process_results.
    """
    print("Processing results ------------------------------------------")

    # Extract arguments
    forest, fuels, sfb_directory, dpv_path, fe_path, gf_path, fe_file_total, percentages = args
    results_path = f"/home/matias/Documents/Emisiones/{forest}/results"

    # Parallel execution
    mean_list = []
    sem_list = []
    confidence_list = []

    with ProcessPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(process_single_percentage, perc, args,zero_emissions_list): perc for perc in percentages}

        for future in futures:
            perc, mean_co2, sem_co2, confidence = future.result()
            mean_list.append(mean_co2)
            sem_list.append(sem_co2)
            confidence_list.append(confidence)

    return mean_list, confidence_list

def plot_results(mean_list,confidence,percentages_list,forest):
    # Simulated data

    confidence_low = [i[0] for i in confidence]
    confidence_up = [i[1] for i in confidence]

    # Prettify settings
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18
    })

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot shaded error band
    ax.fill_between(percentages_list, confidence_low, confidence_up, color='gray', alpha=0.4)

    # Plot main line
    ax.plot(percentages_list,mean_list, 'o--', color='black', markersize=6, label='Emissions')

    # Y-axis label with LaTeX formatting
    ax.set_ylabel(r'CO2E emissions tons ($\varepsilon_\alpha$)')
    #ax.set_ylabel(r'hola')

    # X-axis label (optional)
    ax.set_xlabel(r'Percentage of firebreaks ($\alpha$)')

    # Optional: Add inset label "A"
    ax.text(-0.5, max(mean_list) + 1000, 'A', fontsize=20, fontweight='bold')

    # Customize ticks
    ax.tick_params(direction='in', length=6, width=1, colors='black', grid_alpha=0.5)

    # Optional: add grid
    ax.grid(False)

    # Tight layout for clean spacing
    #plt.tight_layout()
    plt.savefig(f'{forest}.png')
    plt.savefig(f'{forest}.pdf')
    plt.show()

def multiply_rasters_extra(loads_raster_path, fraction_raster_path,raster_extra):
    # 1. Read the fuel loads raster
    with rasterio.open(loads_raster_path) as src_load:
        load_data = src_load.read(1)  # Read band 1
        profile = src_load.profile.copy()  # Copy metadata

    # 2. Read the surface fraction raster
    with rasterio.open(fraction_raster_path) as src_fraction:
        fraction_data = src_fraction.read(1)

    with rasterio.open(raster_extra) as src_extra:
        extra_data = src_extra.read(1)

    # Optional: Check that both rasters have the same dimensions
    if load_data.shape != fraction_data.shape:
        raise ValueError("The two rasters must have the same shape/dimensions.")

    # 3. Perform division
    multiplied_data = (load_data+extra_data) * fraction_data_safe

    return multiplied_data

if __name__ == "__main__":

    #DATA
    #forest selection
    forest = "sub40"
    fuels = f"/home/matias/Documents/Emisiones/{forest}/forest/fuels.asc"
    #directory of surf fraction burn rasters (in c2f output, the values are sfb x fuel load)
    sfb_directory = f"/home/matias/Documents/Emisiones/{forest}/results/preset/SurfFractionBurn/"
    dpv_path = f"/home/matias/Documents/Emisiones/{forest}/results/preset/{forest}_dpv.asc"
    #full emissions file path (emissions if sfb=1)
    fe_path = f"/home/matias/Documents/Emisiones/{forest}/forest/{forest}_fe.asc"
    fe_path_copa = f"/home/matias/Documents/Emisiones/{forest}/forest/{forest}_fe_copa.asc"
    fe_file_total = f"/home/matias/Documents/Emisiones/{forest}/forest/{forest}_fe_total.asc"
    #gas factors raster path (raster of sum(gfe x ponderator))
    gf_path = f"/home/matias/Documents/Emisiones/{forest}/forest/{forest}_gf.asc"

    #0. RUN PRESET SIMULATIONS
    c2f_preset(forest,nsims=10000)
    
    percentages = [0.1]
    args = [forest,fuels,sfb_directory,dpv_path,fe_path,gf_path,fe_file_total,percentages]
    #1. RUN VALIDATION USING C2F
    #c2f_validation(args,nsims=10000)
    #os.rename("/home/matias/Documents/Emisiones/dogrib/results/validation/0.1/IgnitionsHistory/replication.csv", "/home/matias/Documents/Emisiones/dogrib/results/validation/0.1/IgnitionsHistory/Ignitions.csv")
    #os.replace("/home/matias/Documents/Emisiones/dogrib/results/validation/0.1/IgnitionsHistory/Ignitions.csv", "/home/matias/Documents/Emisiones/dogrib/forest/Ignitions.csv")
    
    percentages = [0,0.01,0.03,0.05,0.07]
    args = [forest,fuels,sfb_directory,dpv_path,fe_path,gf_path,fe_file_total,percentages]
    #1. RUN VALIDATION USING C2F
    #c2f_validation2(args,nsims=10000)

    #2. PROCESS RESULTS
    percentages = [0]
    args = [forest,fuels,sfb_directory,dpv_path,fe_path,gf_path,fe_file_total,percentages]
   # mean_cero, confidence_cero, zero_emissions_list = process_results_uni(args)
    
    percentages = [0.01,0.03,0.05,0.07,0.1]
    args = [forest,fuels,sfb_directory,dpv_path,fe_path,gf_path,fe_file_total,percentages]
   # mean_list, confidence = process_results(args,zero_emissions_list)
    
   # mean_list.insert(0,mean_cero[0])
   # confidence.insert(0,confidence_cero[0])
   # print(mean_list)
    
    percentages_list = [0,0.01,0.03,0.05,0.07,0.1]
    #3. PLOT RESULTS
   # plot_results(mean_list,confidence,percentages_list,forest)
    
    
    
