import os
import re
import numpy as np
import scipy.stats as st
from multiprocessing import Pool
from operations_raster import multiplicar_raster_por_valor
from carbono import multiply_rasters_woo, sum_data_values, multiply_raster_data
import seaborn as sns
import matplotlib.pyplot as plt


# HACER MIL SIMULACIONES PARA SUB40 CON CANADA
# CALCULAR LAS EMISIONES DE CARBONO PARA CADA SIMULACION CON EL MÉTODO DE IPCC Y EL DEL PAPER
# OBTENER EL SCATTER PLOT Y EL KERNEL DENSITY PLOT

# Tengo que leer las mil simulaciones y calcular las emisiones de carbono usando el ipcc y el método del paper.
# Para calcular las emisiones necesito multiplicar el cf por el resto de la formula. El problema es que
# el simulador lo que me entrega es el total fuel consumption y yo necesito sólo el surface burn fraction.
# para obtener las emisiones del paper, es suficiente con tener el tfc del simulador. para el ipcc necesito un
# archivo con el fuel load de copa y el fuel load de superficie.


results_path = '/Users/matiasvilches/Documents/F2A/carbono/validacion_ipcc'
forest_path = '/Users/matiasvilches/Documents/F2A/carbono/sub40/'
gf_path = f"{forest_path}/sub40_gf.asc"
crown_fuel_load = f"{forest_path}/sub40_fl_copa.asc"
surf_fuel_load = f"{forest_path}/sub40_fl.asc"

val_sfb_folder = f"{results_path}/SurfFractionBurn/"
val_crown_folder = f"{results_path}/CrownFire/"

emissions_paper = []
emissions_ipcc = []
# Calculo de emisiones de carbono usando el método del paper
for filename in os.listdir(val_sfb_folder):
    
    # Calculo de consumo de superficie
    sfb_file = os.path.join(val_sfb_folder, filename)
    ipcc_tfc_surf = multiplicar_raster_por_valor(surf_fuel_load,0.65)

    # Extract file number
    file_number = re.findall(r'\d+', filename)[0]
    
    # calculo de consumo de copa
    crown_file = os.path.join(val_crown_folder, f'Crown{file_number}.asc')
    crown_sfb_data = multiply_rasters_woo(crown_fuel_load, crown_file)
    ipcc_tfc_crown = multiplicar_raster_por_valor(crown_file, 0.43)
    
    # Calculo de emisiones de carbono usando el método del paper
    paper_surf_emissions = multiply_rasters_woo(gf_path, sfb_file)
    paper_crown_emissions = multiply_raster_data(gf_path, crown_sfb_data)

    for i in range(len(paper_surf_emissions)):
        for j in range(len(paper_surf_emissions[i])):
            if paper_surf_emissions[i][j] == 0:
                ipcc_tfc_surf[i][j] = 0
                ipcc_tfc_crown[i][j] = 0

    # Calculo de emisiones de carbono usando el método del IPCC     
    ipcc_surf_emissions = multiply_raster_data(gf_path, ipcc_tfc_surf)
    ipcc_crown_emissions = multiply_raster_data(gf_path, ipcc_tfc_crown)

    # Suma de emisiones de carbono metodo del paper
    wildfire_surf_emissions = sum_data_values(paper_surf_emissions)
    wildfire_crown_emissions = sum_data_values(paper_crown_emissions)
    total_emissions_paper = wildfire_surf_emissions + wildfire_crown_emissions

    # Suma de emisiones de carbono metodo del IPCC
    wildfire_surf_emissions = sum_data_values(ipcc_surf_emissions)
    wildfire_crown_emissions = sum_data_values(ipcc_crown_emissions)
    total_emissions_ipcc = wildfire_surf_emissions + wildfire_crown_emissions

    emissions_paper.append(total_emissions_paper)
    emissions_ipcc.append(total_emissions_ipcc)

#print(f"Emisiones del paper: {emissions_paper[0:60]}")
#print(f"Emisiones del IPCC: {emissions_ipcc[0:60]}")

# Supongamos que ya tienes listas:
# emisiones_simuladas = [...]
# emisiones_ipcc = [...]

# KDE Plot
def plot_kde(emissions_paper, emissions_ipcc):
    emissions_paper = [x / 6e3 for x in emissions_paper]
    emissions_ipcc = [x / 6e3 for x in emissions_ipcc]
    sns.kdeplot(x=emissions_paper, label='Simulation', color='blue', fill=True)
    sns.kdeplot(x=emissions_ipcc, label='IPCC', color='red', linestyle='--')
    plt.xlabel('Total Emissions (kTon CO2Eq.)')
    plt.ylabel('Density')
    plt.legend()
    plt.title('Emissions distribuition: Simulation vs IPCC')
    #plt.savefig('/Users/matiasvilches/Documents/F2A/carbono/validacion_ipcc/kde_plot.pdf')
    plt.show()

# Scatter Plot
def plot_scatter(emissions_paper, emissions_ipcc):
    emissions_paper = [x / 6e3 for x in emissions_paper]
    emissions_ipcc = [x / 6e3 for x in emissions_ipcc]
    plt.scatter(emissions_paper, emissions_ipcc, alpha=0.5)
    max_val = max(max(emissions_paper), max(emissions_ipcc))
    plt.plot([0, max_val], [0, max_val], 'k--', label='6:6 Line')
    plt.xlabel('Simulated total emissions (kTon CO2Eq.)')
    plt.ylabel('IPCC total emissions (kTon CO2Eq.)')
    plt.legend()
    plt.title('Total Emissions Comparison: Simulation vs IPCC')    
    #plt.savefig('/Users/matiasvilches/Documents/F2A/carbono/validacion_ipcc/scatter_plot.pdf')
    plt.show()

plot_kde(emissions_paper, emissions_ipcc)
plot_scatter(emissions_paper, emissions_ipcc)