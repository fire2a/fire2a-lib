import pandas as pd
import rasterio
import numpy as np
import os
import networkx as nx

def create_map(gases):

    #Leer csv con emisiones
    fe_csv = "/home/matias/Documents/Emisiones/fuel_emissions.csv"
    fuel_column = "fuelType"  # Nombre de la columna que contiene el nombre del fuel
    df_fuel_emissions = pd.read_csv(fe_csv)

    #Leer csv con codigos de combustible
    fuel_codes_csv = "/home/matias/Documents/Emisiones/sub40/forest/fbp_lookup_table.csv"
    fuel_column_codes = "fuel_type"  # Nombre de la columna con el nombre del fuel
    fuel_code_column = "grid_value"  # Nombre de la columna con el código del fuel
    df_fuel_codes = pd.read_csv(fuel_codes_csv)  # Leer el CSV
    print(df_fuel_codes)

    #print(df_fuel_emissions)
    #Sumar los valores de los gases para cada fuelType
    ponderadores = [0.01,0.272,2.73]
    df_fuel_emissions["total_fuel_emissions"] = (df_fuel_emissions[gases] * ponderadores).sum(axis=1)

    df_combined = pd.merge(
        df_fuel_codes[[fuel_column_codes, fuel_code_column]],
        df_fuel_emissions[[fuel_column, "total_fuel_emissions"]],
        left_on=fuel_column_codes,
        right_on=fuel_column,
        how="inner"
    )

    print(df_fuel_emissions)

    # Crear un diccionario de mapeo {codigo: total_fuel_load}
    code_to_fuel_load = dict(zip(df_combined[fuel_code_column], df_combined["total_fuel_emissions"]))

    # Paso 5: Leer el raster de entrada con los códigos de fuel
    input_raster = "/home/matias/Documents/Emisiones/sub40/forest/fuels.asc"
    output_raster = "/home/matias/Documents/Emisiones/sub40/forest/sub40_gf.asc"  # Archivo único de salida

    with rasterio.open(input_raster) as src:
        raster_data = src.read(1)  # Leer la banda 1
        profile = src.profile  # Obtener metadatos del raster

        # Paso 6: Crear un nuevo raster sumando los valores de fuel loads
        fuel_load_raster = np.zeros_like(raster_data, dtype=np.float32)
        for code, total_fuel_emissions in code_to_fuel_load.items():
            fuel_load_raster[raster_data == code] = total_fuel_emissions

    # Paso 7: Guardar el raster de salida
    profile.update(dtype=rasterio.float32, nodata=0)  # Asegurarse de que el dtype sea float32
    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(fuel_load_raster, 1)

    print(f"Raster único generado con la suma de los gases: {output_raster}")

def generate_fc_map():
    # 1. Read the CSV containing (fuel code -> fuel load)
    fuel_load_csv = "/home/matias/Documents/Emisiones/fuel_values.csv"
    fuel_column = "fuelType"
    fuel_load_column = "fuelLoad"

    df_fuel_load = pd.read_csv(fuel_load_csv)
    print(df_fuel_load.columns)

    # Create a dictionary {fuel_code: fuel_load}
    code_to_fuel_load = dict(zip(df_fuel_load[fuel_column],
                                df_fuel_load[fuel_load_column]))

    # 2. Paths to input (fuel-code) raster and output (fuel-load) ASCII
    input_raster = "/home/matias/Documents/Emisiones/dogrib-asc/fuels.asc"
    output_raster = "/home/matias/Documents/Emisiones/fuel_load.asc"

    # 3. Open the input raster
    with rasterio.open(input_raster) as src:
        raster_data = src.read(1)      # Read the first band as a NumPy array
        profile = src.profile.copy()   # Copy the metadata (profile)

    # 4. Create an empty array (float32) to store fuel loads
    fuel_load_raster = np.zeros_like(raster_data, dtype=np.float32)

    # 5. Replace each fuel code with the corresponding fuel load
    for code, fuel_load in code_to_fuel_load.items():
        # Where the raster_data equals "code", set the output to the fuel load
        fuel_load_raster[raster_data == code] = fuel_load

    # 6. Update the profile for ASCII output
    #    - Specify driver='AAIGrid' so rasterio writes Arc/Info ASCII
    #    - Make sure dtype and nodata are set appropriately
    profile.update(
        driver="AAIGrid",
        dtype=rasterio.float32,
        nodata=0
    )

    # 7. Write the output ASCII grid
    with rasterio.open(output_raster, "w", **profile) as dst:
        dst.write(fuel_load_raster, 1)

    print(f"ASCII grid with fuel loads created: {output_raster}")

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

    # 3. Perform division
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

def divide_rasters(loads_raster_path, fraction_raster_path, output_raster_path,raster_extra):
    
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
    
    # ⚠️ Replace zeros with a small value (to avoid division by zero)
    fraction_data_safe = np.where(fraction_data == 0, np.nan, fraction_data)

    # 3. Perform division
    multiplied_data = (load_data+extra_data) / fraction_data_safe

    # Replace NaN values (from division by zero) with 0 or another value
    multiplied_data = np.nan_to_num(multiplied_data, nan=0)
    
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

def sum_raster_values(raster_path):
    with rasterio.open(raster_path) as src:
        data = src.read(1)  # Read the first band (assuming single-band raster)
        total_sum = np.nansum(data)  # Sum all values, ignoring NaNs (if any)
    return total_sum

def check_messages(msg_path):

    H = nx.read_edgelist(path = msg_path,
                            delimiter=',',
                            create_using = nx.DiGraph(),
                            nodetype = int,
                            data = [('time', float), ('ros', float)])

    nodos = list(H.nodes())
    print(len(nodos))
    #print(nodos)

def sum_raster(raster1,raster2,output_raster):
    with rasterio.open(raster1) as src:
        data1 = src.read(1)  # Read the first band (assuming single-band raster)

    with rasterio.open(raster2) as src:
        data2 = src.read(1)  # Read the first band (assuming single-band raster)

    data3 = data1+data2
    header,_ = read_asc(raster1)

    write_asc(output_raster,header,data3)

def write_asc(file_path, header, array):
    with open(file_path, 'w') as file:
        # Escribir el encabezado
        file.writelines(header)
        # Escribir los datos numéricos
        np.savetxt(file, array, fmt='%.6f')

def read_asc(file_path):
    with open(file_path, 'r') as file:
        # Leer las primeras seis líneas y almacenarlas
        header = [next(file) for _ in range(6)]
        # Leer los datos numéricos
        data = np.loadtxt(file, dtype=np.float32)
    return header, data

#CREATE FOREST_GF FILE
#gases = ["CO2","CH4","N2O"]
#create_map(gases)
forest = "sub40"
raster1 = f"/home/matias/Documents/Emisiones/{forest}/forest/{forest}_gf.asc"
raster2 = f"/home/matias/Documents/Emisiones/{forest}/forest/{forest}_fl_copa.asc"
raster3 = f"/home/matias/Documents/Emisiones/{forest}/forest/{forest}_fe_copa.asc"
multiply_rasters(raster1,raster2,raster3)

raster1 = f"/home/matias/Documents/Emisiones/{forest}/forest/{forest}_fe_copa.asc" 
raster2 = f"/home/matias/Documents/Emisiones/{forest}/results/preset/{forest}_mean_crown.asc"
raster3 = f"/home/matias/Documents/Emisiones/{forest}/results/preset/{forest}_mean_cfb.asc"
multiply_rasters(raster1,raster2,raster3)

#CREATE FULL EMISIONS FOREST_FE FILE
raster1 = f"/home/matias/Documents/Emisiones/{forest}/results/preset/{forest}_mean_sfb.asc"
raster2 = f"/home/matias/Documents/Emisiones/{forest}/results/preset/{forest}_bp.asc"
output3 = f"/home/matias/Documents/Emisiones/{forest}/results/preset/var_fe_total.asc"
raster_add = f"/home/matias/Documents/Emisiones/{forest}/results/preset/{forest}_mean_cfb.asc"
divide_rasters(raster1,raster2,output3,raster_add)

raster1 = f"/home/matias/Documents/Emisiones/{forest}/forest/{forest}_fe.asc"
raster2 = f"/home/matias/Documents/Emisiones/{forest}/forest/{forest}_fe_copa.asc"
output_raster = f"/home/matias/Documents/Emisiones/{forest}/forest/{forest}_fe_total.asc"
sum_raster(raster1,raster2,output_raster)
