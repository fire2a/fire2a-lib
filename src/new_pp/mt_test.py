import time
import multiprocessing as mp

def run_all_seasons_parallel(temporadas, n_cores, forest, forest_path, fuel_wip_folder, ruta_base, database_path):
    original_fuels = os.path.join(forest_path, "fuels.asc")
    fuel_files = prepare_fuel_pool(original_fuels, fuel_wip_folder, n_cores)

    # Usar Pool para gestionar procesos
    with Pool(n_cores) as pool:
        pool.starmap(
            run_season_with_pool,
            [
                (t, forest, fuel_wip_folder, ruta_base, database_path, original_fuels)
                for t in range(1, temporadas + 1)
            ]
        )

    # Limpiar archivos temporales
    for file in fuel_files:
        os.remove(file)

if __name__ == "__main__":

    cola = mp.Queue()
    p = mp.Process(target=worker,args=(cola,))
    p.start()

    p.join()

    mensaje = cola.get()
    print(mensaje)  # Salida: "Hola desde el proceso hijo!"

    


