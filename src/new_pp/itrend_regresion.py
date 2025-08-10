import pandas as pd
import rasterio
from pyproj import Transformer
import numpy as np

def filter_biobio_incendios():
    # read file data
    file = "/Users/matiasvilches/Documents/F2A/ITREND/data/incendios_control.csv"
    df = pd.read_csv(file)

    # only keep columns "Región", "Coord. operativas Lat", "Coord. operativas Lon", "Superficie total"
    df = df[["Región", "Coord. operativas Lat", "Coord. operativas Lon", "Duración (horas)","Superficie total"]]

    # only keep rows where "Región" is "Biobío"
    df = df[df["Región"] == "Biobío"]

    # save to new file
    df.to_csv("/Users/matiasvilches/Documents/F2A/ITREND/data/incendios_biobio.csv", index=False)

def read_biobio_incendios():
    # read file with historical fires
    historical_fires_file = "/Users/matiasvilches/Documents/F2A/ITREND/data/incendios_biobio.csv"
    df = pd.read_csv(historical_fires_file)

    # make regression with duracion and superficie
    df["Duración (horas)"] = pd.to_numeric(df["Duración (horas)"], errors='coerce')
    df["Superficie total"] = pd.to_numeric(df["Superficie total"], errors='coerce')
    
    df = df.dropna(subset=["Duración (horas)", "Superficie total"])
    
    df = df[df["Superficie total"] > 0]  # Filter out non-positive areas
    df = df[df["Duración (horas)"] > 0]  # Filter out non-positive durations
    
    reg = regression_model(df, "Superficie total", "Duración (horas)")
    print("Regression model created successfully.")

def get_distance(lat, lon, src, distance_array):
    """
    Get raster value at given WGS84 coordinates for a raster in EPSG:32718
    
    Args:
        lat: Latitude in WGS84 (EPSG:4326)
        lon: Longitude in WGS84 (EPSG:4326)
        src: Opened rasterio dataset (EPSG:32718)
        distance_array: Numpy array with raster values
        
    Returns:
        Distance value from raster or np.nan if out of bounds
    """
    # Create coordinate transformer
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:32718", always_xy=True)
    
    try:
        # Transform WGS84 to UTM Zone 18S
        x, y = transformer.transform(lon, lat)
        
        # Check if coordinates are within raster bounds
        if not (src.bounds.left <= x <= src.bounds.right and 
                src.bounds.bottom <= y <= src.bounds.top):
            return np.nan
        
        # Convert to row/col indices
        row, col = src.index(x, y)
        
        # Return value if within array bounds
        if (0 <= row < distance_array.shape[0] and 
            0 <= col < distance_array.shape[1]):
            return distance_array[row, col]
        
        return np.nan
    
    except Exception as e:
        print(f"Error processing ({lat}, {lon}): {str(e)}")
        return np.nan

def regression_model(historical_fires_df,column1, column2):
    # make regression with "Distancia a pueblos" and "Duración (horas)"
    from sklearn.linear_model import LinearRegression
    X = historical_fires_df[[column2]]
    y = historical_fires_df[column1]
    model = LinearRegression()
    model.fit(X, y)

    # print coefficients
    print("Coeficiente:", model.coef_[0])
    print("Intercepto:", model.intercept_)

    # print R^2 score
    r2_score = model.score(X, y)
    print("R^2 score:", r2_score)

    return model

def write_raster(model, distance_raster_path, output_path):
    """
    Predict values from a distance raster and save as new raster, ignoring NaNs or nodata values.

    Args:
        model: Trained model with .predict()
        distance_raster_path: Path to input raster (with distances)
        output_path: Path to write predicted raster
    """
    with rasterio.open(distance_raster_path) as src:
        distance = src.read(1)  # Read first band (2D numpy array)
        profile = src.profile.copy()

        # Create mask for invalid values (NaN or nodata)
        nodata_value = profile.get("nodata")
        if nodata_value is not None:
            mask = (distance == nodata_value) | np.isnan(distance)
        else:
            mask = np.isnan(distance)

        # Flatten input for prediction
        flat_distance = distance.flatten()
        valid_indices = ~mask.flatten()
        X_valid = pd.DataFrame(flat_distance[valid_indices], columns=["Distancia a pueblos"])

        # Predict only on valid entries
        predicted_flat = np.full_like(flat_distance, fill_value=nodata_value if nodata_value is not None else -9999, dtype=np.float32)
        predicted_flat[valid_indices] = model.predict(X_valid)

        # Reshape back to raster shape
        predicted = predicted_flat.reshape(distance.shape)

        # Update metadata for output raster
        profile.update({
            "driver": "AAIGrid",
            "dtype": "float32",
            "count": 1,
            "nodata": nodata_value if nodata_value is not None else -9999
        })

        # Write output raster
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(predicted, 1)

if __name__ == "__main__":

    #historical_fires_file = "/Users/matiasvilches/Documents/F2A/ITREND/data/incendios_biobio.csv"
    read_biobio_incendios()
    #filter_biobio_incendios()
    # read file with distance to roads
    #distance_file = "/Users/matiasvilches/Documents/F2A/ITREND/data/distancia_pob.tif"
    #with rasterio.open(distance_file) as src:
    #    distance = src.read(1)

    """
    distance_file = "/Users/matiasvilches/Documents/F2A/ITREND/data/datForModelingAllObs.tif"

    with rasterio.open("/Users/matiasvilches/Documents/F2A/ITREND/data/datForModelingAllObs.tif") as src:
        band_index = src.descriptions.index("pob_dist") + 1  # rasterio is 1-based
        distance = src.read(band_index)

    # read file with historical fires and add distance to pueblos
    historical_fires_file = "/Users/matiasvilches/Documents/F2A/ITREND/data/incendios_biobio_sup.csv"
    historical_fires_df = pd.read_csv(historical_fires_file)
    historical_fires_df["Distancia a pueblos"] = historical_fires_df.apply(
        lambda row: get_distance(row["Coord. operativas Lat"], row["Coord. operativas Lon"], src, distance), axis=1
    )

    # count nan values in "Distancia a pueblos"
    nan_count = historical_fires_df["Distancia a pueblos"].isna().sum()
    historical_fires_df = historical_fires_df.dropna(subset=["Distancia a pueblos"])
    print(f"Number of fires in dataframe: {len(historical_fires_df)}")

    # write new file with distance to pueblos
    #historical_fires_df.to_csv(
    #    "/Users/matiasvilches/Documents/F2A/ITREND/data/incendios_biobio_with_distance.csv",
    #    index=False)

    if False:
        # Load the raster (example)
        with rasterio.open("/Users/matiasvilches/Documents/F2A/ITREND/data/combatPoints.asc") as src:
            combat_array = src.read(1)  # Read the raster values
            # Apply the function row-wise
            historical_fires_df["raster_value"] = historical_fires_df.apply(
                lambda row: get_distance(
                    row["Coord. operativas Lat"],
                    row["Coord. operativas Lon"],
                    src,
                    combat_array
                ), axis=1
            )

        # Keep only those where the raster has value 1
        historical_fires_df = historical_fires_df[historical_fires_df["raster_value"] == 1].copy()

    #count number of fires in dataframe
    print("Number of fires considered:", len(historical_fires_df))

    model = regression_model()
    #write_raster(model, distance_file, "/Users/matiasvilches/Documents/F2A/ITREND/data/estimated_duration.tif")
    """