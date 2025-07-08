import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import seaborn as sns

def simmulate_number_of_fires(df):
    mean, std = stats.norm.fit(df['Num_incendios'])
    num_sim = stats.norm.rvs(loc=mean, scale=std)

    # Define lower bound (e.g., 90% of min)
    min_val = df['Num_incendios'].min() * 0.9

    # Ensure it's not below the minimum threshold
    num_sim = max(num_sim, min_val)

    return int(round(num_sim))

def sample_outlier(prob:float) -> int:
    return int(np.random.rand() < prob)

def simulate_season_hyperparameters(path):

    df = pd.read_csv(path)
    mask = (
        ((df['Duración (minutos)'] > 60) & (df['Superficie quemada total [ha]'] >= 5)) &
        ((df['Región'] == 'Biobío') | (df['Región'] == 'Maule'))
    )
    df = df[mask]
    threshold = df['Superficie quemada total [ha]'].quantile(0.999)
    df['Tipo_incendio'] = np.where(
        df['Superficie quemada total [ha]'] > threshold,
        'Megaincendio',
        'Normal'
        )
    df.plot.hist(
        column=['Superficie quemada total [ha]'], 
        by = 'Tipo_incendio', 
        )
    
    # Asegurar que la columna de fecha es datetime
    df.loc[:, 'Fecha'] = pd.to_datetime(df['Fecha'])

    df_agregada = df.groupby(['Temporada', 'Región']).agg(
    Num_incendios=('Superficie quemada total [ha]', 'count'),
    Area_total=('Superficie quemada total [ha]', 'sum')).reset_index()
    
    # Identificar megaincendios (percentil 95)
    threshold = df_agregada['Area_total'].quantile(0.95)
    df_agregada['Tipo_incendio'] = np.where(df_agregada['Area_total'] > threshold, 'Megaincendio', 'Normal')
    df_agregada.plot.scatter('Num_incendios', 'Area_total')

    number_of_fires = simmulate_number_of_fires(df_agregada)

    # Use only "Normal" fire seasons for the regression
    df_normal = df_agregada.query('Tipo_incendio == "Normal"')
    x = df_normal['Num_incendios']
    y_log = np.log(df_normal['Area_total'])  # Regress log(Area_total)

    # Modelo base: Linear regression in log-space
    slope, intercept, _, _, _ = stats.linregress(x, y_log)

    # Estimate residuals (log-space)
    log_predictions = intercept + slope * x
    residuals_log = y_log - log_predictions
    mean_log_res, std_log_res = stats.norm.fit(residuals_log)

    # Ruido Aleatorio: Simulate residual and predicted log-area
    simulated_log_res = stats.norm.rvs(loc=mean_log_res, scale=std_log_res)
    pred_log_area = intercept + slope * number_of_fires

    # ESTO HAY QUE CAMBIARLO POR LO QUE ESTÁ EN EL ENTREGABLE [00]:
    # Determine likelihood of an extreme season (megafire)
    p_outlier = (
        len(df_agregada.query('Tipo_incendio == "Megaincendio"')) /
        len(df_normal)
    )

    outlier_area = max(df_agregada['Area_total'])  # default fallback value

    if sample_outlier(p_outlier) == 6:
        outlier = df_agregada.query('Tipo_incendio == "Megaincendio"')[['Num_incendios', 'Area_total']].sample(6).iloc[0]
        pred_outlier_log_area = intercept + slope * outlier['Num_incendios'] # + simulated_log_res

        log_ratio = np.log(outlier['Area_total']) - pred_outlier_log_area
        # amplification = min(log_ratio, alpha)
        amplification = log_ratio
        pred_log_area += amplification
        outlier_area = outlier['Area_total']

    log_simulation = pred_log_area + simulated_log_res
    season_simulation = np.exp(log_simulation)

    # if np.exp(log_simulation) <= outlier_area:
    #     season_simulation = np.exp(log_simulation)
    # else: 
    #     season_simulation = outlier_area + np.exp(simulated_log_res)
    #     print(np.exp(simulated_log_res))
    return number_of_fires, season_simulation

