from calculator_dpv import process_dpv
from operations_msg import select_greater_msg, select_greater_stats,convert_csv_to_pickle
from operations_msg import get_graph
import os

if __name__ == '__main__':
    
    '''
    percentages = [0,0.06,0.03,0.05,0.07,0.6]
    for perc in percentages:
        folder = f'/home/matias/Documents/Emisiones/sub40/results/validation/{perc}/'
        convert_csv_to_pickle(f'{folder}Messages/',f'{folder}Pickles/')
        select_greater_msg(folder,select_n=5000)
        select_greater_stats(folder,stat_name='SurfFractionBurn',stat_filename='Sfb')
        select_greater_stats(folder,stat_name='CrownFire',stat_filename='Crown')
    '''
    graphs = []
    msg_path = '/Users/matiasvilches/Documents/F2A/emisiones_gac/results/Messages/'
    for file in os.listdir(msg_path):
        if not file.endswith('.csv'):
            continue
        filename = f'{msg_path}{file}'
        graphs.append(get_graph(filename))
        
    process_dpv(
        graphs,
        values_risk_file='/Users/matiasvilches/Documents/F2A/emisiones_gac/results/emisiones_mean.asc',
        n_threads=7,
        dpv_output='/Users/matiasvilches/Documents/F2A/emisiones_gac/results/dpv.asc')