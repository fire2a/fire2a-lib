from calculator_dpv import process_dpv
from operations_msg import select_greater_msg, select_greater_stats,convert_csv_to_pickle
import os

if __name__ == '__main__':
    
    '''
    percentages = [0,0.01,0.03,0.05,0.07,0.1]
    for perc in percentages:
        folder = f'/home/matias/Documents/Emisiones/sub40/results/validation/{perc}/'
        convert_csv_to_pickle(f'{folder}Messages/',f'{folder}Pickles/')
        select_greater_msg(folder,select_n=5000)
        select_greater_stats(folder,stat_name='SurfFractionBurn',stat_filename='Sfb')
        select_greater_stats(folder,stat_name='CrownFire',stat_filename='Crown')
    '''
    procces_dpv()
