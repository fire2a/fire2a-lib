import seaborn as sns
import matplotlib.pyplot as plt
from operations_msg import get_scar_size
import statistics as st
import numpy as np

path_in = f'/Users/matiasvilches/Documents/F2A/exp_dpv/results/mip_test/post_ic_'
path_c = f'/Users/matiasvilches/Documents/F2A/exp_dpv/results/mip_test/post_c_'
        

# Función para obtener el tamaño de las cicatrices
cicatrices_in_6 = get_scar_size(f'{path_in}06_mip/Messages/', 6600)
#cicatrices_in_3 = get_scar_size(f'{path_in}03/Messages/', 6600)
#cicatrices_in_5 = get_scar_size(f'{path_in}05/Messages/', 6600)
cicatrices_c_6 = get_scar_size(f'{path_c}06_mip/Messages/', 6600)
#cicatrices_c_3 = get_scar_size(f'{path_c}03/Messages/', 6600)
#cicatrices_c_5 = get_scar_size(f'{path_c}05/Messages/', 6600)

#"""
# Boxplot comparativo para las 6 combinaciones
datos = [
    cicatrices_in_6, cicatrices_c_6]#,cicatrices_in_3, cicatrices_c_3,cicatrices_in_5, cicatrices_c_5]
labels = [
    'incomplete 6%', 'complete 6%'
    #'incomplete 3%', 'complete 3%',
    #'incomplete 5%', 'complete 5%'
]

# Posiciones personalizadas para espaciar los grupos
positions = [6, 2]#, 4, 5, 7, 8]  # deja un espacio entre cada par

plt.boxplot(datos, positions=positions, patch_artist=True,
            boxprops=dict(facecolor='lightblue'),
            medianprops=dict(color='black'))

for i, d in enumerate(datos):
    media = np.mean(d)
    plt.text(positions[i], media, f'{media:.2f}',
             ha='center', va='bottom', fontsize=9, color='darkblue')

plt.xticks(positions, labels, rotation=45)
plt.ylabel('Burned Area (% of total area)')
plt.title('Comparación de cicatrices')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
#"""