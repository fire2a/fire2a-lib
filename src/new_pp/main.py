from calculator_dpv import process_dpv
from operations_msg import get_graph

if __name__ == '__main__':
    msg_path = 'simulationFile2.csv'
    var = 'uniraster.asc'

    grafo = get_graph(msg_path)
    g = [grafo]

    print(grafo)
    print(grafo.get_edge_data(90, 51))

    dpv = process_dpv(g,var,1,'')

    for i in range(0,len(dpv)):
        if dpv[i] > 0:
            print(i+1, dpv[i])

