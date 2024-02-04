#!/bin/env python3
"""
$cd C2FSB
C2FSB$ python3 downstream_protection_value.py

all functions are tested and plotted on main

Calculate downstream protection value from Messages/MessagesFile<int>.csv files
Each file has 4 columns: from cellId, to cellId, period when burns & hit ROS

https://github.com/fire2a/C2FK/blob/main/Cell2Fire/Heuristics.py

https://networkx.org/documentation/networkx-1.8/reference/algorithms.shortest_paths.html

propagation tree: (c) fire shortest traveling times

Performance review
1. is faster to dijkstra than minimun spanning

    In [50]: %timeit shortest_propagation_tree(G,root)
    1.53 ms ± 5.47 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    
    In [51]: %timeit nx.minimum_spanning_arborescence(G, attr='time')
    16.4 ms ± 61 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

2. is faster numpy+add_edges than nx.from_csv

    In [63]: %timeit custom4(afile)
    2.3 ms ± 32 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
    
    In [64]: %timeit canon4(afile)
    3.35 ms ± 20 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

2.1 even faster is you discard a column!!
    In [65]: %timeit digraph_from_messages(afile)
    1.84 ms ± 15.4 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)

"""
import re
import sys
from logging import debug
from pathlib import Path

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from osgeo import gdal


def single_simulation_downstream_protection_value(msgfile="MessagesFile01.csv", pvfile="py.asc"):
    """load one diGraph count succesors"""
    msgG, root = digraph_from_messages(msgfile)
    treeG = shortest_propagation_tree(msgG, root)
    pv, W, H = get_flat_pv(pvfile)
    #
    dpv = np.zeros(pv.shape, dtype=pv.dtype)
    i2n = [n - 1 for n in treeG]
    mdpv = dpv_maskG(treeG, root, pv, i2n)
    dpv[i2n] = mdpv
    return mdpv, dpv


def downstream_protection_value(out_dir, pvfile):
    pv, W, H = get_flat_pv(pvfile)
    dpv = np.zeros(pv.shape, dtype=pv.dtype)
    file_list = read_files(out_dir)
    for msgfile in file_list:
        msgG, root = digraph_from_messages(msgfile)
        treeG = shortest_propagation_tree(msgG, root)
        i2n = [n - 1 for n in treeG]  # TODO change to list(treeG)
        mdpv = dpv_maskG(treeG, root, pv, i2n)
        dpv[i2n] += mdpv
        # plot_pv( dpv, w=W, h=H)
    return dpv / len(file_list)


def canon3(afile):
    # NO IMPLEMENTADO
    G = nx.read_edgelist(
        path=afile, create_using=nx.DiGraph(), nodetype=np.int32, data=[("time", np.int16)], delimiter=","
    )
    return G


def canon4(afile):
    G = nx.read_edgelist(
        path=afile,
        create_using=nx.DiGraph(),
        nodetype=np.int32,
        data=[("time", np.int16), ("ros", np.float32)],
        delimiter=",",
    )
    return G


def digraph_from_messages(afile):
    """Not checking if file exists or if size > 0
    This is done previously on read_files
    """
    data = np.loadtxt(
        afile, delimiter=",", dtype=[("i", np.int32), ("j", np.int32), ("time", np.int16)], usecols=(0, 1, 2)
    )
    root = data[0][0]  # checkar que el primer valor del message sea el punto de ignición
    G = nx.DiGraph()
    G.add_weighted_edges_from(data)
    return G, root


func = np.vectorize(lambda x, y: {"time": x, "ros": y})


def custom4(afile):
    data = np.loadtxt(
        afile, delimiter=",", dtype=[("i", np.int32), ("j", np.int32), ("time", np.int16), ("ros", np.float32)]
    )
    root = data[0][0]
    G = nx.DiGraph()
    bunch = np.vstack((data["i"], data["j"], func(data["time"], data["ros"]))).T
    G.add_edges_from(bunch)
    return G.add_edges_from(bunch), root


def shortest_propagation_tree(G, root):
    """construct a tree with the all shortest path from root
    TODO try accumulating already added edges for checking before asigning should be faster?
    """
    # { node : [root,...,node], ... }
    shortest_paths = nx.single_source_dijkstra_path(G, root, weight="time")
    del shortest_paths[root]
    T = nx.DiGraph()
    for node, shopat in shortest_paths.items():
        for i, node in enumerate(shopat[:-1]):
            T.add_edge(node, shopat[i + 1])
    return T


def recursiveUp(G):
    """count up WTF!!!
    leafs = [x for x in T.nodes if T.out_degree(x)==0]
    """
    for i in G.nodes:
        G.nodes[i]["dv"] = 1

        # G.nodes[i]['dv']=0
    # for leaf in (x for x in G.nodes if G.out_degree(x)==0):
    #    G.nodes[leaf]['dv']=1
    def count_up(G, j):
        for i in G.predecessors(j):
            # G.nodes[i]['dv']+=G.nodes[j]['dv']
            G.nodes[i]["dv"] += 1
            print(i, j, G.nodes[i]["dv"])
            count_up(G, i)

    for leaf in (x for x in G.nodes if G.out_degree(x) == 0):
        count_up(G, leaf)


def dpv_maskG(G, root, pv, i2n=None):
    """calculate downstream protection value in a flat protection value raster
    i2n = [n for n in treeG.nodes]
    1. copies a slice of pv, only Graph nodes
    2. recursively sums downstream for all succesors of the graph (starting from root)
    3. returns the slice (range(len(G) indexed)
    G must be a tree
    """
    if i2n is None:
        i2n = [n - 1 for n in treeG]
    # -1 ok?
    mdpv = pv[i2n]

    # assert mdpv.base is None ,'the slice is not a copy!'
    def recursion(G, i, pv, mdpv, i2n):
        for j in G.successors(i):
            mdpv[i2n.index(i - 1)] += recursion(G, j, pv, mdpv, i2n)
        return mdpv[i2n.index(i - 1)]

    recursion(G, root, pv, mdpv, i2n)
    return mdpv


def add_dpv_graph(G, root, pv):
    """modifies the input Graph recursively:
        1. sums pv into predecesor (or calling)
        2. recursively sums downstream for all succesors
        3. returns itself if no successors
    G must be a tree with 'dv'
    hence returns nothing
    """
    for n in G.nodes:
        G.nodes[n]["dv"] += pv[n - 1]

    def recursion(G, i):
        for j in G.successors(i):
            G.nodes[i]["dv"] += recursion(G, j)
        return G.nodes[i]["dv"]

    recursion(G, root)


def sum_dpv_graph(T, root, pv):
    """returns a copy of T that:
        1. sets pv into each node
        2. recursively sums pv downstream
    T must be a tree (not checked)
    """
    G = T.copy()
    for i in G.nodes:
        G.nodes[i]["dv"] = pv[i - 1]

    def recursion(G, i):
        for j in G.successors(i):
            G.nodes[i]["dv"] += recursion(G, j)
        return G.nodes[i]["dv"]

    recursion(G, root)
    return G


def count_downstream_graph(T, root) -> nx.DiGraph:
    """returns a new Graph with node values of the number of conected nodes downstream"""
    assert nx.is_tree(T), "not tree"
    G = T.copy()
    for i in G.nodes:
        G.nodes[i]["dv"] = 1

    def recursion(G, i):
        for j in G.successors(i):
            G.nodes[i]["dv"] += recursion(G, j)
        return G.nodes[i]["dv"]

    recursion(G, root)
    return G


def read_files(apath):
    """read all MessagesFile<int>.csv files from Messages directory
    return ordered pathlib filelist & simulation_number lists
    TODO is it worth it to sort messages?
    """
    directory = Path(apath, "Messages")
    file_name = "MessagesFile"
    file_list = [f for f in directory.glob(file_name + "[0-9]*.csv") if f.stat().st_size > 0]
    file_string = " ".join([f.stem for f in file_list])
    # sort
    sim_num = np.fromiter(re.findall("([0-9]+)", file_string), dtype=int, count=len(file_list))
    asort = np.argsort(sim_num)
    sim_num = sim_num[asort]
    file_list = np.array(file_list)[asort]
    return file_list


def get_flat_pv(afile):
    """opens the file with gdal as raster, get 1st band, flattens it
    also returns width and height
    """
    dataset = gdal.Open(str(afile), gdal.GA_ReadOnly)
    return dataset.GetRasterBand(1).ReadAsArray().ravel(), dataset.RasterXSize, dataset.RasterYSize


def plot(G, pos=None, labels=None):
    """matplotlib
    TODO scientific format numeric labels
    """
    if not pos:
        pos = {node: [*id2xy(node)] for node in G.nodes}
    if not labels:
        labes = {node: node for node in G.nodes}
    plt.figure()
    nx.draw(G, pos=pos, with_labels=False)
    nx.draw_networkx_labels(G, pos=pos, labels=labels)
    return plt.show()


def plot_pv(pv, w=40, h=40):
    mat = pv.reshape(h, w)
    plt.matshow(mat)
    plt.show()


def id2xy(idx, w=40, h=40):
    """idx: index, w: width, h:height"""
    idx -= 1
    return idx % w, idx // w


if __name__ == "__main__":
    # if len(sys.argv)>1:
    #     input_dir = sys.argv[1]
    #     output_dir = sys.argv[2]
    # else:
    print("run in C2FSB folder")
    # input_dir = Path.cwd() / 'data'
    input_dir = Path("/home/fdo/source/C2F-W3/data/Vilopriu_2013/firesim_231008_110004")
    # output_dir = Path.cwd() / 'results'
    output_dir = Path("/home/fdo/source/C2F-W3/data/Vilopriu_2013/firesim_231008_110004/results")
    print(input_dir, output_dir)
    assert input_dir.is_dir() and output_dir.is_dir()

    # abro el directorio de los messages como una lista con los nombres de los archivos
    file_list = read_files(output_dir)

    # agarrar la capa que ocuparemos como valor a proteger
    ## pv: valores en riesgo
    ## W: Width
    ## H: Height
    pv, W, H = get_flat_pv(input_dir / "fuels.asc")

    #
    # single simulation
    #
    afile = file_list[0]
    msgG, root = digraph_from_messages(afile)
    pos = {node: [*id2xy(node)] for node in msgG}
    treeG = shortest_propagation_tree(msgG, root)

    # count the number of nodes downstream
    countG = count_downstream_graph(treeG, root)

    # asignar el número de nodos aguas abajo a cada nodo respectivamente
    countGv = {n: countG.nodes[n]["dv"] for n in countG}
    plot(countG, pos=pos, labels=countGv)
    # {'dv': 137} == 137 root connects all tree
    assert countG.nodes[root]["dv"] == len(countG), "count_downstream at root is not the same as number of nodes!"
    #
    onev = np.ones(pv.shape)
    #
    # sum dpv=1
    sumG = sum_dpv_graph(treeG, root, onev)
    sumGv = {n: sumG.nodes[n]["dv"] for n in sumG}
    plot(sumG, pos=pos, labels=sumGv)
    assert np.all([sumGv[n] == countGv[n] for n in treeG.nodes]), "sum_dpv(pv=1) != countG values!"
    #
    # add dpv=1
    addG = treeG.copy()
    for n in addG.nodes:
        addG.nodes[n]["dv"] = 0
    add_dpv_graph(addG, root, onev)
    addGv = {n: addG.nodes[n]["dv"] for n in addG}
    plot(addG, pos=pos, labels=addGv)
    assert np.all([addGv[n] == countGv[n] for n in treeG.nodes]), "add_dpv(pv=1) != countG values!"
    #
    # cum dpv=1
    dpv = np.zeros(pv.shape, dtype=pv.dtype)
    i2n = [n - 1 for n in treeG]
    mdpv = dpv_maskG(treeG, root, onev, i2n)
    dpv[i2n] = mdpv
    plot_pv(dpv, w=W, h=H)
    assert np.all([mdpv[i2n.index(n - 1)] == countGv[n] for n in treeG.nodes]), "dpv_maskG(pv=1) != countG values!"
    #
    # single full test
    mdpv, dpv = single_simulation_downstream_protection_value(msgfile=afile, pvfile=input_dir / "bp.asc")
    plot_pv(dpv, w=W, h=H)
    plot(treeG, pos=pos, labels={n: np.format_float_scientific(dpv[n], precision=2) for n in treeG})
    assert np.all(np.isclose(mdpv, dpv[i2n])), "dpv_maskG != dpvalues!"
    #
    # finally
    dpv = downstream_protection_value(output_dir, pvfile=input_dir / "bp.asc")
    plot_pv(dpv, w=W, h=H)
