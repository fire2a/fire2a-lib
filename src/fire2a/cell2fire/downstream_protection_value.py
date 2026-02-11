#!/bin/env python3
__author__ = "Fernando Badilla"
__revision__ = "$Format:%H$"
__doc__ = """
DownStream Protection Value Algorithm can be reduced to recursively counting out nodes in a directed tree graph; When providing node values, recursively summing them. In the context of Cell2Fire simulator, the graph is a (or many) fire propagation tree(s), represented in a (multi) directed graph.

Its main idea is that upstream cells are priority over downstream cells because of the calculated fire propagations; So if you protect an upstream cell, fire won't get to the downstream cells; pondered by the protection values

Its inputs are:

* A directed graph(*), passed as a 'Messages.csv' file, with 'in_node', 'out_node', 'weight' columns
* [optional] a raster with the values to protect (any Non-Data, any Value types) As the aggregation function is sum, means bigger values are more protection than smaller or negative values.

Its output is:

* A raster represeting the graph with added values (or number of cells) downstream (or out nodes, recursively).

(*): If it is not a tree, it will be reduced to a tree by the shortest path algorithm

This code:

* Reads all Messages[0-9]+.csv files from a directory ([0-9]+ represents any integer number)
* Reads the protection values from a gdal compatible raster file
* Calculates the downstream protection value algorithm serially for windows, and parallel for linux
* Is hooked up to usage_template/downstream_protection_value.ipynb for graphical exploration
"""
import logging
import re
import sys
from pathlib import Path

import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from networkx import DiGraph
from numpy import ndarray
from osgeo import gdal

from . import setup_logger
from .raster import id2xy as r_id2xy
from .raster import read_raster

logger = logging.getLogger(__name__)


def id2xy(idx, w=40, h=40):
    """idx: index, w: width, h:height"""
    return r_id2xy(idx - 1, w, h)


# def id2xy(idx, w=40, h=40):
#     """idx: index, w: width, h:height"""
#     idx -= 1
#     return idx % w, idx // w

# def read_raster(filename: str, band: int = 1, data: bool = True, info: bool = True) -> tuple[np.ndarray, dict]:


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


def recursion2(G: DiGraph, i: int, mdpv: ndarray, i2n: list[int]) -> ndarray:
    for j in G.successors(i):
        mdpv[i2n.index(i)] += recursion2(G, j, mdpv, i2n)
    return mdpv[i2n.index(i)]


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


def glob_int_sorted(directory: Path, filename: str = "MessagesFile.csv"):
    """reads all MessagesFile<int>.csv >0 size files from directory, regexes numbers casting to int, sorts them
    Args:
        directory (Path): directory to read
        filename (str): filename to match
    Returns:
        list: ordered list of pathlib files
    """
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory=} is not a directory")
    file_name = Path(filename)
    file_list = [f for f in directory.glob(file_name.stem + "[0-9]*" + file_name.suffix) if f.stat().st_size > 0]
    if len(file_list) == 0:
        raise FileNotFoundError(f"{directory=} has no {file_name=} matching >0 size files")
    file_string = " ".join([f.stem for f in file_list])
    # sort
    sim_num = np.fromiter(re.findall("([0-9]+)", file_string), dtype=int, count=len(file_list))
    asort = np.argsort(sim_num)
    sim_num = sim_num[asort]
    file_list = list(np.array(file_list)[asort])
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


def worker(data, pv, sid):

    # digraph_from_messages(msgfile) -> msgG, root
    msgG = DiGraph()
    msgG.add_weighted_edges_from(data)
    root = data[0][0]
    # shortest_propagation_tree(G, root) -> treeG
    shortest_paths = nx.single_source_dijkstra_path(msgG, root, weight="time")
    del shortest_paths[root]
    treeG = DiGraph()
    for node, shopat in shortest_paths.items():
        for i, node in enumerate(shopat[:-1]):
            treeG.add_edge(node, shopat[i + 1])
    # dpv_maskG(G, root, pv, i2n) -> mdpv
    i2n = [n for n in treeG]
    mdpv = pv[i2n]
    recursion(treeG, root, mdpv, i2n)
    # dpv[i2n] += mdpv
    return mdpv, i2n, sid


from re import search


def load_msg(afile: Path):
    try:
        sim_id = search(r"\d+", afile.stem).group(0)
    except:
        sim_id = "-1"
    data = np.loadtxt(
        afile, delimiter=",", dtype=[("i", np.int32), ("j", np.int32), ("t", np.int32)], usecols=(0, 1, 2), ndmin=1
    )
    return data, sim_id


def get_data(files, callback=None):
    data = []
    for count, afile in enumerate(files):
        sim_id = search("\\d+", afile.stem).group(0)
        data += [
            np.loadtxt(
                afile,
                delimiter=",",
                dtype=[("i", np.int32), ("j", np.int32), ("t", np.int32)],
                usecols=(0, 1, 2),
                ndmin=1,
            )
        ]
        # 1 based to 0 based
        data[-1]["i"] -= 1
        data[-1]["j"] -= 1
        if callback:
            callback(count, sim_id, afile)
        yield data
    with open("messages.pickle", "wb") as f:
        pickle_dump(data, f)


def one_sim_work(afile, pv, sid):
    # digraph_from_messages(msgfile) -> msgG, root
    data = np.loadtxt(
        afile,
        delimiter=",",
        dtype=[("i", np.int32), ("j", np.int32), ("t", np.int32)],
        usecols=(0, 1, 2),
        ndmin=1,
    )
    # 1 based to 0 based
    data["i"] -= 1
    data["j"] -= 1
    msgG = nx.DiGraph()
    msgG.add_weighted_edges_from(data)
    root = data[0][0]
    # shortest_propagation_tree(G, root) -> treeG
    shortest_paths = nx.single_source_dijkstra_path(msgG, root, weight="time")
    del shortest_paths[root]
    treeG = nx.DiGraph()
    for node, shopat in shortest_paths.items():
        for i, node in enumerate(shopat[:-1]):
            treeG.add_edge(node, shopat[i + 1])
    # dpv_maskG(G, root, pv, i2n) -> mdpv
    i2n = [n for n in treeG]
    mdpv = pv[i2n]
    recursion2(treeG, root, mdpv, i2n)
    # dpv[i2n] += mdpv
    print("fin", sid)
    return mdpv, i2n, sid


def argument_parser(argv):
    """parse arguments
    - logger: verbosity, logfile
    """
    import argparse

    def path_check(path):
        path = Path(path)
        if path.is_dir() or (path.is_file() and path.stat().st_size > 0):
            return path
        raise argparse.ArgumentTypeError(f"{path.as_uri()=} is not a valid Path or file size is 0")

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--verbosity",
        help="increase output verbosity (0: warning(default), 1: info, 2>=: debug)",
        action="count",
        default=0,
    )
    parser.add_argument("-l", "--logfile", help="rotating log file see fire2a.setup_logger", type=Path)
    parser.add_argument(
        "-dpvn",
        "--dpv-filename",
        help="dpv raster layer output filename",
        type=str,
        default="dpv.tif",
    )
    parser.add_argument(
        "-pv",
        "--protection-value",
        help="pv raster layer, must be gdal.Open compatible, and match messages width & height",
        type=path_check,
    )
    parser.add_argument(
        "-pvb",
        "--protection-value-band",
        help="pv raster layer band to read, defaults to 1",
        type=int,
        default=1,
    )
    parser.add_argument("-W", help="width of the raster (overrides width read from pv raster)", type=int)
    parser.add_argument("-H", help="height of the raster (overrides width read from pv raster)", type=int)
    parser.add_argument(
        nargs="+",
        dest="messages",
        help=(
            "Messages[0-9+].csv file(s) or directory to parse (hint use glob: Messages*.csv to pass many files or pass"
            " a single directory to read all Messages[0-9+].csv files in it)"
        ),
        type=path_check,
    )
    return parser.parse_args(argv)


def main(argv=None):
    """steps to run dpv"""
    if argv is None:
        argv = sys.argv[1:]
    args = argument_parser(argv)
    logger = setup_logger(__name__, args.verbosity, args.logfile)
    logger.info(f"{args=}")
    logger.debug("debugging...")

    file_list = []
    for path_str in args.messages:
        path = Path(path_str)
        if path.is_file() and path.stat().st_size > 0:
            file_list += [path]
        elif path.is_dir():
            file_list += glob_int_sorted(path)
        else:
            logger.warning(f"{path} is not a file or directory")
    # remove duplicates
    file_list = list(dict.fromkeys(file_list))
    logger.info(f"{file_list=}")

    if args.protection_value:
        pv_data, pv_info = read_raster(str(args.protection_value), args.protection_value_band)
        W, H, GT, PJ = pv_info["RasterXSize"], pv_info["RasterYSize"], pv_info["Transform"], pv_info["Projection"]
        # make 1D
        pv = pv_data.ravel()
        logger.info(f"{pv_data.dtype=}, {pv_data.shape=}, {pv_info=}")
        dpv = np.zeros(pv.shape, dtype=pv.dtype)
    elif args.W and args.H:
        W, H = args.W, args.H
        GT = (0, 1, 0, 0, 0, 1)
        PJ = "EPSG:4326"
        pv = np.ones(W * H, dtype=np.int32)
        dpv = np.zeros(W * H, dtype=np.int32)

    from multiprocessing import Pool, cpu_count

    with Pool(processes=cpu_count() - 1) as pool:
        results = [pool.apply_async(one_sim_work, args=(afile, pv, i)) for i, afile in enumerate(file_list)]
        for result in results:
            sdpv, si2n, sid = result.get()
            dpv[si2n] += sdpv
    # scale
    dpv = dpv / len(file_list)
    print(f"{dpv=}")
    # write
    dst_ds = gdal.GetDriverByName("GTiff").Create(args.dpv_filename, W, H, 1, gdal.GDT_Float32)
    # get driver by name to create a geo tiff raster
    dst_ds.SetGeoTransform(GT)
    dst_ds.SetProjection(PJ)
    band = dst_ds.GetRasterBand(1)
    band.SetUnitType("protection_value")
    if 0 != band.SetNoDataValue(0):
        feedback.pushWarning(f"Set No Data failed for {self.OUT_R}")
    if 0 != band.WriteArray(np.float32(dpv.reshape(H, W))):
        feedback.pushWarning(f"WriteArray failed for {self.OUT_R}")
    band.FlushCache()
    dst_ds.FlushCache()
    dst_ds = None
    return 0


if __name__ == "__main__":
    sys.exit(main())


def no():
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
