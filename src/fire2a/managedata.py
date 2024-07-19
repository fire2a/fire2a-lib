#!python3
"""
Previous Read Data Prometheus
"""
__author__ = "David Palacios Meneses"
__revision__ = "$Format:%H$"

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from numpy import dtype
from numpy import empty as npempty
from numpy import full as npfull
from numpy import max as npmax
from numpy import nan as npnan
from numpy import ndarray
from numpy import zeros as npzeros
from pandas import DataFrame


def Lookupdict(filename: Union[Path, str]) -> Tuple[dict, dict]:
    """Reads lookup_table.csv and creates dictionaries for the fuel types and cells' colors

    Args:
        filename (string): Path to fuel model lookup_table.csv, format is XXX_lookup_table.csv, e.g: spain_lookup_table.csv

    Returns:
        dict [int,str]: Dictionary with fuel code number as key and fuel model name as value.
        dict [int,list [int,int,int,int]]: Dictionary with fuel code number as key and list of colors in rgb as value.
    """  # fmt:skip

    aux = 1
    file = open(filename, "r")
    row = {}
    colors = {}
    all = {}

    # Read file and save colors and ftypes dictionaries
    for line in file:
        if aux > 1:
            aux += 1
            line = line.replace("-", "")
            line = line.replace("\n", "")
            line = line.replace("No", "NF")
            line = line.split(",")

            if line[3][0:3] in ["FM1"]:
                row[line[0]] = line[3][0:4]
            elif line[3][0:3] in ["Non", "NFn"]:
                row[line[0]] = "NF"
            else:
                row[line[0]] = line[3][0:3]

            colors[line[0]] = (float(line[4]) / 255.0, float(line[5]) / 255.0, float(line[6]) / 255.0, 1.0)
            all[line[0]] = line

        if aux == 1:
            aux += 1

    return row, colors


# Tuple[(list, list, int, int, list, list, int)]
# Tuple[list[Any], list[Any], int, int, list[Any], list[Any], int]
def ForestGrid(
    filename: str, Lookupdict: dict
) -> Tuple[list[int], list[str], int, int, list[dict[str, Optional[list[int]]]], ndarray[Any, dtype[Any]], float]:
    """Reads fuels.asc file and returns an array with all the cells, and grid dimension nxm

    Args:
        filename (string): Path to fuel fuel model in ascii format (fuels.asc).
        Lookupdict (int,str): Dictionary with fuel code number as key and fuel model name as value.

    Returns:
        list [int]: List of forest grid with fuel code number, where non fuel are represented as 0
        list [string]: List of forest grid with fuel code name, where non fuel are represented as NF.
        int: Number of rows of forest grid.
        int: Number of columns of forest grid.
        list [dict[str,list[int]]]: List of dictionaries that contains the neighbors of each cell in each compass rose direction
        list [list[int,int]]: List of lists that stores the x and y coordinate of each cell
        int: Size of cells in forest grid
    """  # fmt:skip

    AdjCells = []
    North = "N"
    South = "S"
    East = "E"
    West = "W"
    NorthEast = "NE"
    NorthWest = "NW"
    SouthEast = "SE"
    SouthWest = "SW"

    with open(filename, "r") as f:
        filelines = f.readlines()

    line = filelines[4].replace("\n", "")
    parts = line.split()

    if parts[0] != "cellsize":
        print("line=", line)
        raise RuntimeError("Expected cellsize on line 5 of " + filename)
    cellsize = float(parts[1])

    cells = 0
    row = 1
    trows = 0
    tcols = 0
    gridcell1 = []
    gridcell2 = []
    gridcell3 = []
    gridcell4 = []
    grid = []
    grid2 = []

    # Read the ASCII file with the grid structure
    for row in range(6, len(filelines)):
        line = filelines[row]
        line = line.replace("\n", "")
        line = " ".join(line.split())
        line = line.split(" ")
        # print(line)

        for c in line:  # range(0,len(line)-1):
            if c not in Lookupdict.keys():
                gridcell1.append("NF")
                gridcell2.append("NF")
                gridcell3.append(int(0))
                gridcell4.append("NF")
            else:
                gridcell1.append(c)
                gridcell2.append(Lookupdict[c])
                gridcell3.append(int(c))
                gridcell4.append(Lookupdict[c])
            tcols = npmax([tcols, len(line)])

        grid.append(gridcell1)
        grid2.append(gridcell2)
        gridcell1 = []
        gridcell2 = []

    # Adjacent list of dictionaries and Cells coordinates
    CoordCells = npempty([len(grid) * (tcols), 2]).astype(int)
    n = 1
    tcols += 1
    for r in range(0, len(grid)):
        for c in range(0, tcols - 1):
            # CoordCells.append([c,len(grid)-r-1])
            CoordCells[c + r * (tcols - 1), 0] = c
            CoordCells[c + r * (tcols - 1), 1] = len(grid) - r - 1

            if len(grid) > 1:

                if r == 0:
                    if c == 0:
                        AdjCells.append(
                            {
                                North: None,
                                NorthEast: None,
                                NorthWest: None,
                                South: [n + tcols - 1],
                                SouthEast: [n + tcols],
                                SouthWest: None,
                                East: [n + 1],
                                West: None,
                            }
                        )
                        n += 1
                    if c == tcols - 2:
                        AdjCells.append(
                            {
                                North: None,
                                NorthEast: None,
                                NorthWest: None,
                                South: [n + tcols - 1],
                                SouthEast: None,
                                SouthWest: [n + tcols - 2],
                                East: None,
                                West: [n - 1],
                            }
                        )
                        n += 1
                    if c > 0 and c < tcols - 2:
                        AdjCells.append(
                            {
                                North: None,
                                NorthEast: None,
                                NorthWest: None,
                                South: [n + tcols - 1],
                                SouthEast: [n + tcols],
                                SouthWest: [n + tcols - 2],
                                East: [n + 1],
                                West: [n - 1],
                            }
                        )
                        n += 1

                if r > 0 and r < len(grid) - 1:
                    if c == 0:
                        AdjCells.append(
                            {
                                North: [n - tcols + 1],
                                NorthEast: [n - tcols + 2],
                                NorthWest: None,
                                South: [n + tcols - 1],
                                SouthEast: [n + tcols],
                                SouthWest: None,
                                East: [n + 1],
                                West: None,
                            }
                        )
                        n += 1
                    if c == tcols - 2:
                        AdjCells.append(
                            {
                                North: [n - tcols + 1],
                                NorthEast: None,
                                NorthWest: [n - tcols],
                                South: [n + tcols - 1],
                                SouthEast: None,
                                SouthWest: [n + tcols - 2],
                                East: None,
                                West: [n - 1],
                            }
                        )
                        n += 1
                    if c > 0 and c < tcols - 2:
                        AdjCells.append(
                            {
                                North: [n - tcols + 1],
                                NorthEast: [n - tcols + 2],
                                NorthWest: [n - tcols],
                                South: [n + tcols - 1],
                                SouthEast: [n + tcols],
                                SouthWest: [n + tcols - 2],
                                East: [n + 1],
                                West: [n - 1],
                            }
                        )
                        n += 1

                if r == len(grid) - 1:
                    if c == 0:
                        AdjCells.append(
                            {
                                North: [n - tcols + 1],
                                NorthEast: [n - tcols + 2],
                                NorthWest: None,
                                South: None,
                                SouthEast: None,
                                SouthWest: None,
                                East: [n + 1],
                                West: None,
                            }
                        )
                        n += 1

                    if c == tcols - 2:
                        AdjCells.append(
                            {
                                North: [n - tcols + 1],
                                NorthEast: None,
                                NorthWest: [n - tcols],
                                South: None,
                                SouthEast: None,
                                SouthWest: None,
                                East: None,
                                West: [n - 1],
                            }
                        )
                        n += 1

                    if c > 0 and c < tcols - 2:
                        AdjCells.append(
                            {
                                North: [n - tcols + 1],
                                NorthEast: [n - tcols + 2],
                                NorthWest: [n - tcols],
                                South: None,
                                SouthEast: None,
                                SouthWest: None,
                                East: [n + 1],
                                West: [n - 1],
                            }
                        )
                        n += 1

            if len(grid) == 1:
                if c == 0:
                    AdjCells.append(
                        {
                            North: None,
                            NorthEast: None,
                            NorthWest: None,
                            South: None,
                            SouthEast: None,
                            SouthWest: None,
                            East: [n + 1],
                            West: None,
                        }
                    )
                    n += 1
                if c == tcols - 2:
                    AdjCells.append(
                        {
                            North: None,
                            NorthEast: None,
                            NorthWest: None,
                            South: None,
                            SouthEast: None,
                            SouthWest: None,
                            East: None,
                            West: [n - 1],
                        }
                    )
                    n += 1
                if c > 0 and c < tcols - 2:
                    AdjCells.append(
                        {
                            North: None,
                            NorthEast: None,
                            NorthWest: None,
                            South: None,
                            SouthEast: None,
                            SouthWest: None,
                            East: [n + 1],
                            West: [n - 1],
                        }
                    )
                    n += 1

    return gridcell3, gridcell4, len(grid), tcols - 1, AdjCells, CoordCells, cellsize


# Tuple[(list, list, list, list, list, list, list, list, list)]
def DataGrids(InFolder: str, NCells: int) -> Tuple[
    ndarray[Any, dtype[Any]],
    ndarray[Any, dtype[Any]],
    ndarray[Any, dtype[Any]],
    ndarray[Any, dtype[Any]],
    ndarray[Any, dtype[Any]],
    ndarray[Any, dtype[Any]],
    ndarray[Any, dtype[Any]],
    ndarray[Any, dtype[Any]],
    ndarray[Any, dtype[Any]],
]:
    """
    Reads *.asc files and returns an array per each ASCII file with the correspondant information per each cell. Currently supports 
    elevation, ascpect, slope, curing, canopy bulk density, crown base height, conifer percent dead fir, probability of ignition and foliar moisture content.

    Args:
        InFolder (string): Path to data folder.
        NCells (int): Number of cells in grid.

    Returns:
        list [float]: List of elevations of each cell
        list [float]: List of aspect of each cell
        list [float]: List of slope of each cell
        list [float]: List of curing degree of each cell
        list [float]: List of canopy bulk density of each cell
        list [float]: List of crown base height of each cell
        list [float]: List of conifer percent dead fir of each cell
        list [float]: List of ignition probability of each cell
        list [float]: List of foliar moisture content of each cell
    """  # fmt:skip
    p = Path(InFolder)
    filenames = [
        "elevation.asc",
        "saz.asc",
        "slope.asc",
        "cur.asc",
        "cbd.asc",
        "cbh.asc",
        "ccf.asc",
        "py.asc",
        "fmc.asc",
    ]
    Elevation = npfull(NCells, npnan)
    SAZ = npfull(NCells, npnan)
    PS = npfull(NCells, npnan)
    Curing = npfull(NCells, npnan)
    CBD = npfull(NCells, npnan)
    CBH = npfull(NCells, npnan)
    CCF = npfull(NCells, npnan)
    PY = npfull(NCells, npnan)
    FMC = npfull(NCells, npnan)

    for name in filenames:
        ff = p / name
        if ff.exists() == True:
            aux = 0
            with open(ff, "r") as f:
                filelines = f.readlines()

                line = filelines[4].replace("\n", "")
                parts = line.split()

                if parts[0] != "cellsize":
                    print("line=", line)
                    raise RuntimeError("Expected cellsize on line 5 of " + ff)
                cellsize = float(parts[1])

                row = 1

                # Read the ASCII file with the grid structure
                for row in range(6, len(filelines)):
                    line = filelines[row]
                    line = line.replace("\n", "")
                    line = " ".join(line.split())
                    line = line.split(" ")
                    # print(line)

                    for c in line:
                        if name == "elevation.asc":
                            Elevation[aux] = float(c)
                            aux += 1
                        if name == "saz.asc":
                            SAZ[aux] = float(c)
                            aux += 1
                        if name == "slope.asc":
                            PS[aux] = float(c)
                            aux += 1
                        if name == "cbd.asc":
                            CBD[aux] = float(c)
                            aux += 1
                        if name == "cbh.asc":
                            CBH[aux] = float(c)
                            aux += 1
                        if name == "ccf.asc":
                            CCF[aux] = float(c)
                            aux += 1
                        if name == "curing.asc":
                            Curing[aux] = float(c)
                            aux += 1
                        if name == "py.asc":
                            PY[aux] = float(c)
                            aux += 1
                        if name == "fmc.asc":
                            FMC[aux] = float(c)
                            aux += 1

        else:
            print("   No", name, "file, filling with NaN")

    return Elevation, SAZ, PS, Curing, CBD, CBH, CCF, PY, FMC


# Generates the Data.dat file (csv) from all data files (ready for the simulator)
def GenerateDat(
    GFuelType: list,
    GFuelTypeN: list,
    Elevation: list,
    PS: list,
    SAZ: list,
    Curing: list,
    CBD: list,
    CBH: list,
    CCF: list,
    PY: list,
    FMC: list,
    InFolder: str,
) -> DataFrame:
    """
    Reads forest information and generates Data.csv file

    Args:
        GFuelType (list [int]): List of forest grid with fuel code number, where non fuel are represented as 0
        GFuelTypeN (list [string]): List of forest grid with fuel code name, where non fuel are represented as NF.
        Elevation (list [float]): List of elevations of each cell
        PS (list [float]): List of slope of each cell
        SAZ (list [float]): List of aspect of each cell
        Curing (list [float]): List of curing degree of each cell
        CBD (list [float]): List of canopy bulk density of each cell
        CBH (list [float]): List of crown base height of each cell
        CCF (list [float]): List of conifer percent dead fir of each cell
        PY (list [float]): List of ignition probability of each cell
        FMC (list [float]): List of foliar moisture content of each cell
        InFolder (string): Path to data folder.

    Returns:

        Dataframe: Dataframe containing information of forest
    """  # fmt:skip
    p = Path(InFolder)
    # DF columns
    Columns = [
        "fueltype",
        "lat",
        "lon",
        "elev",
        "ws",
        "waz",
        "ps",
        "saz",
        "cur",
        "cbd",
        "cbh",
        "ccf",
        "ftypeN",
        "fmc",
        "py",
    ]

    # Dataframe
    DF = DataFrame(columns=Columns)
    DF["fueltype"] = [x for x in GFuelType]
    DF["elev"] = Elevation
    DF["ps"] = PS
    DF["saz"] = SAZ
    DF["cbd"] = CBD
    DF["cbh"] = CBH
    DF["ccf"] = CCF
    DF["py"] = PY
    DF["fmc"] = FMC
    DF["lat"] = npzeros(len(GFuelType)) + 51.621244
    DF["lon"] = npzeros(len(GFuelType)).astype(int) - 115.608378

    # Populate fuel type number
    DF["ftypeN"] = GFuelTypeN
    # print(np.asarray(GFuelTypeN).flatten())

    # Data File
    filename = p / "Data.csv"
    DF.to_csv(path_or_buf=filename, index=False, index_label=False, header=True)
    return DF


def GenDataFile(InFolder: str, Simulator: str) -> None:
    """Main function that reads information available in folder and generates Data.csv file

    Args:
        InFolder (string): Path to data folder.
        Simulator (string): Simulator version, currently only supports "K" (for kitral) and "S" (for Scott & Burgan)

    Return:
        None
    """  # fmt:skip
    p = Path(InFolder)
    if Simulator == "K":
        FBPlookup = p / "kitral_lookup_table.csv"
    elif Simulator == "S":
        FBPlookup = p / "spain_lookup_table.csv"
    else:  # beta version
        FBPlookup = p / "spain_lookup_table.csv"

    FBPDict, _ = Lookupdict(FBPlookup)

    FGrid = p / "fuels.asc"
    GFuelTypeN, GFuelType, _, _, _, _, _ = ForestGrid(FGrid, FBPDict)

    NCells = len(GFuelType)
    Elevation, SAZ, PS, Curing, CBD, CBH, CCF, PY, FMC = DataGrids(InFolder, NCells)
    GenerateDat(GFuelType, GFuelTypeN, Elevation, PS, SAZ, Curing, CBD, CBH, CCF, PY, FMC, InFolder)


if __name__ == "__main__":
    p = Path("tests")
    p_fuels = p / "fuels.asc"
    p_lookup = p / "spain_lookup_table.csv"
    GenDataFile("tests", "S")
    # check if generate data exists
