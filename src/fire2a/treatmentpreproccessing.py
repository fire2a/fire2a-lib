#!python3
"""
Fuelbreak treatment pre proccessing functions
"""
__author__ = "David Palacios Meneses"
__revision__ = "$Format:%H$"
from numpy import insert as npinsert
from pandas import DataFrame


def bin_to_nod(solution: list, filename="treatment.csv") -> None:
    """Transforms a binary array to a Cell2Fire firebreak file

    Args:
        solution (list[int]): List with id of treatment cell
        filename (string): Path and name of the destiny folder (should be a .csv)

    Returns:
        None
        Appends to filename.csv with columns "Year Number", "Cell Numbers"

    Raises:
        ValueError: If the extension of the filename is not a csv
    """
    if filename[-4:] != ".csv":
        raise ValueError("Extension must be .csv or .txt")
    nod = [i + 1 for i in solution]
    datos = [npinsert(nod, 0, 1)]
    if len(nod) == 0:
        cols = ["Year  Number"]
    else:
        colu = ["Year Number", "Cell Numbers"]
        col2 = [""] * (len(nod) - 1)
        cols = colu + col2
    df = DataFrame(datos, columns=cols)
    df.to_csv(filename, index=None, mode="a")


if __name__ == "__main__":
    vector_test = [1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1]
    name = "test_treatment.csv"
    bin_to_nod(vector_test, name)
