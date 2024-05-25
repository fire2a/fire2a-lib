#!python3
"""👋🌎 🌲🔥
This is the raster module docstring
"""
__author__ = "Rodrigo Mahaluf Recasens"
__revision__ = "$Format:%H$"

from scipy.sparse import lil_matrix


def adjacent_cells(grid: lil_matrix, connectivity: int = 4) -> list:
    """
    Get adjacent cells for each cell in a grid.

    Parameters:
        grid (lil_matrix): Sparse matrix representing a landscape.
        connectivity (int, optional): The type of connectivity (4 or 8). Defaults to 4.

    Returns:
        list: List of lists containing neighbors for each cell.
    """  # fmt: skip
    if connectivity == 4:
        adj_cells = adjacent_cells4(grid)
    elif connectivity == 8:
        adj_cells = adjacent_cells8(grid)
    else:
        raise ValueError("Invalid connectivity value. Use 4 or 8.")

    return adj_cells


def adjacent_cells4(grid: lil_matrix) -> list:
    """
    Get 4-connected adjacent cells for each cell in the forest grid.

    Parameters:
        grid (lil_matrix): Sparse matrix representing a landscape.

    Returns:
        list: List of lists containing 4-connected neighbors for each cell.
    """  # fmt: skip
    nrows, ncols = grid.shape
    AdjCells = []

    for i in range(nrows):
        for j in range(ncols):
            neighbors = []

            if i > 0:
                neighbors.append(grid[i - 1, j])  # Up

            if i < nrows - 1:
                neighbors.append(grid[i + 1, j])  # Down

            if j > 0:
                neighbors.append(grid[i, j - 1])  # Left

            if j < ncols - 1:
                neighbors.append(grid[i, j + 1])  # Right

            AdjCells.append(neighbors)
    return AdjCells


def adjacent_cells8(grid: lil_matrix) -> list:
    """
    Get 8-connected adjacent cells for each cell in the forest grid.

    Parameters:
        grid (lil_matrix): Sparse matrix representing a landscape.

    Returns:
        list: List of lists containing 8-connected neighbors for each cell.
    """  # fmt: skip
    nrows, ncols = grid.shape
    AdjCells = []

    for i in range(nrows):
        for j in range(ncols):
            neighbors = []

            for x in range(max(0, i - 1), min(nrows, i + 2)):
                for y in range(max(0, j - 1), min(ncols, j + 2)):
                    if x != i or y != j:
                        neighbors.append(grid[x, y])

            AdjCells.append(neighbors)
    return AdjCells


if __name__ == "__main__":

    # Create a sparse forest grid (5x5) with random values
    nrows = 5
    ncols = 5
    ncells = nrows * ncols
    id_pixel = list(range(1, ncells + 1))

    # Create a sparse forest grid from the id_pixel list
    grid = lil_matrix((nrows, ncols), dtype=int)
    for idx, value in enumerate(id_pixel):
        row = idx // ncols
        col = idx % ncols
        grid[row, col] = value

    print("grid:")
    for row in grid.toarray():
        print(row)

    # Call the adjacent_cells function to get 4-connected neighbors
    adj_cells_4 = adjacent_cells(grid, connectivity=4)
    print("\n4-Connected Neighbors:")
    for row in adj_cells_4:
        print(row)

    # Call the adjacent_cells function to get 8-connected neighbors
    adj_cells_8 = adjacent_cells(grid, connectivity=8)
    print("\n8-Connected Neighbors:")
    for row in adj_cells_8:
        print(row)
