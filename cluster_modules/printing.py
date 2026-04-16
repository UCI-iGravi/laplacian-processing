import numpy as np

def print_coord_list(label, coords, grid_shape, precision=1):
    """
    Print a list of sparse coordinates (either moving or fixed) in grid format.

    Args:
        label: string to label the print (e.g., "Moving Points")
        coords: (N, 2) array of coordinates [y, x]
        grid_shape: (Y, X) dimensions of the grid
        precision: number of decimal digits
    """
    Y, X = grid_shape
    coord_map = {tuple(p): i for i, p in enumerate(coords)}
    print(f"\n{label}:\n")
    for y in range(Y):
        row = []
        for x in range(X):
            key = (y, x)
            if key in coord_map:
                i = coord_map[key]
                val = coords[i]
                entry = f"({val[0]:.{precision}f},{val[1]:.{precision}f})"
            else:
                entry = " " * (7 + 2 * precision)
            row.append(entry)
        print("  ".join(row))


def print_fixed_at_moving(label, moving_points, fixed_points, grid_shape, precision=1):
    """
    Print a grid where the positions of moving_points are filled with the corresponding fixed_points.
    All other grid cells are blank.

    Args:
        label: string to label the printout
        moving_points: (N, 2) array of [y, x]
        fixed_points: (N, 2) array of [y', x']
        grid_shape: (Y, X)
        precision: decimal digits to display
    """
    Y, X = grid_shape
    print(f"\n{label}:\n")
    move_to_fixed = {tuple(m): fixed_points[i] for i, m in enumerate(moving_points)}
    for y in range(Y):
        row = []
        for x in range(X):
            key = (y, x)
            if key in move_to_fixed:
                val = move_to_fixed[key]
                entry = f"({val[0]:.{precision}f},{val[1]:.{precision}f})"
            else:
                entry = " " * (7 + 2 * precision)
            row.append(entry)
        print("  ".join(row))


def print_interpolated_coords_grid(mapped_coords, precision=1):
    """
    Print the interpolated coordinate grid in [y', x'] format.

    Args:
        mapped_coords: (Y, X, 2) interpolated coordinates
        precision: decimal digits to show
    """
    Y, X, _ = mapped_coords.shape
    print("\nInterpolated Coordinates Grid [y', x']:\n")
    for y in range(Y):
        row = []
        for x in range(X):
            yv, xv = mapped_coords[y, x]
            entry = f"({yv:.{precision}f},{xv:.{precision}f})"
            row.append(entry)
        print("  ".join(row))
