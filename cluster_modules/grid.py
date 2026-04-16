import csv

import numpy as np
from scipy.interpolate import griddata

def boundary_points_on_grid(shape):
    """
    Generate (y, x) points on the boundary of a grid of given shape.
    Returns an (N, 2) array of (y, x) coordinates.
    """
    Y, X = shape
    points = []
    # Top and bottom rows
    for x in range(X):
        points.append((0, 0, x))
        points.append((0, Y - 1, x))
    # Left and right columns (excluding corners already added)
    for y in range(1, Y - 1):
        points.append((0, y, 0))
        points.append((0, y, X - 1))
    return np.array(points)


def export_interpolated_grid_to_csv(interp_coords, scale, filename="interpolated_grid.csv"):
    """
    Export the interpolated coordinates grid to a CSV file.

    Args:
        interp_coords: np.ndarray, shape (2, Y, X), interpolated coordinates.
        scale: int or float, scaling factor used for the grid.
        filename: str, output CSV file name.
    """
    Y, X = interp_coords.shape[1], interp_coords.shape[2]
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Write column headers (x-axis)
        header = ["y/x"] + [f"{x/scale:.2f}" for x in range(X)]
        writer.writerow(header)
        # Write each row
        for y in range(Y):
            row = [f"{y/scale:.2f}"]
            for x in range(X):
                yv = interp_coords[1, y, x] / scale
                xv = interp_coords[0, y, x] / scale
                if np.isnan(yv) or np.isnan(xv):
                    entry = ""
                else:
                    entry = f"({yv:.1f},{xv:.1f})"
                row.append(entry)
            writer.writerow(row)
    print(f"Exported interpolated grid to {filename}")


def interpolate_coordinates(moving_points, fixed_points, grid_shape=(10, 10), scale=1):
    """
    Interpolates fixed point coordinates over a full grid from sparse moving-to-fixed correspondences.
    Optionally upscales the grid and points by an integer scale factor before interpolation.

    Args:
        moving_points: (N, 2) array of [y, x] source positions.
        fixed_points: (N, 2) array of [y', x'] target positions.
        grid_shape: (Y, X) shape of the output grid.
        scale: int, optional. If >1, upscales the grid and points by this factor before interpolation.

    Returns:
        interp_coords: (2, Y*scale, X*scale) array of interpolated [y', x'] at each grid coordinate.
    """
    Y, X = grid_shape
    if scale > 1:
        # Upscale grid shape
        Y_up, X_up = Y * scale, X * scale
        # Upscale points
        moving_points_up = moving_points * scale
        fixed_points_up = fixed_points * scale
        # New grid
        grid_y, grid_x = np.meshgrid(np.arange(Y_up), np.arange(X_up), indexing='ij')
        grid_coords = np.stack([grid_y.ravel(), grid_x.ravel()], axis=-1)
    else:
        Y_up, X_up = Y, X
        moving_points_up = moving_points
        fixed_points_up = fixed_points
        grid_y, grid_x = np.meshgrid(np.arange(Y), np.arange(X), indexing='ij')
        grid_coords = np.stack([grid_y.ravel(), grid_x.ravel()], axis=-1)

    interpolated_coords = np.zeros((Y_up * X_up, 2), dtype=np.float32)

    for i in range(2):  # y', x'
        interp = griddata(
            moving_points_up, fixed_points_up[:, i],
            grid_coords, method='linear', fill_value=np.nan
        )
        fallback = np.nan
        interpolated_coords[:, i] = np.where(np.isnan(interp), fallback, interp)

    interp_coords = interpolated_coords.reshape(Y_up, X_up, 2)
    # Move last axis to first: (2, Y_up, X_up)
    return np.moveaxis(interp_coords, -1, 0)  # (2, Y_up, X_up) shape for [y', x'] coordinates


def upscale_grid_values(grid, scale=3, order=1):
    """
    Upscale a 2D grid of values by a given scale factor using interpolation.

    Args:
        grid: (H, W) array of values (e.g., image or field).
        scale: int, upscaling factor.
        order: int, interpolation order (1=linear, 3=cubic, etc.).

    Returns:
        upscaled_grid: (H*scale, W*scale) array of interpolated values.
    """
    H, W = grid.shape
    upscaled = zoom(grid, zoom=scale, order=order)
    for y in range(H):
        for x in range(W):
            upscaled[y*scale, x*scale] = grid[y, x]
    return upscaled


def upscale_grid_with_priors(grid, scale=3):
    H, W = grid.shape
    H2, W2 = H * scale, W * scale
    upscaled = np.zeros((H2, W2), dtype=grid.dtype)

    # Copy original values to upscaled grid
    for y in range(H):
        for x in range(W):
            upscaled[y*scale, x*scale] = grid[y, x]

    # Interpolate missing values (bilinear)
    for y2 in range(H2):
        for x2 in range(W2):
            if (y2 % scale == 0) and (x2 % scale == 0):
                continue  # already filled
            # Map to original grid coordinates
            y = y2 / scale
            x = x2 / scale
            y0 = int(np.floor(y))
            x0 = int(np.floor(x))
            y1 = min(y0 + 1, H - 1)
            x1 = min(x0 + 1, W - 1)
            wy = y - y0
            wx = x - x0
            v00 = grid[y0, x0]
            v01 = grid[y0, x1]
            v10 = grid[y1, x0]
            v11 = grid[y1, x1]
            upscaled[y2, x2] = (
                (1 - wy) * (1 - wx) * v00 +
                (1 - wy) * wx * v01 +
                wy * (1 - wx) * v10 +
                wy * wx * v11
            )
    return upscaled[:-scale + 1, :-scale + 1]  # Crop to original grid size
