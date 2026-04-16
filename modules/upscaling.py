import numpy as np
from scipy.ndimage import zoom
from scipy.optimize import linear_sum_assignment


def upscale_dvf_preserve(dvf, scale=3):
    """
    Upscales a DVF (3, 1, H, W) by a factor with linear interpolation.
    Ensures original values at original locations are preserved.

    Parameters:
        dvf (np.ndarray): Deformation vector field of shape (3, 1, H, W)
        scale (int): Upscaling factor (e.g., 3)

    Returns:
        np.ndarray: Upscaled DVF of shape (3, 1, scale*H, scale*W)
    """
    z, c, H, W = dvf.shape
    # Linearly interpolate entire DVF
    dvf_up = zoom(dvf, zoom=(1, 1, scale, scale), order=1)

    # Replace exact original points at the scaled locations (0, scale, 2*scale, ...)
    for y in range(H):
        for x in range(W):
            y_up = y * scale
            x_up = x * scale
            dvf_up[:, 0, y_up, x_up] = dvf[:, 0, y, x]

    return dvf_up


def upscale_dvf_linear_y(dvf, scale=3):
    """
    Upscale a DVF (3, 1, H, W) by a factor, using strict linear interpolation in y-direction.
    Only works if displacements are only in y (i.e., dvf[2, ...] == 0).
    """
    z, c, H, W = dvf.shape
    up_H, up_W = H * scale, W * scale
    dvf_up = np.zeros((z, c, up_H, up_W), dtype=dvf.dtype)

    for x in range(W):
        for y in range(H - 1):
            # Original displacements at y and y+1
            v0 = dvf[:, 0, y, x]
            v1 = dvf[:, 0, y + 1, x]
            for s in range(scale):
                alpha = s / scale
                interp = (1 - alpha) * v0 + alpha * v1
                dvf_up[:, 0, y * scale + s, x * scale] = interp
        # Last point: just copy
        dvf_up[:, 0, (H - 1) * scale, x * scale] = dvf[:, 0, H - 1, x]

    # Optionally, fill in x-direction by copying (if you want a column to be constant)
    for x in range(W):
        for y in range(up_H):
            for s in range(1, scale):
                dvf_up[:, 0, y, x * scale + s] = dvf_up[:, 0, y, x * scale]

    return dvf_up


def assign_unique_correspondences(fixed_coords, displacements, moving_coords):
    """
    Assign each fixed pixel to a unique moving pixel based on closest displaced location.
    Args:
        fixed_coords: (N, 2) array of (y, x) for fixed pixels
        displacements: (N, 2) array of (dy, dx) for each fixed pixel
        moving_coords: (M, 2) array of (y, x) for moving pixels (candidates)
    Returns:
        assignment: (N,) array, assignment[i] = index in moving_coords assigned to fixed_coords[i]
    """
    # Compute displaced locations for each fixed pixel
    displaced = fixed_coords + displacements

    # Compute cost matrix: distance from each displaced fixed pixel to each moving pixel
    cost_matrix = np.linalg.norm(displaced[:, None, :] - moving_coords[None, :, :], axis=2)

    # Solve the assignment problem (minimize total distance, unique assignments)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # row_ind: indices in fixed_coords, col_ind: assigned indices in moving_coords

    # For each fixed pixel, get the assigned moving pixel index
    assignment = np.full(len(fixed_coords), -1, dtype=int)
    assignment[row_ind] = col_ind
    return assignment


def generate_additional_correspondences_from_upscaled_dvf(
    dvf_up, jacobian_up, scale=3, assigned_fixed_points=None,
    existing_moving_pts=None, y_only=True, jacobian_thresh=0.0
):
    H_up, W_up = jacobian_up.shape
    H_orig, W_orig = H_up // scale, W_up // scale

    if assigned_fixed_points is None:
        assigned_fixed_points = set()
    if existing_moving_pts is not None:
        existing_moving_coords = set(map(tuple, existing_moving_pts[:, 1:].astype(int)))
    else:
        existing_moving_coords = set()

    new_pairs = []

    for y_up in range(1, H_up - 1):
        for x_up in range(1, W_up - 1):
            if jacobian_up[y_up, x_up] <= jacobian_thresh:
                continue

            neighbors = [(y_up + dy, x_up + dx) for dy in [-1, 0, 1] for dx in [-1, 0, 1]
                         if 0 <= y_up + dy < H_up and 0 <= x_up + dx < W_up and (dy != 0 or dx != 0)]

            for ny, nx in neighbors:
                y_orig = ny / scale
                x_orig = nx / scale
                y_int = int(np.round(y_orig))
                x_int = int(np.round(x_orig))

                if not (0 <= y_int < H_orig and 0 <= x_int < W_orig):
                    continue

                moving_coord = (y_int, x_int)
                if moving_coord in existing_moving_coords:
                    continue

                disp_y = dvf_up[1, 0, ny, nx]
                disp_x = dvf_up[2, 0, ny, nx]
                if y_only:
                    disp_x = 0

                fixed_y = int(np.round(y_orig + disp_y / scale))
                fixed_x = int(np.round(x_orig + disp_x / scale))
                fixed_coord = (fixed_y, fixed_x)

                if not (0 <= fixed_y < H_orig and 0 <= fixed_x < W_orig):
                    continue

                if fixed_coord in assigned_fixed_points:
                    continue

                new_pairs.append((moving_coord, fixed_coord))
                assigned_fixed_points.add(fixed_coord)
                existing_moving_coords.add(moving_coord)

    return new_pairs
