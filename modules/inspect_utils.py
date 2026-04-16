import numpy as np

from modules import jacobian


def get_point_info(dvf, pt, scale=1):
    """
    Get the current point's deformation and Jdet information.
    Args:
        d_original: Data object containing the deformation field and Jdet field.
        pt: Tuple (y, x) coordinates of the point in the image.
    Returns:
        current_jdet: Jacobian determinant at the current point.
        current: Deformation vector at the current point.
        down: Deformation vector at the point below.
        up: Deformation vector at the point above.
        left: Deformation vector at the point to the left.
        right: Deformation vector at the point to the right.
    """
    pt = np.array(pt)
    current_jdet = jacobian.sitk_jacobian_determinant(dvf)[0][pt[0], pt[1]]
    current = dvf[:, 0, pt[0], pt[1]]
    down = dvf[:, 0, pt[0]-1, pt[1]]
    up = dvf[:, 0, pt[0]+1, pt[1]]
    left = dvf[:, 0, pt[0], pt[1]-1]
    right = dvf[:, 0, pt[0], pt[1]+1]

    print("Mapping in fixed space:")
    print(f"Current point: {np.array(pt) / scale}")
    print("Mapped point:", np.array((pt[0] / scale + current[1], pt[1] / scale + current[2])))
    print("Jdet:", current_jdet)

    print("Current displacement:", current)
    #print("\tDown:", down)
    #print("\tUp:", up)
    #print("\tLeft:", left)
    #print("\tRight:", right)
    print()


def get_expanding_coordinates(jdet_field, threshold=1, verbose=True):
    """
    Get the coordinates of expanding Jacobian determinants in the grid.

    Args:
        show (bool): Whether to print the coordinates of the expanding Jacobian determinants.

    Returns:
        np.ndarray: Array of coordinates of expanding Jacobian determinants.
    """
    # Get coordinate of expanding Jacobian determinants
    found_indices = np.argwhere(jdet_field > threshold)
    if verbose:
        print("Expanding Jacobian determinants:")
        for y, x in found_indices:
            print(f"({y}, {x})")
    print(f"Found {len(found_indices)} expanding Jacobian determinants with threshold {threshold}.")
    return found_indices


def get_contracting_coordinates(jdet_field, threshold=1, verbose=True):
    """
    Get the coordinates of contracting Jacobian determinants in the grid.

    Args:
        show (bool): Whether to print the coordinates of the contracting Jacobian determinants.

    Returns:
        np.ndarray: Array of coordinates of expanding contracting determinants.
    """
    # Get coordinate of expanding Jacobian determinants
    found_indices = np.argwhere(jdet_field < threshold)
    if verbose:
        print("Contracting Jacobian determinants:")
        for y, x in found_indices:
            print(f"({y}, {x})")
    print(f"Found {len(found_indices)} contracting Jacobian determinants with threshold {threshold}.")
    return found_indices


def get_mapped_point(pt, dvf):
    """
    Get the mapped point in the fixed space given a point in the moving space and a deformation vector field.

    Args:
        pt (tuple): Coordinates of the point in the moving space (y, x).
        dvf (np.ndarray): Deformation vector field of shape (3, 1, H, W).

    Returns:
        tuple: Mapped coordinates in the fixed space.
    """
    if len(pt) == 2:
        y, x = pt
    elif len(pt) == 3:
        y, x = pt[1], pt[2]

    mapped_y = y + dvf[1, 0, y, x]
    mapped_x = x + dvf[2, 0, y, x]
    return np.array((mapped_y, mapped_x))


def get_displacement(pt, dvf):
    """
    Get the displacement vector at a point in the moving space.

    Args:
        pt (tuple): Coordinates of the point in the moving space (y, x).
        dvf (np.ndarray): Deformation vector field of shape (3, 1, H, W).

    Returns:
        np.ndarray: Displacement vector at the point.
    """
    if len(pt) == 2:
        y, x = pt
    elif len(pt) == 3:
        y, x = pt[1], pt[2]

    return dvf[:, 0, y, x]
