import numpy as np
import SimpleITK as sitk


def sitk_jacobian_determinant(deformation: np.ndarray, transpose_displacements=True):
    '''
    deformation - 3, X, Y, Z, 3
    '''
    deformation = np.transpose(deformation, [1,2,3,0])
    #print("SITK deformation shape:", deformation.shape)
    if transpose_displacements:
        deformation = deformation[:, :, :, [2,1,0]]
    #print(deformation[350, 200, 200, :])
    sitk_displacement_field = sitk.GetImageFromArray(deformation, isVector=True)
    jacobian_det_volume = sitk.DisplacementFieldJacobianDeterminant(sitk_displacement_field)
    jacobian_det_np_arr = sitk.GetArrayFromImage(jacobian_det_volume)
    #n_count = np.sum(jacobian_det_np_arr < 0)
    return jacobian_det_np_arr



def surrounding_points(coord: tuple, deformation: np.ndarray, jacobian_det: np.ndarray):
    """
    Print out the surrounding points of a specific coordinate in the deformation field
    along with their displacement vectors and Jacobian determinants.
    
    Parameters:
    - coord: The coordinate of the point (z, y, x).
    - deformation: The deformation field (3D vector field).
    - jacobian_det: The Jacobian determinant volume.
    """
    curr_coord = coord
    # Get the coordinates of the surrounding points
    curr_coord_up = (curr_coord[0], curr_coord[1] - 1, curr_coord[2])
    curr_coord_down = (curr_coord[0], curr_coord[1] + 1, curr_coord[2])
    curr_coord_left = (curr_coord[0], curr_coord[1], curr_coord[2] - 1)
    curr_coord_right = (curr_coord[0], curr_coord[1], curr_coord[2] + 1)
    #curr_coord_prev = (curr_coord[0] - 1, curr_coord[1], curr_coord[2])
    #curr_coord_next = (curr_coord[0] + 1, curr_coord[1], curr_coord[2])

    # Get the displacement vectors
    curr_vector = deformation[:, curr_coord[0], curr_coord[1], curr_coord[2]]
    left_vector = deformation[:, curr_coord_left[0], curr_coord_left[1], curr_coord_left[2]]
    right_vector = deformation[:, curr_coord_right[0], curr_coord_right[1], curr_coord_right[2]]
    up_vector = deformation[:, curr_coord_up[0], curr_coord_up[1], curr_coord_up[2]]
    down_vector = deformation[:, curr_coord_down[0], curr_coord_down[1], curr_coord_down[2]]
    #prev_vector = deformation[:, curr_coord_prev[0], curr_coord_prev[1], curr_coord_prev[2]]
    #next_vector = deformation[:, curr_coord_next[0], curr_coord_next[1], curr_coord_next[2]]

    # Get the jacobian determinants
    curr_det = jacobian_det[curr_coord[0], curr_coord[1], curr_coord[2]]
    left_det = jacobian_det[curr_coord_left[0], curr_coord_left[1], curr_coord_left[2]]
    right_det = jacobian_det[curr_coord_right[0], curr_coord_right[1], curr_coord_right[2]]
    up_det = jacobian_det[curr_coord_up[0], curr_coord_up[1], curr_coord_up[2]]
    down_det = jacobian_det[curr_coord_down[0], curr_coord_down[1], curr_coord_down[2]]
    #prev_det = jacobian_det[curr_coord_prev[0], curr_coord_prev[1], curr_coord_prev[2]]
    #next_det = jacobian_det[curr_coord_next[0], curr_coord_next[1], curr_coord_next[2]]

    # Print out information
    print("Current point:", curr_coord)
    print("Displacement vectors (z, y, x)")
    print("\tCurrent displacement vector at", curr_coord, ":\t\t\t", curr_vector)
    print("\t\tNew position:", curr_coord + curr_vector)
    print("\tLeft displacement vector at", curr_coord_left, ":\t\t\t", left_vector)
    print("\t\tNew position:", curr_coord_left + left_vector)
    print("\tRight displacement vector at", curr_coord_right, ":\t\t\t", right_vector)
    print("\t\tNew position:", curr_coord_right + right_vector)
    print("\tUp displacement vector at", curr_coord_up, ":\t\t\t\t", up_vector)
    print("\t\tNew position:", curr_coord_up + up_vector)
    print("\tDown displacement vector at", curr_coord_down, ":\t\t\t", down_vector)
    print("\t\tNew position:", curr_coord_down + down_vector)
    #print("\tPrevious section displacement vector at", curr_coord_prev, ":\t", prev_vector)
    #print("\t\tNew position:", curr_coord_prev + prev_vector)
    #print("\tNext section displacement vector at", curr_coord_next, ":\t\t", next_vector)
    #print("\t\tNew position:", curr_coord_next + next_vector)

    print("\nDeterminants")
    print("\tCurrent point - Jacobian determinant at", curr_coord, ":\t\t\t", curr_det)
    print("\tLeft Jacobian determinant at", curr_coord_left, ":\t\t\t", left_det)
    print("\tRight Jacobian determinant at", curr_coord_right, ":\t\t\t", right_det)
    print("\tUp Jacobian determinant at", curr_coord_up, ":\t\t\t\t", up_det)
    print("\tDown Jacobian determinant at", curr_coord_down, ":\t\t\t", down_det)
    #print("\tPrevious section Jacobian determinant at", curr_coord_prev, ":\t", prev_det)
    #print("\tNext section Jacobian determinant at", curr_coord_next, ":\t\t", next_det)


def jacobian_determinant_central(dvf, pt=None):
    """
    Compute the Jacobian determinant of a 2D deformation vector field using central differences.
    dvf: numpy array of shape (3, 1, H, W), where dvf[1, 0, ...] is dy, dvf[2, 0, ...] is dx
    Returns: jdet (H, W) array
    """
    # Extract displacement fields
    dy = dvf[1, 0]  # shape (H, W)
    dx = dvf[2, 0]  # shape (H, W)

    # Compute gradients using central differences
    # For interior points: (f[i+1] - f[i-1]) / 2
    # For borders: use forward/backward difference

    # ∂dx/∂x
    dx_x = np.zeros_like(dx)
    dx_x[:, 1:-1] = (dx[:, 2:] - dx[:, :-2]) / 2
    dx_x[:, 0] = dx[:, 1] - dx[:, 0]
    dx_x[:, -1] = dx[:, -1] - dx[:, -2]

    # ∂dx/∂y
    dx_y = np.zeros_like(dx)
    dx_y[1:-1, :] = (dx[2:, :] - dx[:-2, :]) / 2
    dx_y[0, :] = dx[1, :] - dx[0, :]
    dx_y[-1, :] = dx[-1, :] - dx[-2, :]

    # ∂dy/∂x
    dy_x = np.zeros_like(dy)
    dy_x[:, 1:-1] = (dy[:, 2:] - dy[:, :-2]) / 2
    dy_x[:, 0] = dy[:, 1] - dy[:, 0]
    dy_x[:, -1] = dy[:, -1] - dy[:, -2]

    # ∂dy/∂y
    dy_y = np.zeros_like(dy)
    dy_y[1:-1, :] = (dy[2:, :] - dy[:-2, :]) / 2
    dy_y[0, :] = dy[1, :] - dy[0, :]
    dy_y[-1, :] = dy[-1, :] - dy[-2, :]

    # Show each step
    if pt is not None:
        print("dx_x (∂dx/∂x):\n", dx_x[pt[0], pt[1]])
        print("dx_y (∂dx/∂y):\n", dx_y[pt[0], pt[1]])
        print("dy_x (∂dy/∂x):\n", dy_x[pt[0], pt[1]])
        print("dy_y (∂dy/∂y):\n", dy_y[pt[0], pt[1]])

    # Jacobian determinant formula for 2D deformation:
    # J = | 1 + dy_y   dy_x |
    #     | dx_y     1 + dx_x |
    # det(J) = (1 + dx_x) * (1 + dy_y) - dx_y * dy_x
    jdet = (1 + dx_x) * (1 + dy_y) - dx_y * dy_x

    print("Jacobian determinant:\n", jdet[pt[0], pt[1]])
    return jdet


def jacobian_determinant_finite(dvf, pt=None):
    """
    Compute the Jacobian determinant of a 2D deformation vector field using forward finite differences.
    dvf: numpy array of shape (3, 1, H, W), where dvf[1, 0, ...] is dy, dvf[2, 0, ...] is dx
    Returns: jdet (H, W) array
    """
    dy = dvf[1, 0]  # (H, W)
    dx = dvf[2, 0]  # (H, W)
    H, W = dx.shape

    # ∂dx/∂x (forward difference)
    dx_x = np.zeros_like(dx)
    dx_x[:, :-1] = dx[:, 1:] - dx[:, :-1]
    dx_x[:, -1] = dx[:, -1] - dx[:, -2]

    # ∂dx/∂y (forward difference)
    dx_y = np.zeros_like(dx)
    dx_y[:-1, :] = dx[1:, :] - dx[:-1, :]
    dx_y[-1, :] = dx[-1, :] - dx[-2, :]

    # ∂dy/∂x (forward difference)
    dy_x = np.zeros_like(dy)
    dy_x[:, :-1] = dy[:, 1:] - dy[:, :-1]
    dy_x[:, -1] = dy[:, -1] - dy[:, -2]

    # ∂dy/∂y (forward difference)
    dy_y = np.zeros_like(dy)
    dy_y[:-1, :] = dy[1:, :] - dy[:-1, :]
    dy_y[-1, :] = dy[-1, :] - dy[-2, :]

    # Show each step
    if pt is not None:
        print("dx_x (∂dx/∂x):\n", dx_x[pt[0], pt[1]])
        print("dx_y (∂dx/∂y):\n", dx_y[pt[0], pt[1]])
        print("dy_x (∂dy/∂x):\n", dy_x[pt[0], pt[1]])
        print("dy_y (∂dy/∂y):\n", dy_y[pt[0], pt[1]])

    # Jacobian determinant formula for 2D deformation:
    # J = | 1 + dy_y   dy_x |
    #     | dx_y     1 + dx_x |
    # det(J) = (1 + dx_x) * (1 + dy_y) - dx_y * dy_x
    jdet = (1 + dx_x) * (1 + dy_y) - dx_y * dy_x

    if pt is not None:
        print("Jacobian determinant:\n", jdet[pt[0], pt[1]])
    return jdet


def compute_jacobian_det(dvf):
    dy = dvf[1, 0]
    dx = dvf[2, 0]
    dx_x = np.gradient(dx, axis=1)
    dx_y = np.gradient(dx, axis=0)
    dy_x = np.gradient(dy, axis=1)
    dy_y = np.gradient(dy, axis=0)
    return (1 + dx_x) * (1 + dy_y) - dx_y * dy_x
