import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def show(d, px, figsize: tuple = (10, 5), fontsize: int = 6, show_text=True, show_axis=True, show_correspondence=True, show_orientation=False, binarize_negatives=False):
    """Show the deformation field and the Jacobian determinant field.

    Args:
        title (str, optional): Title of the plot. Defaults to None.
        figsize (tuple, optional): Size of the plot. Defaults to (10, 5).
        fontsize (int, optional): Font size of the text. Defaults to 6.
        show_text (bool, optional): Whether to show text on the plot. Defaults to True.
        show_axis (bool, optional): Whether to show the normalized axis text or use the default. Defaults to True.
    """
    #norm = mcolors.TwoSlopeNorm(vmin=min(self.jdet_field.min(), -1), vcenter=0, vmax=self.jdet_field.max())
    norm = mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)

    jdet_field = d.jdet_field.copy()

    #if binarize_negatives:
    #    jdet_field[jdet_field < 0] = -1
    #    jdet_field[jdet_field > 0] = 1  # Clip values to [-1, 1] for better visualization

    f = plt.figure(figsize=figsize)
    plt.imshow(jdet_field, cmap="seismic", norm=norm)
    plt.colorbar()

    plt.scatter(px[1], px[0], c='green', s=5)
    plt.scatter(px[1] + d.deformation[2, 0, px[0], px[1]], px[0] + d.deformation[1, 0, px[0], px[1]], c='violet', s=5)
    plt.title(f"Moving image, marked px {px}")
    plt.show()
    f.clear()
    plt.close(f)


def find_index(arr, values):
    """
    arr: np.ndarray of shape (N, 3)
    values: tuple or list of 3 values to match
    Returns: index or -1 if not found
    """
    matches = np.where(np.all(arr == values, axis=1))[0]
    return matches[0] if len(matches) > 0 else -1
