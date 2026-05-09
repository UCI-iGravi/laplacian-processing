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


def jacobian_plot(d, pt=None, figsize=(10, 5), title=None,
                  xlim=None, ylim=None,
                  binarize_negatives=False,
                  moving_pts=None, fixed_pts=None,
                  paper=False):
    """Plot the Jacobian determinant field.

    Set ``paper=True`` for the clean paper-figure styling
    (dynamic color norm, no colorbar, no ticks, larger fonts).
    """
    jdet_field = d.jdet_field.copy()

    if binarize_negatives:
        jdet_field[jdet_field < 0] = -1

    plt.figure(figsize=figsize)
    if paper:
        norm = mcolors.TwoSlopeNorm(vmin=jdet_field.min(), vcenter=0, vmax=1)
        plt.imshow(jdet_field, cmap='seismic', norm=norm)
    else:
        plt.imshow(jdet_field, cmap='seismic', vmin=-1, vmax=1)
        plt.colorbar()

    text_fs = 20 if paper else 7
    if xlim is not None and ylim is not None:
        for y in range(ylim[0] + 1, ylim[1]):
            for x in range(xlim[0] + 1, xlim[1]):
                if jdet_field[y, x] < 0:
                    plt.text(x, y, f"{d.jdet_field[y, x]:.2f}", fontsize=text_fs, ha='center', va='center', color='red', fontweight='bold')
                else:
                    plt.text(x, y, f"{d.jdet_field[y, x]:.2f}", fontsize=text_fs, ha='center', va='center', color='black')

    if pt is not None:
        plt.scatter(pt[1], pt[0], c='green', s=20)
        plt.scatter(pt[1] + d.deformation[2, 0, pt[0], pt[1]],
                    pt[0] + d.deformation[1, 0, pt[0], pt[1]],
                    c='violet', s=20)
        plt.title(f"Moving image, marked px {pt}")
    else:
        plt.title("Jacobian Determinant Field")
    if title:
        plt.title(title, fontsize=30) if paper else plt.title(title)
    if xlim:
        plt.xlim(xlim)
        if not paper:
            plt.xticks(ticks=np.arange(xlim[0], xlim[1], 1), labels=np.arange(xlim[0], xlim[1], 1))
    if ylim:
        plt.ylim(ylim)
        if not paper:
            plt.yticks(ticks=np.arange(ylim[0], ylim[1], 1), labels=np.arange(ylim[0], ylim[1], 1))

    if moving_pts is not None:
        plt.scatter(moving_pts[:, 2], moving_pts[:, 1], c='green', s=10, label='Moving Points')
    if fixed_pts is not None:
        plt.scatter(fixed_pts[:, 2], fixed_pts[:, 1], c='violet', s=10, label='Fixed Points')

    if paper:
        plt.xticks([])
        plt.yticks([])
    else:
        plt.grid(True, which='both', linestyle='--', linewidth=0.25)
    plt.gca().invert_yaxis()
    plt.show()


def find_index(arr, values):
    """
    arr: np.ndarray of shape (N, 3)
    values: tuple or list of 3 values to match
    Returns: index or -1 if not found
    """
    matches = np.where(np.all(arr == values, axis=1))[0]
    return matches[0] if len(matches) > 0 else -1
