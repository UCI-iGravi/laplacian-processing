import matplotlib.pyplot as plt
import numpy as np

def plot_point_correspondences(points1, points2, 
                              points3=None, points4=None,
                              labels=None, colors=None, 
                              line_alpha=0.3, point_size=30, figsize=(12, 8),
                              title="Point Correspondences", show_indices=False):
    """
    Plot two sets of points with lines connecting corresponding points by index.

    Parameters:
    -----------
    points1 : numpy.ndarray
        First set of points, shape (N, 2) with (x, y) coordinates
    points2 : numpy.ndarray  
        Second set of points, shape (N, 2) with (x, y) coordinates
    labels : tuple, optional
        Labels for the two point sets, e.g., ('Set A', 'Set B')
    colors : tuple, optional
        Colors for the two point sets, e.g., ('red', 'blue')
    line_alpha : float, default=0.3
        Transparency of correspondence lines
    point_size : int, default=30
        Size of the plotted points
    figsize : tuple, default=(12, 8)
        Figure size
    title : str, default="Point Correspondences"
        Plot title
    show_indices : bool, default=False
        Whether to show point indices as text annotations

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """

    # Ensure points are numpy arrays
    points1 = np.array(points1)
    points2 = np.array(points2)

    # Check that both sets have the same number of points
    if len(points1) != len(points2):
        print(f"Warning: Point sets have different lengths ({len(points1)} vs {len(points2)})")
        min_len = min(len(points1), len(points2))
        points1 = points1[:min_len]
        points2 = points2[:min_len]
        print(f"Using first {min_len} points from each set")

    # Set default labels and colors
    if labels is None:
        labels = ('Points Set 1', 'Points Set 2', 'Points Set 3', 'Points Set 4')
    if colors is None:
        colors = ('red', 'blue', 'green', 'orange')

    # Create the plot
    fig = plt.figure(figsize=figsize)

    # Draw correspondence lines first (so they appear behind points)
    for i in range(len(points1)):
        plt.plot([points1[i, 0], points2[i, 0]], 
                [points1[i, 1], points2[i, 1]], 
                'gray', alpha=line_alpha, linewidth=1, zorder=1)

    # Draw expansion lines first (so they appear behind points)
    if points3 is not None and points4 is not None:
        for i in range(len(points3)):
            plt.plot([points3[i, 0], points4[i, 0]], 
                    [points3[i, 1], points4[i, 1]], 
                    'red', alpha=line_alpha, linewidth=1, zorder=1)

    # Plot the points
    scatter1 = plt.scatter(points1[:, 0], points1[:, 1], 
                         c=colors[0], s=point_size, alpha=0.8, 
                         label=labels[0], zorder=3, edgecolors='black', linewidth=0.5)
    scatter2 = plt.scatter(points2[:, 0], points2[:, 1], 
                         c=colors[1], s=point_size, alpha=0.8, 
                         label=labels[1], zorder=3, edgecolors='black', linewidth=0.5)
    if points3 is not None:
        scatter3 = plt.scatter(points3[:, 0], points3[:, 1], 
                             c=colors[2], s=point_size, alpha=0.8, 
                             label=labels[2], zorder=3, edgecolors='black', linewidth=0.5)
    if points4 is not None:
        scatter4 = plt.scatter(points4[:, 0], points4[:, 1], 
                             c=colors[3], s=point_size, alpha=0.8, 
                             label=labels[3], zorder=3, edgecolors='black', linewidth=0.5)

    # Add point indices if requested
    if show_indices:
        for i in range(len(points1)):
            plt.annotate(f'{i}', (points1[i, 0], points1[i, 1]), 
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, alpha=0.7, color=colors[0])
            plt.annotate(f'{i}', (points2[i, 0], points2[i, 1]), 
                       xytext=(5, 5), textcoords='offset points', 
                       fontsize=8, alpha=0.7, color=colors[1])

    # Formatting
    plt.title(title, fontsize=20)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()  # Invert y-axis to match image coordinates
    plt.show()

    return fig
