import numpy as np


def square_to_circle_correspondences(n_points=100, square_size=1.0, circle_radius=1.0, resolution=(256, 256), padding=20):
    """
    Generates n_points correspondences from a square to a circle,
    mapped to a given image resolution with padding from the edges.
    Returns:
        square_points: (n_points, 2) array of (x, y) in image coordinates
        circle_points: (n_points, 2) array of (x, y) in image coordinates
    """
    # Square points: uniformly sample points along the square perimeter in CCW order starting from bottom left
    per_side = n_points // 4
    remainder = n_points % 4
    sides = [per_side] * 4
    for i in range(remainder):
        sides[i] += 1

    # Bottom (left to right)
    x0 = np.linspace(-square_size, square_size, sides[0], endpoint=False)
    y0 = np.full_like(x0, -square_size)
    # Right (bottom to top)
    y1 = np.linspace(-square_size, square_size, sides[1], endpoint=False)
    x1 = np.full_like(y1, square_size)
    # Top (right to left)
    x2 = np.linspace(square_size, -square_size, sides[2], endpoint=False)
    y2 = np.full_like(x2, square_size)
    # Left (top to bottom)
    y3 = np.linspace(square_size, -square_size, sides[3], endpoint=False)
    x3 = np.full_like(y3, -square_size)

    square_points = np.concatenate([
        np.stack([x0, y0], axis=1),
        np.stack([x1, y1], axis=1),
        np.stack([x2, y2], axis=1),
        np.stack([x3, y3], axis=1)
    ], axis=0)

    # Map each square point to a circle point by angle, starting at -pi/2 (bottom)
    angles = np.linspace(-np.pi/2, 3*np.pi/2, n_points, endpoint=False)
    circle_points = np.stack([
        circle_radius * np.cos(angles),
        circle_radius * np.sin(angles)
    ], axis=1)

    # Center both shapes in the image
    img_center = np.array(resolution) / 2

    def to_image_coords(points, obj_size):
        # Scale so that the largest shape fits within the padded area
        scale = (np.array(resolution) - 2 * padding) / (2 * max(square_size, circle_radius))
        img_points = points * scale + img_center
        return img_points

    square_img_points = to_image_coords(square_points, square_size)
    circle_img_points = to_image_coords(circle_points, circle_radius)

    return square_img_points, circle_img_points


def circle_to_circle_correspondences(n_points=100, inner_radius=0.5, outer_radius=1.0, resolution=(256, 256), padding=20):
    """
    Generates n_points correspondences from an inner circle to an outer circle,
    mapped to a given image resolution with padding from the edges.
    Returns:
        inner_points: (n_points, 2) array of (x, y) in image coordinates
        outer_points: (n_points, 2) array of (x, y) in image coordinates
    """
    # Sample points along the perimeter, CCW starting from bottom
    angles = np.linspace(-np.pi/2, 3*np.pi/2, n_points, endpoint=False)
    inner_points = np.stack([
        inner_radius * np.cos(angles),
        inner_radius * np.sin(angles)
    ], axis=1)
    outer_points = np.stack([
        outer_radius * np.cos(angles),
        outer_radius * np.sin(angles)
    ], axis=1)

    # Center both shapes in the image
    img_center = np.array(resolution) / 2

    def to_image_coords(points, obj_size):
        # Scale so that the largest shape fits within the padded area
        scale = (np.array(resolution) - 2 * padding) / (2 * outer_radius)
        img_points = points * scale + img_center
        return img_points

    inner_img_points = to_image_coords(inner_points, inner_radius)
    outer_img_points = to_image_coords(outer_points, outer_radius)

    return inner_img_points, outer_img_points


def create_blobby_circular_shapes(n_points=30, noise_scale=4, seed=42, plot=True):
    """
    Create two blobby, mostly circular polygon-like shapes with corresponding points.
    Ensures the points do not intersect each other on the same shape.
    Both shapes are centered around the same region.
    Returns:
        shape1_points: (N, 2) array of integers
        shape2_points: (N, 2) array of integers
    """
    np.random.seed(seed)
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    # Make radii mostly circular, with small smooth perturbations
    r1 = 50 + 4 * np.sin(2 * theta) + 2 * np.cos(3 * theta)
    r2 = 48 + 5 * np.sin(2 * theta + 0.5) + 3 * np.cos(4 * theta + 1.2)

    # Add small, smooth noise (not per-point, to avoid self-intersections)
    r1 += np.interp(theta, [0, 2*np.pi], np.random.normal(0, noise_scale, 2))
    r2 += np.interp(theta, [0, 2*np.pi], np.random.normal(0, noise_scale, 2))

    # Center both shapes at the same location
    center = np.array([128, 128])

    # Convert polar to cartesian and round to integers
    x1 = np.round(r1 * np.cos(theta) + center[0]).astype(int)
    y1 = np.round(r1 * np.sin(theta) + center[1]).astype(int)
    x2 = np.round(r2 * np.cos(theta + 0.18) + center[0]).astype(int)
    y2 = np.round(r2 * np.sin(theta + 0.18) + center[1]).astype(int)

    shape1_points = np.stack([x1, y1], axis=1)
    shape2_points = np.stack([x2, y2], axis=1)

    if plot:
        plt.figure(figsize=(8, 8))
        plt.plot(*shape1_points.T, 'o-', color='purple', label='Shape 1')
        plt.plot(*shape2_points.T, 'o-', color='orange', label='Shape 2')
        for i in range(n_points):
            plt.plot([shape1_points[i, 0], shape2_points[i, 0]],
                     [shape1_points[i, 1], shape2_points[i, 1]],
                     color='gray', alpha=0.4)
        plt.scatter(shape1_points[:, 0], shape1_points[:, 1], c='purple', s=60, edgecolors='k', zorder=3)
        plt.scatter(shape2_points[:, 0], shape2_points[:, 1], c='orange', s=60, edgecolors='k', zorder=3)
        plt.title("Blobby Circular Shapes with Corresponding Points (Centered)")
        plt.axis('equal')
        plt.legend()
        plt.show()

    return shape1_points, shape2_points

def create_crossing_lines_correspondences(n_points=80, center=(128, 128), length=180, plot=True):
    """
    Create two long crossing lines forming an X, with corresponding points.
    Both lines are centered at the same location.
    Returns:
        line1_points: (N, 2) array of integers
        line2_points: (N, 2) array of integers
    """
    # Line 1: from top-left to bottom-right
    x1 = np.linspace(center[0] - length // 2, center[0] + length // 2, n_points)
    y1 = np.linspace(center[1] - length // 2, center[1] + length // 2, n_points)
    # Line 2: from bottom-left to top-right
    x2 = np.linspace(center[0] - length // 2, center[0] + length // 2, n_points)
    y2 = np.linspace(center[1] + length // 2, center[1] - length // 2, n_points)

    # Round to integer pixel coordinates
    x1 = np.round(x1).astype(int)
    y1 = np.round(y1).astype(int)
    x2 = np.round(x2).astype(int)
    y2 = np.round(y2).astype(int)

    line1_points = np.stack([x1, y1], axis=1)
    line2_points = np.stack([x2, y2], axis=1)

    if plot:
        plt.figure(figsize=(8, 8))
        plt.plot(line1_points[:, 0], line1_points[:, 1], 'o-', color='blue', label='Line 1')
        plt.plot(line2_points[:, 0], line2_points[:, 1], 'o-', color='red', label='Line 2')
        for i in range(n_points):
            plt.plot([line1_points[i, 0], line2_points[i, 0]],
                     [line1_points[i, 1], line2_points[i, 1]],
                     color='gray', alpha=0.4)
        plt.scatter(line1_points[:, 0], line1_points[:, 1], c='blue', s=60, edgecolors='k', zorder=3)
        plt.scatter(line2_points[:, 0], line2_points[:, 1], c='red', s=60, edgecolors='k', zorder=3)
        plt.title("Crossing Lines (X) Correspondences")
        plt.axis('equal')
        plt.legend()
        plt.show()

    return line1_points, line2_points

def create_double_helix_waves(grid_size=256, num_cycles=4, amplitude_ratio=0.2, 
                             vertical_center_ratio=0.5, padding_ratio=0.1, 
                             phase_offset=np.pi, plot=True, horizontal=True):
    """
    Create two intersecting sinusoidal waves that form a double helix pattern.
    If horizontal=True, the helix is oriented horizontally (waves run vertically).
    If horizontal=False, the helix is oriented vertically (waves run horizontally, original behavior).

    Parameters:
    -----------
    grid_size : int, default=256
        Size of the square grid (grid_size x grid_size)
    num_cycles : float, default=4
        Number of complete sine wave cycles across the grid
    amplitude_ratio : float, default=0.2
        Amplitude as a ratio of grid_size 
    vertical_center_ratio : float, default=0.5
        Vertical center position as ratio of grid_size (0.5 = center)
    padding_ratio : float, default=0.1
        Padding around the waves as ratio of grid_size
    phase_offset : float, default=np.pi
        Phase difference between the two waves (π creates perfect interleaving)
    plot : bool, default=True
        Whether to create plots of the waves
    horizontal : bool, default=True
        If True, helix is horizontal (waves run vertically). If False, original vertical helix.

    Returns:
    --------
    wave1_points : numpy.ndarray
        Array of (x, y) coordinates for the first wave
    wave2_points : numpy.ndarray
        Array of (x, y) coordinates for the second wave
    grid : numpy.ndarray
        2D grid with both waves marked (wave1=1, wave2=2, intersections=3)
    """

    padding = int(grid_size * padding_ratio)
    effective_grid_size = grid_size - 2 * padding

    if horizontal:
        # Swap axes: now y is the main axis, x is the wave
        y_coords = np.arange(effective_grid_size) + padding
        frequency = 2 * np.pi * num_cycles / effective_grid_size
        amplitude = grid_size * amplitude_ratio
        horizontal_center = grid_size * vertical_center_ratio

        y_wave = np.arange(effective_grid_size)
        # First wave (standard sine, runs horizontally)
        x1_coords = amplitude * np.sin(frequency * y_wave) + horizontal_center
        # Second wave (phase shifted sine)
        x2_coords = amplitude * np.sin(frequency * y_wave + phase_offset) + horizontal_center

        # Convert to integer coordinates and ensure they stay within bounds
        x1_coords_int = np.round(x1_coords).astype(int)
        x2_coords_int = np.round(x2_coords).astype(int)
        x1_coords_int = np.clip(x1_coords_int, 0, grid_size - 1)
        x2_coords_int = np.clip(x2_coords_int, 0, grid_size - 1)

        # Create points that lie on the waves (x, y)
        wave1_points = np.column_stack((x1_coords_int, y_coords))
        wave2_points = np.column_stack((x2_coords_int, y_coords))
    else:
        # Original vertical helix
        x_coords = np.arange(effective_grid_size) + padding
        frequency = 2 * np.pi * num_cycles / effective_grid_size
        amplitude = grid_size * amplitude_ratio
        vertical_center = grid_size * vertical_center_ratio

        x_wave = np.arange(effective_grid_size)
        y1_coords = amplitude * np.sin(frequency * x_wave) + vertical_center
        y2_coords = amplitude * np.sin(frequency * x_wave + phase_offset) + vertical_center

        y1_coords_int = np.round(y1_coords).astype(int)
        y2_coords_int = np.round(y2_coords).astype(int)
        y1_coords_int = np.clip(y1_coords_int, 0, grid_size - 1)
        y2_coords_int = np.clip(y2_coords_int, 0, grid_size - 1)

        wave1_points = np.column_stack((x_coords, y1_coords_int))
        wave2_points = np.column_stack((x_coords, y2_coords_int))

    # Create a 2D grid visualization
    grid = np.zeros((grid_size, grid_size))

    # Mark wave 1 points
    for x, y in wave1_points:
        grid[y, x] += 1

    # Mark wave 2 points 
    for x, y in wave2_points:
        grid[y, x] += 2

    # Find intersection points (where both waves occupy the same pixel)
    intersections = []
    for i, (x1, y1) in enumerate(wave1_points):
        for j, (x2, y2) in enumerate(wave2_points):
            if x1 == x2 and y1 == y2:
                intersections.append((i, j, x1, y1))

    print(f"Created double helix with {len(wave1_points)} points per wave")
    print(f"Grid size: {grid_size}x{grid_size}")
    print(f"Padding: {padding} pixels on each side")
    print(f"Number of cycles: {num_cycles}")
    print(f"Phase offset: {phase_offset:.3f} radians ({phase_offset*180/np.pi:.1f} degrees)")
    if horizontal:
        print(f"Wave 1 - Y range: [{wave1_points[:,1].min()}, {wave1_points[:,1].max()}], X range: [{wave1_points[:,0].min()}, {wave1_points[:,0].max()}]")
        print(f"Wave 2 - Y range: [{wave2_points[:,1].min()}, {wave2_points[:,1].max()}], X range: [{wave2_points[:,0].min()}, {wave2_points[:,0].max()}]")
    else:
        print(f"Wave 1 - X range: [{wave1_points[:,0].min()}, {wave1_points[:,0].max()}], Y range: [{wave1_points[:,1].min()}, {wave1_points[:,1].max()}]")
        print(f"Wave 2 - X range: [{wave2_points[:,0].min()}, {wave2_points[:,0].max()}], Y range: [{wave2_points[:,1].min()}, {wave2_points[:,1].max()}]")
    print(f"Number of intersection points: {len(intersections)}")

    if plot:
        fig = plt.figure(figsize=(18, 12))

        # Plot 1: Grid view showing both waves
        plt.subplot(2, 3, 1)
        display_grid = np.zeros((grid_size, grid_size, 3))  # RGB

        for x, y in wave1_points:
            display_grid[y, x, 0] = 1.0  # Red channel
        for x, y in wave2_points:
            display_grid[y, x, 2] = 1.0  # Blue channel

        plt.imshow(display_grid, origin='lower', extent=[0, grid_size, 0, grid_size])
        plt.title(f'Double Helix Grid View\n(Red: Wave 1, Blue: Wave 2, Purple: Intersections)')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')

        # Plot 2: Both waves as line plots
        plt.subplot(2, 3, 2)
        plt.plot(wave1_points[:, 0], wave1_points[:, 1], 'r-', linewidth=2, label='Wave 1', alpha=0.8)
        plt.plot(wave2_points[:, 0], wave2_points[:, 1], 'b-', linewidth=2, label='Wave 2', alpha=0.8)
        if intersections:
            intersection_x = [pt[2] for pt in intersections]
            intersection_y = [pt[3] for pt in intersections]
            plt.scatter(intersection_x, intersection_y, c='purple', s=80, 
                       marker='o', label=f'Intersections ({len(intersections)})', 
                       zorder=5, edgecolors='black')
        plt.title('Double Helix Line View')
        plt.xlabel('X coordinate')
        plt.ylabel('Y coordinate')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 3: 3D-like perspective (using offset)
        plt.subplot(2, 3, 3)
        offset = 10
        if horizontal:
            plt.plot(wave1_points[:, 0], wave1_points[:, 1], 'r-', linewidth=3, label='Wave 1 (front)', alpha=0.9)
            plt.plot(wave2_points[:, 0] + offset, wave2_points[:, 1], 'b-', linewidth=3, label='Wave 2 (back)', alpha=0.7)
        else:
            plt.plot(wave1_points[:, 0], wave1_points[:, 1], 'r-', linewidth=3, label='Wave 1 (front)', alpha=0.9)
            plt.plot(wave2_points[:, 0] + offset, wave2_points[:, 1], 'b-', linewidth=3, label='Wave 2 (back)', alpha=0.7)
        plt.title('Pseudo-3D View\n(Offset for depth perception)')
        plt.xlabel('X coordinate (+ offset)')
        plt.ylabel('Y coordinate')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 4: Continuous waves for comparison
        plt.subplot(2, 3, 4)
        smooth = np.linspace(padding, grid_size-padding, 1000)
        smooth_wave = np.linspace(0, effective_grid_size, 1000)
        if horizontal:
            x1_smooth = amplitude * np.sin(frequency * smooth_wave) + horizontal_center
            x2_smooth = amplitude * np.sin(frequency * smooth_wave + phase_offset) + horizontal_center
            plt.plot(x1_smooth, smooth, 'r--', linewidth=2, alpha=0.7, label='Wave 1 (continuous)')
            plt.plot(x2_smooth, smooth, 'b--', linewidth=2, alpha=0.7, label='Wave 2 (continuous)')
            plt.plot(wave1_points[:, 0], wave1_points[:, 1], 'ro', markersize=3, alpha=0.6, label='Wave 1 (discrete)')
            plt.plot(wave2_points[:, 0], wave2_points[:, 1], 'bo', markersize=3, alpha=0.6, label='Wave 2 (discrete)')
            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')
        else:
            y1_smooth = amplitude * np.sin(frequency * smooth_wave) + vertical_center
            y2_smooth = amplitude * np.sin(frequency * smooth_wave + phase_offset) + vertical_center
            plt.plot(smooth, y1_smooth, 'r--', linewidth=2, alpha=0.7, label='Wave 1 (continuous)')
            plt.plot(smooth, y2_smooth, 'b--', linewidth=2, alpha=0.7, label='Wave 2 (continuous)')
            plt.plot(wave1_points[:, 0], wave1_points[:, 1], 'ro', markersize=3, alpha=0.6, label='Wave 1 (discrete)')
            plt.plot(wave2_points[:, 0], wave2_points[:, 1], 'bo', markersize=3, alpha=0.6, label='Wave 2 (discrete)')
            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')
        plt.title('Continuous vs Discrete Waves')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 5: Phase relationship
        plt.subplot(2, 3, 5)
        sample = smooth_wave[:500]
        if horizontal:
            x1_sample = amplitude * np.sin(frequency * sample) + horizontal_center
            x2_sample = amplitude * np.sin(frequency * sample + phase_offset) + horizontal_center
            plt.plot(x1_sample, sample, 'r-', linewidth=3, label='Wave 1')
            plt.plot(x2_sample, sample, 'b-', linewidth=3, label='Wave 2')
            plt.axvline(x=horizontal_center, color='gray', linestyle=':', alpha=0.5, label='Center line')
            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')
        else:
            y1_sample = amplitude * np.sin(frequency * sample) + vertical_center
            y2_sample = amplitude * np.sin(frequency * sample + phase_offset) + vertical_center
            plt.plot(sample, y1_sample, 'r-', linewidth=3, label='Wave 1')
            plt.plot(sample, y2_sample, 'b-', linewidth=3, label='Wave 2')
            plt.axhline(y=vertical_center, color='gray', linestyle=':', alpha=0.5, label='Center line')
            plt.xlabel('X coordinate')
            plt.ylabel('Y coordinate')
        plt.title(f'Phase Relationship\n(Phase offset: {phase_offset*180/np.pi:.1f}°)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot 6: Intersection analysis
        plt.subplot(2, 3, 6)
        if intersections:
            intersection_indices = [pt[0] for pt in intersections]
            if horizontal:
                plt.scatter(intersection_indices, [wave1_points[i, 1] for i in intersection_indices], 
                            c='purple', s=60, alpha=0.8, label='Intersection Y positions')
                plt.xlabel('Wave 1 point index')
                plt.ylabel('Y coordinate of intersection')
            else:
                plt.scatter(intersection_indices, [wave1_points[i, 0] for i in intersection_indices], 
                            c='purple', s=60, alpha=0.8, label='Intersection X positions')
                plt.xlabel('Wave 1 point index')
                plt.ylabel('X coordinate of intersection')
            plt.title('Intersection Distribution')
            plt.grid(True, alpha=0.3)
            plt.legend()
        else:
            plt.text(0.5, 0.5, 'No intersections found\nat current resolution', 
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=12)
            plt.title('Intersection Analysis')

        plt.tight_layout()
        plt.show()

    return wave1_points, wave2_points, grid

def create_ellipse_correspondences(n_points=30, center=(128, 128), 
                                   axes1=(40, 80), axes2=(40, 120), 
                                   angle1=0, angle2=0, seed=42, plot=True):
    """
    Create two ellipses with corresponding points, both centered at the same location.
    The second ellipse is longer along the y-axis.
    Returns:
        ellipse1_points: (N, 2) array of integers
        ellipse2_points: (N, 2) array of integers
    """
    np.random.seed(seed)
    theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

    # First ellipse (shorter y-axis)
    x1 = axes1[0] * np.cos(theta)
    y1 = axes1[1] * np.sin(theta)
    # Second ellipse (longer y-axis)
    x2 = axes2[0] * np.cos(theta)
    y2 = axes2[1] * np.sin(theta)

    # Optionally rotate ellipses (angle in radians)
    def rotate(x, y, angle):
        xr = x * np.cos(angle) - y * np.sin(angle)
        yr = x * np.sin(angle) + y * np.cos(angle)
        return xr, yr

    x1, y1 = rotate(x1, y1, angle1)
    x2, y2 = rotate(x2, y2, angle2)

    # Center both ellipses
    x1 = np.round(x1 + center[0]).astype(int)
    y1 = np.round(y1 + center[1]).astype(int)
    x2 = np.round(x2 + center[0]).astype(int)
    y2 = np.round(y2 + center[1]).astype(int)

    ellipse1_points = np.stack([x1, y1], axis=1)
    ellipse2_points = np.stack([x2, y2], axis=1)

    if plot:
        plt.figure(figsize=(8, 8))
        plt.plot(*ellipse1_points.T, 'o-', color='blue', label='Ellipse 1 (short y)')
        plt.plot(*ellipse2_points.T, 'o-', color='red', label='Ellipse 2 (long y)')
        for i in range(n_points):
            plt.plot([ellipse1_points[i, 0], ellipse2_points[i, 0]],
                     [ellipse1_points[i, 1], ellipse2_points[i, 1]],
                     color='gray', alpha=0.4)
        plt.scatter(ellipse1_points[:, 0], ellipse1_points[:, 1], c='blue', s=60, edgecolors='k', zorder=3)
        plt.scatter(ellipse2_points[:, 0], ellipse2_points[:, 1], c='red', s=60, edgecolors='k', zorder=3)
        plt.title("Ellipse Correspondences (Same Center, Different Y-Axis)")
        plt.axis('equal')
        plt.legend()
        plt.show()

    return ellipse1_points, ellipse2_points
