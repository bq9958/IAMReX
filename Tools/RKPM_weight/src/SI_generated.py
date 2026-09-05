import numpy as np
from src import visual
import sys
import ast

# Read Lagrangian marker coordinates (xp, yp, zp) from a whitespace-delimited
# point-cloud file. Lines beginning with # are comments.
#   # X Y Z
#   -0.0087703851 0.0371783379 0.0369941932
#   ...
def read_geometry_file(file_path):
    points = np.atleast_2d(np.loadtxt(file_path, comments='#'))
    if points.size == 0:
        raise ValueError(f"Geometry file is empty or cannot be parsed: {file_path}")
    if points.shape[1] != 3:
        raise ValueError(
            f"Geometry file must have 3 columns (X Y Z), got "
            f"{points.shape[1]}: {file_path}")
    return points

# Legacy helper for reading marker positions from an ID file.
# def load_lagrangian_from_id_file(filename):
#     """
#     Read an .id file and convert it to an (N, 3) NumPy array.
#     """
#     # Read the complete file.
#     with open(filename, 'r') as f:
#         content = f.read()
#
#     # Convert the valid Python dictionary literal to an object.
#     try:
#         data_dict = ast.literal_eval(content)
#     except Exception as e:
#         print(f"Failed to parse file: {e}")
#         return None
#
#     # Determine N, assuming IDs are contiguous from 0 through N-1.
#     if not data_dict:
#         return np.empty((0, 3))
#
#     num_points = len(data_dict)
#
#     # Initialize the NumPy array.
#     lagrangian_points = np.zeros((num_points, 3))
#
#     # Fill the array using dictionary keys as indices.
#     for lag_id, coords in data_dict.items():
#         lagrangian_points[lag_id] = coords
#
#     return lagrangian_points

# Find the interval index of each point in a grid-coordinate array.
def find_grid_indices(points, coords):
    idxs = np.searchsorted(coords, points, side='right') - 1
    if np.any(idxs < 0) or np.any(idxs >= len(coords) - 1):
        raise ValueError("Some points are out of bounds.")
    return idxs

# Approximate ellipsoid surface area with the Knud Thomsen formula.
def ellipsoid_area_approx(a, b, c):
    """
    Approximate ellipsoid surface area with the Knud Thomsen formula.
    """
    p = 1.6075
    term = ((a * b)**p + (a * c)**p + (b * c)**p) / 3.0
    return 4 * np.pi * (term**(1 / p))

def cylinder_area(radius, height):
    """
    Compute total cylinder surface area, including both end caps.

    Args:
        radius (float): Cylinder radius.
        height (float): Cylinder height.

    Returns:
        float: Total area = 2 * pi * radius * (radius + height).
    """
    return 2 * np.pi * radius * (radius + height)

# Generate a 3D Eulerian grid and Lagrangian markers.
#
# Design: operate directly in the global coordinate system of the solver's
# finest grid without selecting a local subregion or applying a local-to-global
# index offset. The old int(sx/dx) truncation was ambiguous.
#   1. Read prob_lo/prob_hi/n_cell/max_level and compute
#      dx_finest = (prob_hi - prob_lo) / (n_cell * 2**max_level).
#   2. Expand the point-cloud bounds by two finest cells and clamp the result to
#      the computational domain.
#   3. Generate global cell centers as (index + 0.5) * dx + prob_lo. RKPM weights
#      depend only on relative coordinates and are independent of the origin.
#
# Inputs: domain bounds, finest-cell size, point-cloud path, optional body-frame
# center and z-axis rotation angle in degrees. Outputs include Eulerian points,
# transformed Lagrangian points, nearest grid points, support parameters and all
# per-marker support domains.
def generate_grid(prob_lo, prob_hi, dx_finest, geometry_file=None, center=None, angle=0.0):

    prob_lo = np.asarray(prob_lo, dtype=float)
    prob_hi = np.asarray(prob_hi, dtype=float)
    dx_finest = np.asarray(dx_finest, dtype=float)
    dx, dy, dz = dx_finest
    # Number of finest-grid cells per direction, used for domain clamping.
    n_fine = np.round((prob_hi - prob_lo) / dx_finest).astype(int)

    # Lagrangian point cloud.
    if geometry_file is not None:
        lagrangian_points = read_geometry_file(geometry_file)
    else:
        raise NameError("A Lagrangian marker-coordinate file is required")
        # lagrangian_points = load_lagrangian_from_id_file('rkpm_mappings.id')
    # Translate body-frame coordinates to world coordinates.
    if center is not None:
        center = np.asarray(center, dtype=float)
        lagrangian_points = lagrangian_points + center
    # Rotate around z after translation; use center as the body-frame pivot.
    if angle:
        pivot = center if center is not None else np.zeros(3)
        theta = np.radians(angle)
        c, s = np.cos(theta), np.sin(theta)
        Rz = np.array([[ c, -s, 0.],
                       [ s,  c, 0.],
                       [0., 0., 1.]])
        lagrangian_points = (lagrangian_points - pivot) @ Rz.T + pivot
    visual.PointCloud(lagrangian_points)

    # Bounding region: expand point-cloud extrema by two finest-grid cells.
    pmin = lagrangian_points.min(axis=0)
    pmax = lagrangian_points.max(axis=0)
    # The point cloud must remain inside the computational domain.
    if np.any(pmin < prob_lo) or np.any(pmax > prob_hi):
        raise ValueError(
            f"Point cloud extends outside [prob_lo, prob_hi]:\n"
            f"  prob_lo = {prob_lo.tolist()}\n  prob_hi = {prob_hi.tolist()}\n"
            f"  point-cloud min = {pmin.tolist()}\n"
            f"  point-cloud max = {pmax.tolist()}\n"
            "Check the --geometry coordinate system or use --body-frame to "
            "translate the point cloud into the domain."
        )
    ranges = pmax - pmin  # = np.ptp(lagrangian_points, axis=0)
    print('ranges', ranges[0], ranges[1], ranges[2])
    print('ranges', ranges[0]/2, ranges[1]/2, ranges[2]/2)

    # Expand finest-grid global indices and clamp them to [0, n_fine-1].
    i_lo = int(np.floor((pmin[0] - prob_lo[0]) / dx)) - 2
    j_lo = int(np.floor((pmin[1] - prob_lo[1]) / dy)) - 2
    k_lo = int(np.floor((pmin[2] - prob_lo[2]) / dz)) - 2
    i_hi = int(np.floor((pmax[0] - prob_lo[0]) / dx)) + 2
    j_hi = int(np.floor((pmax[1] - prob_lo[1]) / dy)) + 2
    k_hi = int(np.floor((pmax[2] - prob_lo[2]) / dz)) + 2
    i_lo, j_lo, k_lo = max(0, i_lo), max(0, j_lo), max(0, k_lo)
    i_hi = min(int(n_fine[0]) - 1, i_hi)
    j_hi = min(int(n_fine[1]) - 1, j_hi)
    k_hi = min(int(n_fine[2]) - 1, k_hi)
    print(f"[grid] finest index range: i[{i_lo},{i_hi}] j[{j_lo},{j_hi}] k[{k_lo},{k_hi}] "
          f"-> cells ({i_hi-i_lo+1}, {j_hi-j_lo+1}, {k_hi-k_lo+1})")

    # Eulerian grid: global cell centers at (index + 0.5) * dx + prob_lo.
    # x/y/z are cell edges; xc/yc/zc are cell centers.
    x = prob_lo[0] + np.arange(i_lo, i_hi + 2) * dx
    y = prob_lo[1] + np.arange(j_lo, j_hi + 2) * dy
    z = prob_lo[2] + np.arange(k_lo, k_hi + 2) * dz
    xc = 0.5 * (x[:-1] + x[1:])
    yc = 0.5 * (y[:-1] + y[1:])
    zc = 0.5 * (z[:-1] + z[1:])

    XC, YC, ZC = np.meshgrid(xc, yc, zc, indexing='ij')

    # Compute cell volumes.
    Delta_V = np.full((len(x)-1, len(y)-1, len(z)-1), dx * dy * dz)

    # Assemble Eulerian coordinates.
    eulerian_points = np.vstack([XC.ravel(), YC.ravel(), ZC.ravel()]).T

    # Estimate marker area and thickness.
    Ne = len(lagrangian_points)
    area = ellipsoid_area_approx(ranges[0]/2, ranges[1]/2, ranges[2]/2) / Ne
    # area = ellipsoid_area_approx(0.0437, 0.0437, 0.0655) / Ne
    # area = cylinder_area(0.03815, 0.1145) / Ne
    thickness = min(dx, dy, dz)
    V_lag = area * thickness
    print(f"Vl: {V_lag}, area:{area}, frac: {V_lag / Delta_V[0][0][0]}")

    # Get the grid shape.
    grid_shape = XC.shape

    # Find each marker's containing global cell.
    indices_ijk = np.floor((lagrangian_points - prob_lo) / dx_finest).astype(int)
    # Convert global indices to local array indices.
    local_ijk = (indices_ijk[:, 0] - i_lo, indices_ijk[:, 1] - j_lo, indices_ijk[:, 2] - k_lo)
    nearest_indices = np.ravel_multi_index(local_ijk, dims=grid_shape)
    nearest_grid_points = eulerian_points[nearest_indices]

    # Compute support-domain scales.
    delta_I = np.full(Ne, dx + (1 / 1000) * dx)
    eta_I = np.full(Ne, dy + (1 / 1000) * dy)
    theta_I = np.full(Ne, dz + (1 / 1000) * dz)

    # Build all per-marker support domains.
    all_S_I = []
    for idx in range(Ne):

        nearest_point = nearest_grid_points[idx]
        delta_I_lag = delta_I[idx]
        eta_I_lag = eta_I[idx]
        theta_I_lag = theta_I[idx]
        # Select Eulerian points inside the rectangular support region.
        mask_x = np.abs(eulerian_points[:, 0] - nearest_point[0]) < 1.5 * delta_I_lag
        mask_y = np.abs(eulerian_points[:, 1] - nearest_point[1]) < 1.5 * eta_I_lag
        mask_z = np.abs(eulerian_points[:, 2] - nearest_point[2]) < 1.5 * theta_I_lag

        S_I_points = eulerian_points[mask_x & mask_y & mask_z]

        # Find grid indices for all selected points.
        i = find_grid_indices(S_I_points[:, 0], x)
        j = find_grid_indices(S_I_points[:, 1], y)
        k = find_grid_indices(S_I_points[:, 2], z)

        volume_points = Delta_V[i,j,k]

        # Combine coordinates and volume as (x, y, z, volume).
        S_I = np.column_stack((S_I_points, volume_points.reshape(-1, 1)))
        all_S_I.append(S_I)

    return eulerian_points, Ne, lagrangian_points, nearest_grid_points, delta_I, eta_I, theta_I, all_S_I, V_lag
