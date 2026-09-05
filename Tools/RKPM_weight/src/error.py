import numpy as np

# Test function returning sin(pi*x)*cos(pi*y) for scalar or array coordinates.
def test_function(x, y):
    return np.sin(np.pi * x) * np.cos(np.pi * y)

# Return the infinity-norm error between original and interpolated values.
def compute_infinity_norm_error(original_values, interpolated_values):
    diff = original_values - interpolated_values
    return np.max(np.abs(diff))

def compute_error_volume(lagrangian_points, all_S_I, all_modified_w, V_lag):

    interpolated_volumn = np.zeros(len(lagrangian_points))
    for idx, (S_I, modified_w) in enumerate(zip(all_S_I, all_modified_w)):
        modified_w = np.array(modified_w)
        for j, data in enumerate(S_I):
            Delta_V = data[3]
            interpolated_volumn[idx] += Delta_V * modified_w[j] * V_lag / Delta_V

    relavtive_error = np.abs(interpolated_volumn - V_lag) / V_lag
    print(f"Relative volume error: Min: {np.min(relavtive_error):.6e}, Max: {np.max(relavtive_error):.6e}")

    return relavtive_error

# Evaluate interpolation error and force conservation.
def compute_error(eulerian_points, lagrangian_points, all_S_I, all_modified_w, V_lag):
    # original_values = test_function(lagrangian_points[:, 0], lagrangian_points[:, 1])
    original_values = np.ones(len(lagrangian_points))

    dispersion_values = np.zeros(len(eulerian_points))
    for idx, (S_I, modified_w) in enumerate(zip(all_S_I, all_modified_w)):
        modified_w = np.array(modified_w)
        for j, data in enumerate(S_I):
            point = data[:3]
            Delta_V = data[3]
            ide = np.where((eulerian_points == point).all(axis=1))[0][0]
            dispersion_values[ide] += original_values[idx] * modified_w[j] * V_lag / Delta_V

    interpolated_values = np.zeros(len(lagrangian_points))
    for idx, (S_I, modified_w) in enumerate(zip(all_S_I, all_modified_w)):
        modified_w = np.array(modified_w)
        for j, data in enumerate(S_I):
            point = data[:3]
            Delta_V = data[3]
            ide = np.where((eulerian_points == point).all(axis=1))[0][0]
            interpolated_values[idx] += dispersion_values[ide] * modified_w[j]

    # check force conservation
    force_lagrangian = np.sum(original_values * V_lag)
    volumes = np.full(len(eulerian_points), fill_value=all_S_I[0][0][3])
    force_euler = np.sum(dispersion_values * volumes)
    print('force check',force_lagrangian,force_euler, abs(force_lagrangian - force_euler)/force_lagrangian)

    return compute_infinity_norm_error(original_values, interpolated_values)

def compute_conservation_check(center, eulerian_points, lagrangian_points, all_S_I, all_modified_w, V_lag):
    """Check force and moment/torque conservation simultaneously."""
    # Build a coordinate-to-index map to avoid repeated O(N^2) searches.
    # Coordinates are safe tuple keys here because they originate in one pipeline.
    coord_map = {tuple(p): i for i, p in enumerate(eulerian_points)}

    # Define an asymmetric marker force field: constant + omega cross r.
    # The constant component prevents near-zero totals from distorting relative errors.
    rel_pos = lagrangian_points - center
    omega = np.array([0.7, -0.3, 0.5])  # Controls rotational asymmetry.
    F_const = np.array([1.0, 0.8, 1.2])  # Prevents a near-zero total force.
    F_lag = F_const + np.cross(np.tile(omega, (len(lagrangian_points), 1)), rel_pos)

    # Compute total force and torque on the Lagrangian side.
    # Force = sum(F_l * V_l)
    total_force_lag = np.sum(F_lag * V_lag, axis=0)

    # Torque = sum((X_l x F_l) * V_l)
    # np.cross(A, B) evaluates the cross product.
    torques_lag = np.cross(lagrangian_points - center, F_lag)
    total_torque_lag = np.sum(torques_lag * V_lag, axis=0)

    print(f"Lagrangian Total Force: {total_force_lag}")
    print(f"Lagrangian Total Torque: {total_torque_lag}")

    # Spread the force to the Eulerian grid.
    f_euler = np.zeros((len(eulerian_points), 3))
    grid_volumes = np.zeros(len(eulerian_points))

    for idx, (S_I, modified_w) in enumerate(zip(all_S_I, all_modified_w)):
        modified_w = np.array(modified_w)
        F_l = F_lag[idx]  # Force vector at the current marker.

        for j, data in enumerate(S_I):
            point = data[:3]
            Delta_V = data[3]  # Grid-cell volume.

            # Look up the grid index.
            point_tuple = tuple(point)
            if point_tuple in coord_map:
                ide = coord_map[point_tuple]

                # Spreading formula: f(x) += F(X_l) * w(X_l - x).
                f_euler[ide] += F_l * modified_w[j] * V_lag / Delta_V

                # Record the cell volume; it is identical across overlapping supports.
                if grid_volumes[ide] == 0:
                    grid_volumes[ide] = Delta_V
            else:
                # This should occur only because of floating-point mismatch.
                pass

    # Compute total force and torque on the Eulerian side.
    # Force_grid = sum(f(x) * dv)
    # Torque_grid = sum((x x f(x)) * dv)

    # Include only affected grid cells with nonzero recorded volume.
    valid_mask = grid_volumes > 0
    valid_points = eulerian_points[valid_mask]
    valid_forces = f_euler[valid_mask]
    valid_volumes = grid_volumes[valid_mask].reshape(-1, 1) # reshape for broadcasting

    # Eulerian total force.
    total_force_euler = np.sum(valid_forces * valid_volumes, axis=0)

    # Eulerian total torque.
    torques_euler = np.cross(valid_points - center, valid_forces)
    total_torque_euler = np.sum(torques_euler * valid_volumes, axis=0)

    print(f"Eulerian Total Force: {total_force_euler}")
    print(f"Eulerian Total Torque: {total_torque_euler}")

    # Compute relative errors.
    force_diff = np.abs(total_force_lag - total_force_euler)
    force_rel_error = np.linalg.norm(force_diff) / np.linalg.norm(total_force_lag)

    torque_diff = np.abs(total_torque_lag - total_torque_euler)
    torque_rel_error = np.linalg.norm(torque_diff) / np.linalg.norm(total_torque_lag)

    print("-" * 30)
    print(f"Force Conservation Relative Error: {force_rel_error:.6e}")
    print(f"Torque Conservation Relative Error: {torque_rel_error:.6e}")

    return force_rel_error, torque_rel_error
