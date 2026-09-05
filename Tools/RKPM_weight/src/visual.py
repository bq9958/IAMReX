import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src import SI_generated, window, error

# Generate and save a 3D visualization of markers and support domains.
def visualize_results(lagrangian_points, nearest_grid_points,
                     delta_I, eta_I, theta_I, all_S_I, target_idx=0):
    """Visualize generated markers, grid points and a selected support domain."""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Draw a sparse sample of Eulerian grid points.
    # skip = max(1, len(eulerian_points) // 5000)
    # ax.scatter(eulerian_points[::skip, 0], eulerian_points[::skip, 1], eulerian_points[::skip, 2],
    #            c='red', alpha=1, s=10, label=f'Eulerian Grid (1/{skip} points)')

    # Draw all Lagrangian markers.
    ax.scatter(lagrangian_points[:, 0], lagrangian_points[:, 1], lagrangian_points[:, 2],
               c='blue', s=20, alpha=0.7, label=f'Lagrangian Points (n={len(lagrangian_points)})')

    # Highlight the selected marker and its S_I region.
    target_lag = lagrangian_points[target_idx]
    nearest_point = nearest_grid_points[target_idx]

    # Selected Lagrangian marker.
    ax.scatter([target_lag[0]], [target_lag[1]], [target_lag[2]],
               c='red', s=20, label=f'Target Point (idx={target_idx})')

    # Nearest Eulerian grid point.
    ax.scatter([nearest_point[0]], [nearest_point[1]], [nearest_point[2]],
               c='purple', s=30, marker='s', label='Nearest Eulerian Point')

    # Connecting line.
    ax.plot([target_lag[0], nearest_point[0]],
            [target_lag[1], nearest_point[1]],
            [target_lag[2], nearest_point[2]],
            'k--', linewidth=1, alpha=0.5)

    # Draw the S_I region.
    S_I = all_S_I[target_idx]
    if len(S_I) > 0:
        ax.scatter(S_I[:, 0], S_I[:, 1], S_I[:, 2],
                   c='green', s=30, alpha=0.5,
                   label=f'S_I Region (n={len(S_I)})')

        # Draw the bounding box.
        delta = delta_I[target_idx]
        eta = eta_I[target_idx]
        theta = theta_I[target_idx]

        # Bounding-box vertices.
        corners = np.array([
            [nearest_point[0]-1.5*delta, nearest_point[1]-1.5*eta, nearest_point[2]-1.5*theta],
            [nearest_point[0]+1.5*delta, nearest_point[1]-1.5*eta, nearest_point[2]-1.5*theta],
            [nearest_point[0]+1.5*delta, nearest_point[1]+1.5*eta, nearest_point[2]-1.5*theta],
            [nearest_point[0]-1.5*delta, nearest_point[1]+1.5*eta, nearest_point[2]-1.5*theta],
            [nearest_point[0]-1.5*delta, nearest_point[1]-1.5*eta, nearest_point[2]+1.5*theta],
            [nearest_point[0]+1.5*delta, nearest_point[1]-1.5*eta, nearest_point[2]+1.5*theta],
            [nearest_point[0]+1.5*delta, nearest_point[1]+1.5*eta, nearest_point[2]+1.5*theta],
            [nearest_point[0]-1.5*delta, nearest_point[1]+1.5*eta, nearest_point[2]+1.5*theta]
        ])

        # Bounding-box edges.
        edges = [
            [0,1],[1,2],[2,3],[3,0],  # Bottom face.
            [4,5],[5,6],[6,7],[7,4],  # Top face.
            [0,4],[1,5],[2,6],[3,7]   # Side edges.
        ]
        for edge in edges:
            ax.plot(corners[edge, 0], corners[edge, 1], corners[edge, 2],
                   'orange', linestyle='--', linewidth=1, alpha=0.7)

    # Plot settings.
    ax.set_xlabel('X Axis', fontsize=12)
    ax.set_ylabel('Y Axis', fontsize=12)
    ax.set_zlabel('Z Axis', fontsize=12)
    # ax.set_box_aspect([1, 1, 1])
    ax.axis('equal')
    ax.set_title(
        f'3D Grid Visualization\n'
        # f'Grid Size: {np.max(eulerian_points[:,0]):.1f}, '
        f'Lagrangian Points: {len(lagrangian_points)}',
        fontsize=14
    )
    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
    ax.view_init(elev=30, azim=45)
    plt.tight_layout()
    plt.savefig(
        'RKPM_3D.png',  # Output file.
        dpi=300,  # High resolution.
    )

def PointCloud(lagrangian_points):  # Plot a point cloud.
    # Create a 3D figure.
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw the scatter plot.
    scatter = ax.scatter(lagrangian_points[:,0], lagrangian_points[:,1], lagrangian_points[:,2],
                         c='black',        # Point color.
                         s=30,             # Point size.
                         alpha=0.8,        # Opacity.
                         depthshade=True)  # Enable depth shading.

    # Add labels and a title.
    ax.set_xlabel('X Axis', fontsize=12, labelpad=10)
    ax.set_ylabel('Y Axis', fontsize=12, labelpad=10)
    ax.set_zlabel('Z Axis', fontsize=12, labelpad=10)
    ax.set_title('Scatter Plot of Lagrangian Points', fontsize=14, pad=20)

    # Adjust the view angle if needed.
    # ax.view_init(elev=25, azim=45)
    ax.view_init(elev=90, azim=-90)  # x-y plane
    # ax.view_init(elev=0, azim=-90) # x-z plane

    # Add a grid.
    # ax.set_box_aspect([1, 1, 1])
    ax.axis('equal')
    ax.grid(True, linestyle='--', alpha=0.5)

    # Save the figure.
    plt.tight_layout()
    plt.savefig(
        'lagrangian.png',  # Output file.
        dpi=300,  # High resolution.
    )
