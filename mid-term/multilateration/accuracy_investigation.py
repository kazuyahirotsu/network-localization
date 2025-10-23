
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import math
from scipy.optimize import least_squares

# --- Configuration ---
INSTANCE_TO_ANALYZE = 1
MATLAB_DATA_BASE_PATH = "matlab/data/64beacons_100instances/"
OUTPUT_DIR = "mid-term/multilateration/investigation_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Node configuration from the dataset
NUM_ANCHORS = 16
NUM_UNKNOWNS = 48
TOTAL_NODES = NUM_ANCHORS + NUM_UNKNOWNS

# Map origin from MATLAB code for coordinate conversion
MAP_ORIGIN_LAT = 40.466198
MAP_ORIGIN_LON = 33.898610
EARTH_RADIUS = 6378137.0
METERS_PER_DEGREE_LAT = (math.pi / 180) * EARTH_RADIUS
METERS_PER_DEGREE_LON = (math.pi / 180) * EARTH_RADIUS * np.cos(np.deg2rad(MAP_ORIGIN_LAT))

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid')
FONT_SIZE_TITLE = 20
FONT_SIZE_LABEL = 16
FONT_SIZE_TICKS = 14
# --- End Configuration ---


# --- Utility Functions (from 64beacon_200instances.py) ---

def latlon_to_xy(lat, lon):
    """Converts a single lat/lon pair to XY meters."""
    x = (lon - MAP_ORIGIN_LON) * METERS_PER_DEGREE_LON
    y = (lat - MAP_ORIGIN_LAT) * METERS_PER_DEGREE_LAT
    return x, y

def log_distance_model_for_fit(d, A, n):
    """Model function for curve_fit: RSSI = A - 10*n*log10(d)"""
    d_safe = np.maximum(d, 1e-9)
    return A - 10 * n * np.log10(d_safe)

def rssi_to_distance_log_model(rssi, A_param, n_param):
    """Converts RSSI to distance using the log-distance path loss model."""
    if n_param == 0: return float('inf')
    # Prevent overflow
    power_val = (A_param - rssi) / (10 * n_param)
    if power_val > 30: return float('inf')
    return 10**power_val

def multilateration_error_func(position_guess, known_positions, measured_distances):
    """Error function for least_squares multilateration."""
    errors = np.linalg.norm(known_positions - position_guess, axis=1) - measured_distances
    return errors

def estimate_unknown_position_ls(initial_guess_xy, neighbor_positions_xy, neighbor_rssi_values, A_param, n_param):
    """Estimates the position of an unknown node using least squares."""
    measured_distances = np.array([rssi_to_distance_log_model(rssi, A_param, n_param) for rssi in neighbor_rssi_values])
    
    valid_indices = np.isfinite(measured_distances) & (measured_distances < 1e4)
    if np.sum(valid_indices) < 3: return None # Need at least 3 for stable 2D solve
        
    known_pos_filtered = np.array(neighbor_positions_xy)[valid_indices]
    measured_dist_filtered = measured_distances[valid_indices]

    bounds = ([-1000, -1000], [5000, 5000])

    result = least_squares(
        multilateration_error_func,
        initial_guess_xy,
        args=(known_pos_filtered, measured_dist_filtered),
        bounds=bounds,
        method='trf',
        ftol=1e-5, xtol=1e-5, gtol=1e-5
    )
    return result.x if result.success else None

def estimate_rssi_parameters(anchor_positions_xy, anchor_rssi_matrix):
    """Estimates A and n parameters using only anchor-to-anchor measurements."""
    distances, avg_rssis = [], []
    num_current_anchors = anchor_positions_xy.shape[0]

    for i in range(num_current_anchors):
        for j in range(i + 1, num_current_anchors):
            dist_ij = np.linalg.norm(anchor_positions_xy[i] - anchor_positions_xy[j])
            if dist_ij == 0: continue
            
            rssi_ij = anchor_rssi_matrix[i, j]
            if not np.isnan(rssi_ij):
                distances.append(dist_ij)
                avg_rssis.append(rssi_ij)

    if len(distances) < 2:
        return -50.0, 3.0 # Fallback

    distances = np.array(distances)
    avg_rssis = np.array(avg_rssis)
    param_bounds = ([-100, 1.5], [50, 6.0])
    p0 = [np.max(avg_rssis), 3.0]

    try:
        params, _ = curve_fit(log_distance_model_for_fit, distances, avg_rssis, p0=p0, bounds=param_bounds, maxfev=5000)
        return params[0], params[1]
    except (RuntimeError, ValueError):
        return -50.0, 3.0

def load_matlab_instance(instance_idx):
    """Loads and processes data for a single MATLAB instance."""
    filename = os.path.join(MATLAB_DATA_BASE_PATH, f"data_instance_{instance_idx}.mat")
    try:
        mat_data = loadmat(filename)
    except FileNotFoundError:
        print(f"Error: File not found {filename}")
        return None, None

    node_latitudes = mat_data['nodeLatitudes'].flatten()
    node_longitudes = mat_data['nodeLongitudes'].flatten()
    signal_strength_matrix = mat_data['signal_strength_matrix']
    rssi_matrix_avg = np.nanmean(signal_strength_matrix, axis=2)

    node_positions_xy = np.array([latlon_to_xy(lat, lon) for lat, lon in zip(node_latitudes, node_longitudes)])
    
    return node_positions_xy, rssi_matrix_avg

# --- Main Investigation Script ---

def main():
    """
    Main function to run the investigation tasks.
    """
    print(f"--- Running Investigation for Instance {INSTANCE_TO_ANALYZE} ---")

    # 1. Load Data
    node_positions_xy, rssi_matrix_avg = load_matlab_instance(INSTANCE_TO_ANALYZE)
    if node_positions_xy is None:
        return

    # --- TASK 0: Visualize Node Distribution ---
    print("\nTask 0: Visualizing Node Distribution...")
    map_size_for_plot = 4000.0 # Assuming 4km map
    mid_point_for_plot = map_size_for_plot / 2
    
    plt.figure(figsize=(10, 10))
    anchor_pos = node_positions_xy[:NUM_ANCHORS]
    unknown_pos = node_positions_xy[NUM_ANCHORS:]

    plt.scatter(anchor_pos[:, 0], anchor_pos[:, 1], c='green', marker='^', s=120, label='Anchor Nodes', edgecolors='black', zorder=3)
    plt.scatter(unknown_pos[:, 0], unknown_pos[:, 1], c='blue', marker='o', s=60, label='Unknown Nodes', alpha=0.8, zorder=2)

    # Draw quadrant lines
    plt.axvline(x=mid_point_for_plot, color='grey', linestyle='--', linewidth=1.5)
    plt.axhline(y=mid_point_for_plot, color='grey', linestyle='--', linewidth=1.5)

    plt.title(f'Node Distribution for Instance {INSTANCE_TO_ANALYZE}', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('X Position (m)', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Y Position (m)', fontsize=FONT_SIZE_LABEL)
    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    plt.legend(fontsize=FONT_SIZE_LABEL)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.axis('equal')
    plt.xlim(0, map_size_for_plot)
    plt.ylim(0, map_size_for_plot)

    output_path = os.path.join(OUTPUT_DIR, "task_0_node_distribution.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

    # 2. Estimate the Global Path-Loss Model (A, n) using only anchor nodes
    anchor_positions_xy = node_positions_xy[:NUM_ANCHORS]
    anchor_rssi_matrix = rssi_matrix_avg[:NUM_ANCHORS, :NUM_ANCHORS]
    A_global, n_global = estimate_rssi_parameters(anchor_positions_xy, anchor_rssi_matrix)
    print(f"Estimated Global Parameters from Anchors: A={A_global:.2f}, n={n_global:.2f}")

    # 3. Collect data for all links in the network
    all_links_distances = []
    all_links_rssi = []
    
    for i in range(TOTAL_NODES):
        for j in range(i + 1, TOTAL_NODES):
            dist = np.linalg.norm(node_positions_xy[i] - node_positions_xy[j])
            rssi = rssi_matrix_avg[i, j]
            if not np.isnan(rssi):
                all_links_distances.append(dist)
                all_links_rssi.append(rssi)
    
    all_links_distances = np.array(all_links_distances)
    all_links_rssi = np.array(all_links_rssi)

    # --- TASK 1.1: Visualize "All-Links" Signal-Distance Relationship ---
    print("\nTask 1.1: Generating 'All-Links' RSSI vs. Distance plot...")
    plt.figure(figsize=(12, 8))
    plt.scatter(all_links_distances, all_links_rssi, alpha=0.3, label='All Measured Links', s=10)
    
    # Plot the fitted model curve
    dist_range = np.linspace(np.min(all_links_distances), np.max(all_links_distances), 500)
    rssi_predicted_curve = log_distance_model_for_fit(dist_range, A_global, n_global)
    plt.plot(dist_range, rssi_predicted_curve, 'r-', linewidth=3, label=f'Fitted Model (A={A_global:.2f}, n={n_global:.2f})')

    plt.title('Global Model vs. All Links Data', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('True Distance (m)', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Measured RSSI (dBm)', fontsize=FONT_SIZE_LABEL)
    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    plt.legend(fontsize=FONT_SIZE_LABEL)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    output_path = os.path.join(OUTPUT_DIR, "task_1_1_all_links_rssi_vs_distance.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

    # --- TASK 1.2: Analyze the Model's Residuals ---
    print("\nTask 1.2: Generating model residuals histogram...")
    # Predict RSSI for every link using the true distance and the global model
    rssi_predicted_for_all_links = log_distance_model_for_fit(all_links_distances, A_global, n_global)
    
    # Calculate residuals
    residuals = rssi_predicted_for_all_links - all_links_rssi
    
    plt.figure(figsize=(12, 8))
    plt.hist(residuals, bins=100, alpha=0.75, label='Model Residuals')
    
    mean_residual = np.mean(residuals)
    std_residual = np.std(residuals)
    
    plt.axvline(mean_residual, color='r', linestyle='--', linewidth=2, label=f'Mean = {mean_residual:.2f}')
    plt.title('Task 1.2: Histogram of Model Residuals (Predicted - Measured RSSI)', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Residual RSSI (dBm)', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Frequency (Number of Links)', fontsize=FONT_SIZE_LABEL)
    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    plt.legend(fontsize=FONT_SIZE_LABEL)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.text(0.95, 0.95, f'Std Dev = {std_residual:.2f}', transform=plt.gca().transAxes,
             fontsize=FONT_SIZE_LABEL, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    output_path = os.path.join(OUTPUT_DIR, "task_1_2_model_residuals_histogram.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

    # --- TASK 2.1: Create a "Path-Loss Parameter Map" ---
    print("\nTask 2.1: Generating 'Path-Loss Parameter Map'...")
    
    # Define quadrants (assuming a 4000x4000m map area based on MATLAB script)
    map_size = 4000.0
    mid_point = map_size / 2
    quadrants = {
        'Bottom-Left': {'x_b': (0, mid_point), 'y_b': (0, mid_point), 'links': {'d': [], 'r': []}},
        'Bottom-Right': {'x_b': (mid_point, map_size), 'y_b': (0, mid_point), 'links': {'d': [], 'r': []}},
        'Top-Left': {'x_b': (0, mid_point), 'y_b': (mid_point, map_size), 'links': {'d': [], 'r': []}},
        'Top-Right': {'x_b': (mid_point, map_size), 'y_b': (mid_point, map_size), 'links': {'d': [], 'r': []}}
    }

    # Assign nodes to quadrants
    node_quadrant_indices = [None] * TOTAL_NODES
    for i in range(TOTAL_NODES):
        x, y = node_positions_xy[i]
        for name, q_info in quadrants.items():
            if q_info['x_b'][0] <= x < q_info['x_b'][1] and q_info['y_b'][0] <= y < q_info['y_b'][1]:
                node_quadrant_indices[i] = name
                break
    
    # Collect links that are fully within a single quadrant
    for i in range(TOTAL_NODES):
        for j in range(i + 1, TOTAL_NODES):
            # Check if both nodes are in the same quadrant
            if node_quadrant_indices[i] is not None and node_quadrant_indices[i] == node_quadrant_indices[j]:
                quadrant_name = node_quadrant_indices[i]
                dist = np.linalg.norm(node_positions_xy[i] - node_positions_xy[j])
                rssi = rssi_matrix_avg[i, j]
                if not np.isnan(rssi):
                    quadrants[quadrant_name]['links']['d'].append(dist)
                    quadrants[quadrant_name]['links']['r'].append(rssi)

    # Perform local fits and prepare for plotting
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    plot_titles = ['Bottom-Left', 'Bottom-Right', 'Top-Left', 'Top-Right']

    for i, name in enumerate(plot_titles):
        ax = axes[i]
        q_data = quadrants[name]
        distances = np.array(q_data['links']['d'])
        rssis = np.array(q_data['links']['r'])

        ax.scatter(distances, rssis, alpha=0.3, s=10, label='Local Links')
        
        # Fit local model
        if len(distances) > 2:
            # Re-using the logic from the global estimation function
            param_bounds = ([-100, 1.5], [50, 6.0])
            p0 = [np.max(rssis), 3.0]
            try:
                params, _ = curve_fit(log_distance_model_for_fit, distances, rssis, p0=p0, bounds=param_bounds, maxfev=5000)
                A_local, n_local = params
            except (RuntimeError, ValueError):
                A_local, n_local = -50.0, 3.0
            
            # Plot local fit
            dist_range_local = np.linspace(np.min(distances), np.max(distances), 100)
            rssi_local_curve = log_distance_model_for_fit(dist_range_local, A_local, n_local)
            ax.plot(dist_range_local, rssi_local_curve, 'g-', linewidth=2.5, label=f'Local Fit (A={A_local:.2f}, n={n_local:.2f})')

        # Plot global fit for comparison
        dist_range_global = np.linspace(np.min(all_links_distances), np.max(all_links_distances), 500)
        rssi_global_curve = log_distance_model_for_fit(dist_range_global, A_global, n_global)
        ax.plot(dist_range_global, rssi_global_curve, 'r--', linewidth=2.5, label=f'Global Fit (A={A_global:.2f}, n={n_global:.2f})')
        
        ax.set_title(f"Quadrant: {name}", fontsize=FONT_SIZE_LABEL)
        ax.legend(fontsize=12)
        ax.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)

    fig.suptitle('Task 2.1: Path-Loss Parameter Map (Local vs. Global Fit)', fontsize=FONT_SIZE_TITLE + 2)
    fig.supxlabel('True Distance (m)', fontsize=FONT_SIZE_LABEL)
    fig.supylabel('Measured RSSI (dBm)', fontsize=FONT_SIZE_LABEL)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_path = os.path.join(OUTPUT_DIR, "task_2_1_path_loss_parameter_map.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

    # --- ACT 3: Connect Model Failure to Node-Specific Localization Error ---
    print("\n--- Starting Act 3: The Cascade ---")
    
    # Run the iterative multilateration to get final positions
    print("Running iterative multilateration to get final error...")
    num_iterations = 10
    
    # Initialize unknown nodes to the average of anchor positions
    estimated_positions_xy = np.copy(node_positions_xy)
    avg_anchor_pos = np.mean(anchor_positions_xy, axis=0)
    estimated_positions_xy[NUM_ANCHORS:] = avg_anchor_pos + np.random.randn(NUM_UNKNOWNS, 2) * 10
    
    current_estimated_positions = np.copy(estimated_positions_xy)

    for iteration in range(num_iterations):
        newly_estimated_positions = np.copy(current_estimated_positions)
        for i_unknown in range(NUM_ANCHORS, TOTAL_NODES):
            neighbor_pos = []
            neighbor_rssi = []
            for j_neighbor in range(TOTAL_NODES):
                if i_unknown == j_neighbor: continue
                rssi_val = rssi_matrix_avg[i_unknown, j_neighbor]
                if not np.isnan(rssi_val):
                    neighbor_pos.append(current_estimated_positions[j_neighbor])
                    neighbor_rssi.append(rssi_val)
            
            if len(neighbor_pos) >= 3:
                est_pos = estimate_unknown_position_ls(
                    current_estimated_positions[i_unknown], neighbor_pos, neighbor_rssi, A_global, n_global
                )
                if est_pos is not None:
                    newly_estimated_positions[i_unknown] = est_pos
        current_estimated_positions = newly_estimated_positions

    final_estimated_positions = current_estimated_positions
    
    # Calculate final localization errors for each unknown node
    unknown_node_errors = np.linalg.norm(node_positions_xy[NUM_ANCHORS:] - final_estimated_positions[NUM_ANCHORS:], axis=1)

    # --- Task 3.1: Spatial Map of Node Localization Error ---
    print("Task 3.1: Generating spatial map of node localization error...")
    plt.figure(figsize=(12, 10))
    sc = plt.scatter(
        node_positions_xy[NUM_ANCHORS:, 0], 
        node_positions_xy[NUM_ANCHORS:, 1], 
        c=unknown_node_errors, 
        cmap='viridis', 
        s=80, 
        edgecolors='black',
        zorder=2
    )
    plt.scatter(anchor_positions_xy[:, 0], anchor_positions_xy[:, 1], c='red', marker='^', s=150, label='Anchor Nodes', edgecolors='black', zorder=3)
    cbar = plt.colorbar(sc)
    cbar.set_label('Localization Error (m)', fontsize=FONT_SIZE_LABEL)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICKS)
    
    plt.title('Spatial Distribution of Localization Error', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('X Position (m)', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Y Position (m)', fontsize=FONT_SIZE_LABEL)
    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    plt.legend(fontsize=FONT_SIZE_LABEL)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.axis('equal')
    plt.xlim(0, 4000)
    plt.ylim(0, 4000)

    output_path = os.path.join(OUTPUT_DIR, "task_3_1_spatial_error_map.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

    # --- Task 3.2: Correlate Node Error with Link Quality ---
    print("Task 3.2: Correlating localization error with link quality...")
    avg_link_dist_errors = []
    for i in range(NUM_ANCHORS, TOTAL_NODES):
        link_errors = []
        for j in range(TOTAL_NODES):
            if i == j: continue
            rssi = rssi_matrix_avg[i, j]
            if not np.isnan(rssi):
                true_dist = np.linalg.norm(node_positions_xy[i] - node_positions_xy[j])
                estimated_dist = rssi_to_distance_log_model(rssi, A_global, n_global)
                if np.isfinite(estimated_dist):
                    link_errors.append(abs(true_dist - estimated_dist))
        if link_errors:
            avg_link_dist_errors.append(np.mean(link_errors))
        else:
            avg_link_dist_errors.append(0)
    
    avg_link_dist_errors = np.array(avg_link_dist_errors)

    plt.figure(figsize=(12, 8))
    plt.scatter(avg_link_dist_errors, unknown_node_errors, alpha=0.7)
    
    plt.title('Localization Error vs. Average Link Distance Estimation Error', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Average Distance Estimation Error of Node\'s Links (m)', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Final Localization Error (m)', fontsize=FONT_SIZE_LABEL)
    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    output_path = os.path.join(OUTPUT_DIR, "task_3_2_error_vs_link_quality.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

    # --- ACT 4: Dissecting the Iterative Process ---
    print("\n--- Starting Act 4: The Feedback Loop ---")

    # Re-run localization, this time capturing history
    print("Re-running localization to capture iterative history...")
    
    # History storage
    # List of arrays, where each array holds the errors for all unknown nodes at one iteration
    iterative_errors = []
    # List of arrays for position change magnitudes
    iterative_deltas = []

    # Re-initialize positions
    estimated_positions_xy_hist = np.copy(node_positions_xy)
    estimated_positions_xy_hist[NUM_ANCHORS:] = avg_anchor_pos + np.random.randn(NUM_UNKNOWNS, 2) * 10
    current_estimated_positions_hist = np.copy(estimated_positions_xy_hist)

    for iteration in range(num_iterations):
        # Store positions from previous step to calculate delta
        previous_positions = np.copy(current_estimated_positions_hist)
        
        newly_estimated_positions_hist = np.copy(current_estimated_positions_hist)
        for i_unknown in range(NUM_ANCHORS, TOTAL_NODES):
            neighbor_pos = []
            neighbor_rssi = []
            for j_neighbor in range(TOTAL_NODES):
                if i_unknown == j_neighbor: continue
                rssi_val = rssi_matrix_avg[i_unknown, j_neighbor]
                if not np.isnan(rssi_val):
                    neighbor_pos.append(current_estimated_positions_hist[j_neighbor])
                    neighbor_rssi.append(rssi_val)
            
            if len(neighbor_pos) >= 3:
                est_pos = estimate_unknown_position_ls(
                    current_estimated_positions_hist[i_unknown], neighbor_pos, neighbor_rssi, A_global, n_global
                )
                if est_pos is not None:
                    newly_estimated_positions_hist[i_unknown] = est_pos
        
        current_estimated_positions_hist = newly_estimated_positions_hist
        
        # --- Record History for this iteration ---
        # 1. Localization Error
        errors_this_iter = np.linalg.norm(node_positions_xy[NUM_ANCHORS:] - current_estimated_positions_hist[NUM_ANCHORS:], axis=1)
        iterative_errors.append(errors_this_iter)
        
        # 2. Position Change (Delta)
        deltas_this_iter = np.linalg.norm(current_estimated_positions_hist[NUM_ANCHORS:] - previous_positions[NUM_ANCHORS:], axis=1)
        iterative_deltas.append(deltas_this_iter)

    iterative_errors = np.array(iterative_errors) # Shape: (num_iterations, num_unknowns)
    iterative_deltas = np.array(iterative_deltas)

    # --- Task 4.1: Visualize "Error Trajectory" of Individual Nodes ---
    print("Task 4.1: Plotting error trajectories for individual nodes...")
    final_errors = iterative_errors[-1]
    best_node_idx = np.argmin(final_errors)
    worst_node_idx = np.argmax(final_errors)
    median_node_idx = np.argsort(final_errors)[len(final_errors) // 2]

    plt.figure(figsize=(12, 8))
    plt.plot(range(1, num_iterations + 1), iterative_errors[:, worst_node_idx], 'r-o', label=f'Worst Node (Final Error: {final_errors[worst_node_idx]:.1f}m)')
    plt.plot(range(1, num_iterations + 1), iterative_errors[:, median_node_idx], 'b-s', label=f'Median Node (Final Error: {final_errors[median_node_idx]:.1f}m)')
    plt.plot(range(1, num_iterations + 1), iterative_errors[:, best_node_idx], 'g-^', label=f'Best Node (Final Error: {final_errors[best_node_idx]:.1f}m)')
    
    plt.title('Localization Error Trajectory for Individual Nodes', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Iteration Number', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Localization Error (m)', fontsize=FONT_SIZE_LABEL)
    plt.xticks(range(1, num_iterations + 1))
    plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE_TICKS)
    plt.legend(fontsize=FONT_SIZE_LABEL)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    output_path = os.path.join(OUTPUT_DIR, "task_4_1_error_trajectory.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

    # --- Task 4.2: Visualize System's "Error State" Over Time ---
    print("Task 4.2: Plotting overall system error and convergence...")
    mean_error_per_iter = np.mean(iterative_errors, axis=1)
    mean_delta_per_iter = np.mean(iterative_deltas, axis=1)

    fig, ax1 = plt.subplots(figsize=(12, 8))

    ax1.set_xlabel('Iteration Number', fontsize=FONT_SIZE_LABEL)
    ax1.set_ylabel('Average Localization Error (m)', fontsize=FONT_SIZE_LABEL, color='tab:blue')
    ax1.plot(range(1, num_iterations + 1), mean_error_per_iter, 'o-', color='tab:blue', label='Avg. Localization Error')
    ax1.tick_params(axis='y', labelcolor='tab:blue', labelsize=FONT_SIZE_TICKS)
    ax1.tick_params(axis='x', labelsize=FONT_SIZE_TICKS)
    ax1.set_xticks(range(1, num_iterations + 1))

    ax2 = ax1.twinx()
    ax2.set_ylabel('Average Position Change (m)', fontsize=FONT_SIZE_LABEL, color='tab:red')
    ax2.plot(range(1, num_iterations + 1), mean_delta_per_iter, 's--', color='tab:red', label='Avg. Position Change (Delta)')
    ax2.tick_params(axis='y', labelcolor='tab:red', labelsize=FONT_SIZE_TICKS)

    plt.title('System Convergence and Error State Over Iterations', fontsize=FONT_SIZE_TITLE)
    fig.tight_layout()
    fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9), bbox_transform=ax1.transAxes, fontsize=FONT_SIZE_LABEL)

    output_path = os.path.join(OUTPUT_DIR, "task_4_2_system_error_state.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

    print("\n--- Investigation script finished. ---")


if __name__ == '__main__':
    main()

