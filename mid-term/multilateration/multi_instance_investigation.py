import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
import os
import math
from tqdm import tqdm

# --- Configuration ---
NUM_INSTANCES_TO_PROCESS = 1000
MATLAB_DATA_BASE_PATH = "matlab/data/64beacons_100instances/"
OUTPUT_DIR = "investigation_results_multi_instance_dense"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Node configuration
NUM_ANCHORS = 16
NUM_UNKNOWNS = 48
TOTAL_NODES = NUM_ANCHORS + NUM_UNKNOWNS
MAP_SIZE = 4000.0

# Map origin for coordinate conversion
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


# --- Utility Functions (Copied from single-instance script) ---
def latlon_to_xy(lat, lon):
    x = (lon - MAP_ORIGIN_LON) * METERS_PER_DEGREE_LON
    y = (lat - MAP_ORIGIN_LAT) * METERS_PER_DEGREE_LAT
    return x, y

def log_distance_model_for_fit(d, A, n):
    d_safe = np.maximum(d, 1e-9)
    return A - 10 * n * np.log10(d_safe)

def rssi_to_distance_log_model(rssi, A_param, n_param):
    if n_param == 0: return float('inf')
    power_val = (A_param - rssi) / (10 * n_param)
    if power_val > 30: return float('inf')
    return 10**power_val

def multilateration_error_func(position_guess, known_positions, measured_distances):
    errors = np.linalg.norm(known_positions - position_guess, axis=1) - measured_distances
    return errors

def estimate_unknown_position_ls(initial_guess_xy, neighbor_positions_xy, neighbor_rssi_values, A_param, n_param):
    measured_distances = np.array([rssi_to_distance_log_model(rssi, A_param, n_param) for rssi in neighbor_rssi_values])
    valid_indices = np.isfinite(measured_distances) & (measured_distances < 1e4)
    if np.sum(valid_indices) < 3: return None
    known_pos_filtered = np.array(neighbor_positions_xy)[valid_indices]
    measured_dist_filtered = measured_distances[valid_indices]
    bounds = ([-1000, -1000], [5000, 5000])
    result = least_squares(multilateration_error_func, initial_guess_xy, args=(known_pos_filtered, measured_dist_filtered), bounds=bounds, method='trf', ftol=1e-5, xtol=1e-5, gtol=1e-5)
    return result.x if result.success else None

def estimate_rssi_parameters(anchor_positions_xy, anchor_rssi_matrix):
    distances, avg_rssis = [], []
    num_current_anchors = anchor_positions_xy.shape[0]
    for i in range(num_current_anchors):
        for j in range(i + 1, num_current_anchors):
            dist_ij = np.linalg.norm(anchor_positions_xy[i] - anchor_positions_xy[j])
            if dist_ij > 0:
                rssi_ij = anchor_rssi_matrix[i, j]
                if not np.isnan(rssi_ij):
                    distances.append(dist_ij)
                    avg_rssis.append(rssi_ij)
    if len(distances) < 2: return -50.0, 3.0
    try:
        params, _ = curve_fit(log_distance_model_for_fit, np.array(distances), np.array(avg_rssis), p0=[-50.0, 3.0], bounds=([-100, 1.5], [50, 6.0]), maxfev=5000)
        return params[0], params[1]
    except (RuntimeError, ValueError):
        return -50.0, 3.0

def load_matlab_instance(instance_idx):
    filename = os.path.join(MATLAB_DATA_BASE_PATH, f"data_instance_{instance_idx}.mat")
    try:
        mat_data = loadmat(filename)
        node_latitudes = mat_data['nodeLatitudes'].flatten()
        node_longitudes = mat_data['nodeLongitudes'].flatten()
        signal_strength_matrix = mat_data['signal_strength_matrix']
        rssi_matrix_avg = np.nanmean(signal_strength_matrix, axis=2)
        node_positions_xy = np.array([latlon_to_xy(lat, lon) for lat, lon in zip(node_latitudes, node_longitudes)])
        return node_positions_xy, rssi_matrix_avg
    except (FileNotFoundError, RuntimeWarning):
        return None, None

# --- Main Investigation Script ---
def main():
    # --- Data Aggregation ---
    all_residuals = []
    all_final_errors_with_pos = [] # To store tuples of (x, y, error)
    all_avg_link_errors_with_pos = [] # To store (x, y, avg_link_dist_error)
    all_valid_link_counts_with_pos = [] # To store (x, y, num_valid_links)
    
    # We need A_global, n_global first. Since anchors are fixed, we can get this from the first instance.
    node_positions_xy_sample, rssi_matrix_avg_sample = load_matlab_instance(1)
    if node_positions_xy_sample is None:
        print("Error: Could not load instance 1 to determine global parameters. Aborting.")
        return
    anchor_positions_xy = node_positions_xy_sample[:NUM_ANCHORS]
    anchor_rssi_matrix = rssi_matrix_avg_sample[:NUM_ANCHORS, :NUM_ANCHORS]
    A_global, n_global = estimate_rssi_parameters(anchor_positions_xy, anchor_rssi_matrix)
    print(f"Using Global Parameters (from Instance 1 anchors): A={A_global:.2f}, n={n_global:.2f}\n")

    print(f"Processing {NUM_INSTANCES_TO_PROCESS} instances...")
    for instance_idx in tqdm(range(1, NUM_INSTANCES_TO_PROCESS + 1)):
        node_positions_xy, rssi_matrix_avg = load_matlab_instance(instance_idx)
        if node_positions_xy is None: continue

        # --- Aggregate for Task 1.2: Residuals Histogram ---
        for i in range(TOTAL_NODES):
            for j in range(i + 1, TOTAL_NODES):
                rssi = rssi_matrix_avg[i, j]
                if not np.isnan(rssi):
                    true_dist = np.linalg.norm(node_positions_xy[i] - node_positions_xy[j])
                    predicted_rssi = log_distance_model_for_fit(true_dist, A_global, n_global)
                    all_residuals.append(predicted_rssi - rssi)

        # --- Calculate Link Quality and Counts for each unknown node ---
        true_positions_unknowns = node_positions_xy[NUM_ANCHORS:]
        for i_unknown_local_idx, i_unknown_global_idx in enumerate(range(NUM_ANCHORS, TOTAL_NODES)):
            link_errors = []
            num_valid_links = 0
            for j_neighbor in range(TOTAL_NODES):
                if i_unknown_global_idx == j_neighbor: continue
                rssi = rssi_matrix_avg[i_unknown_global_idx, j_neighbor]
                
                # A link is "valid" if the signal is strong enough to be heard by the receiver.
                if not np.isnan(rssi) and rssi > -135:
                    num_valid_links += 1
                    
                    # For this valid link, calculate the distance estimation error
                    estimated_dist = rssi_to_distance_log_model(rssi, A_global, n_global)
                    if np.isfinite(estimated_dist):
                        true_dist = np.linalg.norm(node_positions_xy[i_unknown_global_idx] - node_positions_xy[j_neighbor])
                        link_errors.append(abs(true_dist - estimated_dist))
            
            # Store results for this node
            node_pos = true_positions_unknowns[i_unknown_local_idx]
            if link_errors:
                all_avg_link_errors_with_pos.append((node_pos[0], node_pos[1], np.mean(link_errors)))
            else:
                all_avg_link_errors_with_pos.append((node_pos[0], node_pos[1], 0))
            all_valid_link_counts_with_pos.append((node_pos[0], node_pos[1], num_valid_links))


        # --- Run Localization to get errors for Task 3.1 ---
        num_iterations = 10
        estimated_positions_xy = np.copy(node_positions_xy)
        avg_anchor_pos = np.mean(anchor_positions_xy, axis=0)
        estimated_positions_xy[NUM_ANCHORS:] = avg_anchor_pos + np.random.randn(NUM_UNKNOWNS, 2) * 10
        
        current_estimated_positions = np.copy(estimated_positions_xy)
        for _ in range(num_iterations):
            newly_estimated_positions = np.copy(current_estimated_positions)
            for i_unknown in range(NUM_ANCHORS, TOTAL_NODES):
                neighbor_pos = [current_estimated_positions[j] for j in range(TOTAL_NODES) if i_unknown != j and not np.isnan(rssi_matrix_avg[i_unknown, j])]
                neighbor_rssi = [rssi_matrix_avg[i_unknown, j] for j in range(TOTAL_NODES) if i_unknown != j and not np.isnan(rssi_matrix_avg[i_unknown, j])]
                if len(neighbor_pos) >= 3:
                    est_pos = estimate_unknown_position_ls(current_estimated_positions[i_unknown], neighbor_pos, neighbor_rssi, A_global, n_global)
                    if est_pos is not None: newly_estimated_positions[i_unknown] = est_pos
            current_estimated_positions = newly_estimated_positions
        
        final_errors = np.linalg.norm(node_positions_xy[NUM_ANCHORS:] - current_estimated_positions[NUM_ANCHORS:], axis=1)
        true_positions_unknowns_loc = node_positions_xy[NUM_ANCHORS:]
        for i in range(NUM_UNKNOWNS):
            all_final_errors_with_pos.append((true_positions_unknowns_loc[i, 0], true_positions_unknowns_loc[i, 1], final_errors[i]))

    # --- Plotting Aggregated Data ---
    
    # Task 1.2: Aggregated Residuals Histogram
    print("\nPlotting Aggregated Residuals Histogram...")
    plt.figure(figsize=(12, 8))
    plt.hist(all_residuals, bins=150, density=True, alpha=0.75, label='Aggregated Model Residuals')
    
    mean_residual = np.mean(all_residuals)
    std_residual = np.std(all_residuals)
    
    plt.axvline(mean_residual, color='r', linestyle='--', linewidth=2, label=f'Mean = {mean_residual:.2f}')
    plt.title('Aggregated Model Residuals (All Links, 50 Instances)', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Residual RSSI (dBm)', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Probability Density', fontsize=FONT_SIZE_LABEL)
    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    plt.legend(fontsize=FONT_SIZE_LABEL)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.text(0.95, 0.95, f'Std Dev = {std_residual:.2f}', transform=plt.gca().transAxes,
             fontsize=FONT_SIZE_LABEL, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    output_path = os.path.join(OUTPUT_DIR, "agg_task_1_2_model_residuals_histogram.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

    # Task 3.1: Aggregated Spatial Error Heatmap
    print("\nPlotting Aggregated Spatial Error Heatmap...")
    grid_size = 50 # 50x50 grid
    heatmap = np.zeros((grid_size, grid_size))
    counts = np.zeros((grid_size, grid_size))
    bin_size = MAP_SIZE / grid_size

    for x, y, error in all_final_errors_with_pos:
        col = int(x / bin_size)
        row = int(y / bin_size)
        if 0 <= col < grid_size and 0 <= row < grid_size:
            heatmap[row, col] += error
            counts[row, col] += 1
    
    # Avoid division by zero
    counts[counts == 0] = 1
    average_heatmap = heatmap / counts
    
    plt.figure(figsize=(12, 10))
    # Origin='lower' to match XY coordinates to matrix indices
    im = plt.imshow(average_heatmap, cmap='viridis', origin='lower', extent=[0, MAP_SIZE, 0, MAP_SIZE])
    
    cbar = plt.colorbar(im)
    cbar.set_label('Average Localization Error (m)', fontsize=FONT_SIZE_LABEL)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICKS)
    
    # Plot anchor positions on top for context
    plt.scatter(anchor_positions_xy[:, 0], anchor_positions_xy[:, 1], c='red', marker='^', s=100, label='Anchor Nodes (Fixed)', edgecolors='black')

    plt.title(f'Aggregated Spatial Error Heatmap ({NUM_INSTANCES_TO_PROCESS} Instances)', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('X Position (m)', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Y Position (m)', fontsize=FONT_SIZE_LABEL)
    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    plt.legend(fontsize=FONT_SIZE_LABEL)
    plt.grid(False)

    output_path = os.path.join(OUTPUT_DIR, "agg_task_3_1_spatial_error_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

    # Task 5.1: Aggregated Distance Estimation Error Heatmap
    print("\nPlotting Aggregated Distance Estimation Error Heatmap...")
    heatmap_dist_error = np.zeros((grid_size, grid_size))
    counts_dist_error = np.zeros((grid_size, grid_size))
    for x, y, error in all_avg_link_errors_with_pos:
        col = int(x / bin_size)
        row = int(y / bin_size)
        if 0 <= col < grid_size and 0 <= row < grid_size:
            heatmap_dist_error[row, col] += error
            counts_dist_error[row, col] += 1
    counts_dist_error[counts_dist_error == 0] = 1
    average_heatmap_dist_error = heatmap_dist_error / counts_dist_error
    
    plt.figure(figsize=(12, 10))
    im = plt.imshow(average_heatmap_dist_error, cmap='magma', origin='lower', extent=[0, MAP_SIZE, 0, MAP_SIZE])
    cbar = plt.colorbar(im)
    cbar.set_label('Average Distance Estimation Error (m)', fontsize=FONT_SIZE_LABEL)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICKS)
    plt.title(f'Aggregated Distance Estimation Error ({NUM_INSTANCES_TO_PROCESS} Instances)', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('X Position (m)', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Y Position (m)', fontsize=FONT_SIZE_LABEL)
    plt.grid(False)
    output_path = os.path.join(OUTPUT_DIR, "agg_task_5_1_dist_error_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

    # Task 5.2: Aggregated Valid Link Count Heatmap
    print("\nPlotting Aggregated Valid Link Count Heatmap...")
    heatmap_link_count = np.zeros((grid_size, grid_size))
    counts_link_count = np.zeros((grid_size, grid_size))
    for x, y, count in all_valid_link_counts_with_pos:
        col = int(x / bin_size)
        row = int(y / bin_size)
        if 0 <= col < grid_size and 0 <= row < grid_size:
            heatmap_link_count[row, col] += count
            counts_link_count[row, col] += 1
    counts_link_count[counts_link_count == 0] = 1
    average_heatmap_link_count = heatmap_link_count / counts_link_count
    
    plt.figure(figsize=(12, 10))
    im = plt.imshow(average_heatmap_link_count, cmap='viridis', origin='lower', extent=[0, MAP_SIZE, 0, MAP_SIZE])
    cbar = plt.colorbar(im)
    cbar.set_label('Average Number of Valid Links', fontsize=FONT_SIZE_LABEL)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICKS)
    plt.title(f'Aggregated Valid Link Count ({NUM_INSTANCES_TO_PROCESS} Instances)', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('X Position (m)', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Y Position (m)', fontsize=FONT_SIZE_LABEL)
    plt.grid(False)
    output_path = os.path.join(OUTPUT_DIR, "agg_task_5_2_valid_link_count_heatmap.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {output_path}")

    print("\n--- Multi-instance investigation finished. ---")


if __name__ == '__main__':
    main() 