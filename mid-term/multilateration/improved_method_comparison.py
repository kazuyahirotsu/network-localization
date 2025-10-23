import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
import os
import math
from tqdm import tqdm

# --- Configuration ---
NUM_INSTANCES_TO_PROCESS = 100
MATLAB_DATA_BASE_PATH = "matlab/data/64beacons_100instances/"
OUTPUT_DIR = "investigation_results_comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Node configuration
NUM_ANCHORS = 16
NUM_UNKNOWNS = 48
TOTAL_NODES = NUM_ANCHORS + NUM_UNKNOWNS

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


# --- Utility Functions (Adapted from previous scripts) ---
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

def estimate_rssi_parameters(anchor_positions_xy, anchor_rssi_matrix):
    distances, avg_rssis = [], []
    for i in range(anchor_positions_xy.shape[0]):
        for j in range(i + 1, anchor_positions_xy.shape[0]):
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
        rssi_matrix_avg = np.nanmean(mat_data['signal_strength_matrix'], axis=2)
        node_positions_xy = np.array([latlon_to_xy(lat, lon) for lat, lon in zip(node_latitudes, node_longitudes)])
        return node_positions_xy, rssi_matrix_avg
    except (FileNotFoundError, RuntimeWarning):
        return None, None

def run_localization(node_positions_xy, rssi_matrix_avg, A_param, n_param, use_rssi_filter=False):
    """Runs the iterative multilateration process."""
    num_iterations = 10
    estimated_positions = np.copy(node_positions_xy)
    avg_anchor_pos = np.mean(estimated_positions[:NUM_ANCHORS], axis=0)
    estimated_positions[NUM_ANCHORS:] = avg_anchor_pos + np.random.randn(NUM_UNKNOWNS, 2) * 10

    for _ in range(num_iterations):
        newly_estimated = np.copy(estimated_positions)
        for i_unknown in range(NUM_ANCHORS, TOTAL_NODES):
            neighbor_pos, neighbor_rssi = [], []
            for j in range(TOTAL_NODES):
                if i_unknown == j: continue
                rssi = rssi_matrix_avg[i_unknown, j]
                
                # The core logic change is here!
                should_use_link = (not use_rssi_filter and not np.isnan(rssi)) or \
                                  (use_rssi_filter and not np.isnan(rssi) and rssi > -135)
                
                if should_use_link:
                    neighbor_pos.append(estimated_positions[j])
                    neighbor_rssi.append(rssi)
            
            if len(neighbor_pos) >= 3:
                measured_distances = np.array([rssi_to_distance_log_model(r, A_param, n_param) for r in neighbor_rssi])
                valid_indices = np.isfinite(measured_distances)
                if np.sum(valid_indices) >= 3:
                    res = least_squares(
                        lambda pos, kn, dist: np.linalg.norm(kn - pos, axis=1) - dist,
                        newly_estimated[i_unknown],
                        args=(np.array(neighbor_pos)[valid_indices], measured_distances[valid_indices]),
                        bounds=([-1000, -1000], [5000, 5000]), method='trf'
                    )
                    if res.success: newly_estimated[i_unknown] = res.x
        estimated_positions = newly_estimated
    
    errors = np.linalg.norm(node_positions_xy[NUM_ANCHORS:] - estimated_positions[NUM_ANCHORS:], axis=1)
    return errors


# --- Main Comparison Script ---
def main():
    print("--- Comparing Original vs. Improved Localization Method ---")
    
    # Get global path-loss params from first instance
    pos_sample, rssi_sample = load_matlab_instance(1)
    if pos_sample is None:
        print("Could not load instance 1. Aborting.")
        return
    A_global, n_global = estimate_rssi_parameters(pos_sample[:NUM_ANCHORS], rssi_sample[:NUM_ANCHORS, :NUM_ANCHORS])
    print(f"Using Global Parameters: A={A_global:.2f}, n={n_global:.2f}\n")

    all_errors_original = []
    all_errors_improved = []

    print(f"Processing {NUM_INSTANCES_TO_PROCESS} instances for comparison...")
    for i in tqdm(range(1, NUM_INSTANCES_TO_PROCESS + 1)):
        node_pos, rssi_avg = load_matlab_instance(i)
        if node_pos is None: continue

        # Method A: Original
        errors_orig = run_localization(node_pos, rssi_avg, A_global, n_global, use_rssi_filter=False)
        all_errors_original.extend(errors_orig)

        # Method B: Improved (with RSSI filter)
        errors_imp = run_localization(node_pos, rssi_avg, A_global, n_global, use_rssi_filter=True)
        all_errors_improved.extend(errors_imp)
        
    # --- Print and Plot Results ---
    print("\n--- Results ---")
    print(f"Original Method  | Mean Error: {np.mean(all_errors_original):.2f}m, Median Error: {np.median(all_errors_original):.2f}m")
    print(f"Improved Method  | Mean Error: {np.mean(all_errors_improved):.2f}m, Median Error: {np.median(all_errors_improved):.2f}m")

    plt.figure(figsize=(14, 8))
    bins = np.linspace(0, max(np.percentile(all_errors_original, 95), np.percentile(all_errors_improved, 95)), 100)
    plt.hist(all_errors_original, bins=bins, alpha=0.7, label='Original Method (All Links)', color='orangered')
    plt.hist(all_errors_improved, bins=bins, alpha=0.7, label='Improved Method (RSSI > -135dBm)', color='dodgerblue')
    
    plt.title('Comparison of Localization Error Distributions', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('Localization Error (m)', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Number of Nodes', fontsize=FONT_SIZE_LABEL)
    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    plt.legend(fontsize=FONT_SIZE_LABEL)
    plt.grid(True)

    output_path = os.path.join(OUTPUT_DIR, "error_comparison_histogram.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nComparison plot saved to {output_path}")

if __name__ == '__main__':
    main() 