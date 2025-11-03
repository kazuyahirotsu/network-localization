# %%
import numpy as np
from scipy.io import loadmat
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import os

# %%
# Parameters (matching MATLAB and GCN script where applicable)
num_instances = 200  # Or 1000, depending on your dataset
num_anchors = 16 # As per your current selection
num_unknowns = 48 # As per your current selection
total_nodes = num_anchors + num_unknowns

num_measurements = 10 # Number of RSSI measurements per node pair from MATLAB script

# Threshold (meters) for allowing unknown nodes to be used as neighbors.
# An unknown node is eligible only if its previous iteration's LS residual RMS
# is less than or equal to this value.
UNKNOWN_NEIGHBOR_RMS_THRESHOLD = 50.0

# Tag to differentiate outputs from 64beacon_200instances.py
RUN_TAG = f"rmsgate_{int(UNKNOWN_NEIGHBOR_RMS_THRESHOLD)}m"

# Map origin from MATLAB code
mapOriginLat = 40.466198
mapOriginLon = 33.898610
earthRadius = 6378137.0
metersPerDegreeLat = (math.pi / 180) * earthRadius
metersPerDegreeLon = (math.pi / 180) * earthRadius * np.cos(np.deg2rad(mapOriginLat))

# Adjust this path if your data is elsewhere
matlab_data_base_path = "../../matlab/data/64beacons_100instances/" # As per your current selection
output_dir = "output_visualizations_matlab_im"
os.makedirs(output_dir, exist_ok=True)

# --- Create unique identifiers for output files ---
# Extract a dataset name from the path (e.g., "mid_16beacons")
dataset_name = os.path.basename(os.path.normpath(matlab_data_base_path))
# Create a node configuration string (e.g., "16N_4A")
node_config_str = f"{total_nodes}N_{num_anchors}A"
# --- End unique identifiers ---

# --- Directory for GIF frames ---
GIF_FRAMES_SUBDIR = f"{dataset_name}_{node_config_str}_{RUN_TAG}_im_instance_1_gif_frames"
GIF_FRAMES_PATH = os.path.join(output_dir, GIF_FRAMES_SUBDIR)
os.makedirs(GIF_FRAMES_PATH, exist_ok=True)
# --- End Directory for GIF frames ---

# --- Slide Font Sizes ---
SLIDE_SUPTITLE_FONT_SIZE = 30
SLIDE_TITLE_FONT_SIZE = 30
SLIDE_AXIS_LABEL_FONT_SIZE = 30
SLIDE_TICK_LABEL_FONT_SIZE = 30
SLIDE_LEGEND_FONT_SIZE = 30
# --- End Slide Font Sizes ---

# %%
# Utility Functions
def latlon_to_xy(lat, lon, originLat, originLon):
    x = (lon - originLon) * metersPerDegreeLon
    y = (lat - originLat) * metersPerDegreeLat
    return x, y

def rssi_to_distance_log_model(rssi, A_param, n_param):
    """Converts RSSI to distance using the log-distance path loss model.
    RSSI(d) = A_param - 10 * n_param * log10(d)
    log10(d) = (A_param - RSSI(d)) / (10 * n_param)
    d = 10**((A_param - RSSI(d)) / (10 * n_param))
    
    Args:
        rssi (float): Received Signal Strength Indicator.
        A_param (float): RSSI at a reference distance of 1 meter (dBm).
        n_param (float): Path loss exponent.
    Returns:
        float: Estimated distance in meters.
    """
    # Add a small epsilon to rssi if it's too close to A_param to avoid log10(0) or large numbers if n_param is small
    epsilon_rssi = 1e-6 
    if n_param == 0: # Avoid division by zero
        return float('inf') # Or handle as an error/invalid parameter
    
    power_val = (A_param - (rssi + epsilon_rssi)) / (10 * n_param)
    
    # Cap power_val to prevent overflow if (A_param - rssi) is very large
    # For example, if power_val > 30 (10^30 meters), it's likely an outlier.
    if power_val > 30: # Corresponds to 10^30 meters, which is astronomically large
        return float('inf') # Or a very large, but finite number like 1e5 meters (100km)
        
    distance = 10**power_val
    return distance

# %%
from scipy.optimize import curve_fit

def log_distance_model_for_fit(d, A, n):
    """Model function for curve_fit: RSSI = A - 10*n*log10(d)"""
    # Ensure d is positive to avoid log10(0) or log10(negative)
    d_safe = np.maximum(d, 1e-9) # Avoid d=0, use a very small number instead (e.g., 1mm)
    return A - 10 * n * np.log10(d_safe)

def estimate_rssi_parameters(anchor_positions_xy, anchor_rssi_matrix):
    """
    Estimates RSSI model parameters A and n using anchor-to-anchor measurements.
    Args:
        anchor_positions_xy (np.array): Nx2 array of known X,Y positions of N anchors.
        anchor_rssi_matrix (np.array): NxN_orig_measurements array of RSSI values between anchors.
                                        (e.g., N_orig_measurements could be num_measurements)
    Returns:
        tuple: (A_est, n_est) - Estimated A and n parameters.
               Returns (None, None) if estimation fails.
    """
    distances = []
    avg_rssis = []
    num_current_anchors = anchor_positions_xy.shape[0]

    for i in range(num_current_anchors):
        for j in range(i + 1, num_current_anchors): # Use i+1 to avoid duplicates and self-loops
            dist_ij = np.linalg.norm(anchor_positions_xy[i] - anchor_positions_xy[j])
            if dist_ij == 0: # Should not happen if i != j
                continue

            # Assuming anchor_rssi_matrix is (num_anchors, num_anchors, num_measurements)
            # Or if it's pre-averaged: (num_anchors, num_anchors)
            if anchor_rssi_matrix.ndim == 3:
                rssi_ij_measurements = anchor_rssi_matrix[i, j, :]
                rssi_ji_measurements = anchor_rssi_matrix[j, i, :]
                
                # Filter out NaNs before averaging if any
                valid_rssi_ij = rssi_ij_measurements[~np.isnan(rssi_ij_measurements)]
                valid_rssi_ji = rssi_ji_measurements[~np.isnan(rssi_ji_measurements)]
                
                if len(valid_rssi_ij) > 0:
                    avg_rssis.append(np.mean(valid_rssi_ij))
                    distances.append(dist_ij)
                if len(valid_rssi_ji) > 0: # If measurements are directional and different
                    avg_rssis.append(np.mean(valid_rssi_ji))
                    distances.append(dist_ij) # distance is symmetric
            elif anchor_rssi_matrix.ndim == 2: # Assumes already averaged or single measurement
                 rssi_ij = anchor_rssi_matrix[i,j]
                 rssi_ji = anchor_rssi_matrix[j,i] # Could be symmetric
                 if not np.isnan(rssi_ij):
                    avg_rssis.append(rssi_ij)
                    distances.append(dist_ij)
                 if i !=j and not np.isnan(rssi_ji) and rssi_ij != rssi_ji : # if not symmetric and not already added
                    avg_rssis.append(rssi_ji)
                    distances.append(dist_ij)


    if len(distances) < 2: # Need at least 2 points to fit A and n
        # print("Warning: Not enough valid anchor-to-anchor measurements to estimate RSSI parameters.")
        # Fallback to default/placeholder values if too few points
        return -50.0, 3.0 # Or raise an error

    distances = np.array(distances)
    avg_rssis = np.array(avg_rssis)

    # Bounds for A (e.g., -100 to 50 dBm) and n (e.g., 1.5 to 6)
    # These bounds help curve_fit find more realistic parameters.
    # A is RSSI at 1m. For typical indoor/outdoor, it's often negative.
    # n (path loss exponent) typically ranges from 2 (free space) to 4-6 (obstructed).
    param_bounds = ([-100, 1.5], [50, 6.0])
    
    # Initial guess for A and n
    # A can be roughly the max RSSI observed, n can be 2-3.
    p0 = [np.max(avg_rssis) if len(avg_rssis) > 0 else -50.0, 3.0]


    try:
        params, covariance = curve_fit(log_distance_model_for_fit, distances, avg_rssis, p0=p0, bounds=param_bounds, maxfev=5000)
        A_est, n_est = params
        # Add checks for reasonableness of estimated parameters
        if not (param_bounds[0][0] <= A_est <= param_bounds[1][0] and \
                param_bounds[0][1] <= n_est <= param_bounds[1][1]):
            # print(f"Warning: Estimated RSSI parameters {A_est, n_est} are outside typical bounds. Falling back.")
            return -50.0, 3.0 # Fallback
        return A_est, n_est
    except RuntimeError:
        # print("Warning: curve_fit failed to converge for RSSI parameter estimation. Falling back to defaults.")
        return -50.0, 3.0 # Fallback to default values
    except ValueError as e:
        # print(f"Warning: ValueError during RSSI parameter estimation: {e}. Falling back to defaults.")
        return -50.0, 3.0 # Fallback

# %%
def load_matlab_instance(instance_idx, base_path=matlab_data_base_path):
    """Loads data for a single MATLAB instance."""
    filename = os.path.join(base_path, f"data_instance_{instance_idx}.mat")
    try:
        mat_data = loadmat(filename)
    except FileNotFoundError:
        print(f"Error: File not found {filename}")
        return None, None, None

    nodeLatitudes = mat_data['nodeLatitudes'].flatten()
    nodeLongitudes = mat_data['nodeLongitudes'].flatten()
    # signal_strength_matrix shape from MATLAB: (totalNodes, totalNodes, numSimulations)
    signal_strength_matrix_all_sims = mat_data['signal_strength_matrix']
    
    # Average RSSI across simulations for simplicity in this multilateration model
    # Or, one could use all measurements and treat them as distinct inputs to LS
    # For now, let's average. The GCN model also seems to average or use all (needs check)
    # GCN script: measured_RSSI = data.edge_attr.mean(dim=1) -> averages over num_measurements
    rssi_matrix_avg = np.nanmean(signal_strength_matrix_all_sims, axis=2)


    num_nodes_in_instance = len(nodeLatitudes)
    if num_nodes_in_instance != total_nodes:
        print(f"Warning: Instance {instance_idx} has {num_nodes_in_instance} nodes, expected {total_nodes}.")
        # Decide how to handle this: skip, error, or adapt. For now, let's try to adapt.
    
    node_positions_xy = np.zeros((num_nodes_in_instance, 2))
    for i in range(num_nodes_in_instance):
        node_positions_xy[i, 0], node_positions_xy[i, 1] = latlon_to_xy(
            nodeLatitudes[i], nodeLongitudes[i], mapOriginLat, mapOriginLon
        )
        
    return node_positions_xy, rssi_matrix_avg, signal_strength_matrix_all_sims

# %%
def multilateration_error_func(position_guess, known_positions, measured_distances):
    """
    Error function for least_squares multilateration.
    Args:
        position_guess (np.array): Current [x, y] guess for the unknown node.
        known_positions (np.array): Kx2 array of known neighbor positions.
        measured_distances (np.array): K-element array of measured distances to neighbors.
    Returns:
        np.array: K-element array of errors (calculated_dist - measured_dist).
    """
    errors = []
    for i in range(known_positions.shape[0]):
        calculated_dist = np.linalg.norm(position_guess - known_positions[i])
        errors.append(calculated_dist - measured_distances[i])
    return np.array(errors)

def estimate_unknown_position_ls(initial_guess_xy, neighbor_positions_xy, neighbor_rssi_values, A_param, n_param):
    """
    Estimates the position of an unknown node using least squares multilateration.
    Args:
        initial_guess_xy (np.array): Initial [x,y] guess for the unknown node.
        neighbor_positions_xy (list of np.array): List of [x,y] for known neighbor positions.
        neighbor_rssi_values (list of float): List of average RSSI values from neighbors.
        A_param (float): Instance-specific RSSI model parameter A.
        n_param (float): Instance-specific RSSI model parameter n.
    Returns:
        tuple: (estimated_xy, residual_rms)
               estimated_xy: np.array [x,y] or None if estimation fails.
               residual_rms: float RMS of LS residuals (meters), np.inf if fail.
    """
    if len(neighbor_positions_xy) < 2: # Need at least 2 for 2D, 3 for less ambiguity
        # print("Warning: Fewer than 2 neighbors for multilateration.")
        return None, np.inf 

    measured_distances = [rssi_to_distance_log_model(rssi, A_param, n_param) for rssi in neighbor_rssi_values]
    
    # Filter out inf distances which can happen if RSSI is too low / params are off
    valid_indices = [i for i, d in enumerate(measured_distances) if np.isfinite(d) and d < 1e4] # Cap max distance e.g. 10km
    
    if len(valid_indices) < 2:
        # print("Warning: Fewer than 2 valid distance measurements after filtering.")
        return None, np.inf
        
    known_pos_filtered = np.array([neighbor_positions_xy[i] for i in valid_indices])
    measured_dist_filtered = np.array([measured_distances[i] for i in valid_indices])

    if known_pos_filtered.shape[0] < 2: # Still need at least 2 points
        return None, np.inf

    # Bounds for x, y (e.g., based on map size if known, otherwise can be loose)
    # Assuming the MATLAB script's mapSizeMeters = 4000
    # And positions are relative to origin, so roughly 0 to 4000.
    # Looser bounds: -1000 to 5000 for some margin
    bounds = ([-1000, -1000], [5000, 5000]) 


    result = least_squares(
        multilateration_error_func,
        initial_guess_xy,
        args=(known_pos_filtered, measured_dist_filtered),
        bounds=bounds,
        method='trf', # Trust Region Reflective, good for bounds
        ftol=1e-5, xtol=1e-5, gtol=1e-5 # Tighter tolerances
    )

    if result.success : #and result.cost < 1e3: # Add a cost check if needed
        residuals = result.fun if result.fun is not None else np.array([])
        residual_rms = float(np.sqrt(np.mean(residuals**2))) if residuals.size > 0 else 0.0
        return result.x, residual_rms
    else:
        # print(f"Warning: Least squares optimization failed. Status: {result.status}, Message: {result.message}")
        return None, np.inf

# %%
all_instance_errors = []
num_iterations = 10 # Number of iterations for the iterative multilateration

# Store first instance true/pred positions for plotting later
first_true_positions = None
first_pred_positions = None

for idx_instance in tqdm(range(1, num_instances + 1), desc="Processing Instances"):
    node_positions_xy_true, rssi_matrix_avg_true, _ = load_matlab_instance(idx_instance)
    if node_positions_xy_true is None:
        continue

    true_anchor_positions_xy = node_positions_xy_true[:num_anchors]
    
    anchor_to_anchor_rssi_matrix = rssi_matrix_avg_true[:num_anchors, :num_anchors]
    
    A_inst, n_inst = estimate_rssi_parameters(true_anchor_positions_xy, anchor_to_anchor_rssi_matrix)
    
    if A_inst is None or n_inst is None:
        A_inst, n_inst = -50.0, 3.0 

    estimated_positions_xy = np.copy(node_positions_xy_true) 
    avg_anchor_pos = np.mean(true_anchor_positions_xy, axis=0)
    for i in range(num_anchors, total_nodes):
        estimated_positions_xy[i] = avg_anchor_pos + np.random.randn(2) * 10 
    
    current_estimated_positions_for_iter = np.copy(estimated_positions_xy)
    # Track per-node LS residual RMS from the previous iteration for neighbor gating
    last_rms_errors = np.full(total_nodes, np.inf)
    last_rms_errors[:num_anchors] = 0.0

    for iteration in range(num_iterations):
        newly_estimated_positions_this_iter = np.copy(current_estimated_positions_for_iter)
        new_rms_errors = np.copy(last_rms_errors)
        
        for i_unknown in range(num_anchors, total_nodes):
            neighbor_pos_for_mlat = []
            neighbor_rssi_for_mlat = []
            
            for j_known in range(total_nodes):
                if i_unknown == j_known:
                    continue
                
                rssi_val = rssi_matrix_avg_true[i_unknown, j_known] 
                
                if not np.isnan(rssi_val) and np.isfinite(rssi_val):
                    # Allow anchors always; allow unknowns only if prior RMS is within threshold
                    if j_known < num_anchors or last_rms_errors[j_known] <= UNKNOWN_NEIGHBOR_RMS_THRESHOLD:
                        neighbor_pos_for_mlat.append(current_estimated_positions_for_iter[j_known])
                        neighbor_rssi_for_mlat.append(rssi_val)
            
            if len(neighbor_pos_for_mlat) >= 3: 
                initial_guess = current_estimated_positions_for_iter[i_unknown] 
                
                estimated_pos, residual_rms = estimate_unknown_position_ls(
                    initial_guess,
                    neighbor_pos_for_mlat,
                    neighbor_rssi_for_mlat,
                    A_inst,
                    n_inst
                )
                if estimated_pos is not None:
                    newly_estimated_positions_this_iter[i_unknown] = estimated_pos
                    new_rms_errors[i_unknown] = residual_rms
        
        current_estimated_positions_for_iter = np.copy(newly_estimated_positions_this_iter)
        last_rms_errors = new_rms_errors

    # Capture first instance sample positions for plotting
    if first_true_positions is None:
        try:
            first_true_positions = np.copy(node_positions_xy_true)
            first_pred_positions = np.copy(current_estimated_positions_for_iter)
        except Exception:
            pass

    instance_node_errors = []
    for i in range(num_anchors, total_nodes):
        true_pos = node_positions_xy_true[i]
        est_pos = current_estimated_positions_for_iter[i]
        error = np.linalg.norm(true_pos - est_pos)
        instance_node_errors.append(error)
    all_instance_errors.extend(instance_node_errors) # Corrected this line

# --- Plotting and Final Evaluation ---
all_instance_errors = np.array(all_instance_errors)

plt.figure(figsize=(10, 6))
plt.hist(all_instance_errors, bins=50, alpha=0.7, color='coral', label=f'Iterative Multilateration ({dataset_name} - {node_config_str})') # Updated label
plt.xlabel('Localization Error (meters)', fontsize=SLIDE_AXIS_LABEL_FONT_SIZE)
plt.ylabel('Number of Unknown Nodes', fontsize=SLIDE_AXIS_LABEL_FONT_SIZE)
plt.title(f'Error Distribution ({dataset_name} - {node_config_str}, {num_iterations} Iterations)', fontsize=SLIDE_TITLE_FONT_SIZE) # Updated title
plt.legend(fontsize=SLIDE_LEGEND_FONT_SIZE)
plt.xticks(fontsize=SLIDE_TICK_LABEL_FONT_SIZE)
plt.yticks(fontsize=SLIDE_TICK_LABEL_FONT_SIZE)
plt.grid(True)
plt.tight_layout() # Add tight_layout
plt.savefig(os.path.join(output_dir, f"{dataset_name}_{node_config_str}_{RUN_TAG}_im_error_histogram.png"))
plt.show()

mean_error_im_matlab = np.mean(all_instance_errors) if len(all_instance_errors) > 0 else float('nan')
median_error_im_matlab = np.median(all_instance_errors) if len(all_instance_errors) > 0 else float('nan')
std_error_im_matlab = np.std(all_instance_errors) if len(all_instance_errors) > 0 else float('nan')

# Also compute P90/P95 and save a compact evaluation dump for plotting later
p90_error_im_matlab = np.percentile(all_instance_errors, 90) if len(all_instance_errors) > 0 else float('nan')
p95_error_im_matlab = np.percentile(all_instance_errors, 95) if len(all_instance_errors) > 0 else float('nan')

print(f"Iterative Multilateration ({dataset_name} - {node_config_str} - {num_iterations} iterations):") # Updated print
print(f"  Mean Error: {mean_error_im_matlab:.4f} m")
print(f"  Median Error: {median_error_im_matlab:.4f} m")
print(f"  Std Dev Error: {std_error_im_matlab:.4f} m")

# Save results for later plotting
os.makedirs('new_results', exist_ok=True)

metrics_txt_path = os.path.join('new_results', f"metrics_multilateration_{dataset_name}_{node_config_str}_{RUN_TAG}.txt")
with open(metrics_txt_path, 'w') as f:
    f.write('Localization Error Metrics (meters)\n')
    f.write(f"Model: mlat_{dataset_name}\n")
    f.write(f"Mean: {mean_error_im_matlab:.6f}\n")
    f.write(f"Median: {median_error_im_matlab:.6f}\n")
    f.write(f"P90: {p90_error_im_matlab:.6f}\n")
    f.write(f"P95: {p95_error_im_matlab:.6f}\n")
    f.write(f"Num Samples: {all_instance_errors.size}\n")
print(f"Metrics saved to: {metrics_txt_path}")

# Save raw errors to .npy file for later comparison
if len(all_instance_errors) > 0: # Ensure there are errors to save
    error_save_filename = os.path.join(output_dir, f"{dataset_name}_{node_config_str}_{RUN_TAG}_im_simple_errors_raw.npy")
    np.save(error_save_filename, all_instance_errors)
    print(f"Raw localization errors saved to: {error_save_filename}")
else:
    print("No errors were recorded, so no error file was saved.")

# Save evaluation dump for fast plotting
anchor_mask = np.zeros(total_nodes, dtype=bool)
anchor_mask[:num_anchors] = True
unknown_mask = ~anchor_mask

eval_dump_filename = os.path.join('new_results', f"eval_dump_mlat_{dataset_name}_{node_config_str}_{RUN_TAG}.npz")
np.savez_compressed(
    eval_dump_filename,
    model_file=np.array(f"mlat_{dataset_name}_{RUN_TAG}"),
    errors=all_instance_errors.astype(np.float32),
    first_true_positions=(first_true_positions.astype(np.float32) if first_true_positions is not None else np.array([])),
    first_pred_positions=(first_pred_positions.astype(np.float32) if first_pred_positions is not None else np.array([])),
    first_anchor_mask=anchor_mask,
    first_unknown_mask=unknown_mask,
    mean=np.array(mean_error_im_matlab, dtype=np.float32),
    median=np.array(median_error_im_matlab, dtype=np.float32),
    p90=np.array(p90_error_im_matlab, dtype=np.float32),
    p95=np.array(p95_error_im_matlab, dtype=np.float32)
)
print(f"Evaluation dump saved to: {eval_dump_filename}")

# Optional: Visualize one instance
if num_instances > 0:
    instance_to_viz = 1 
    node_pos_true_viz, rssi_avg_viz, _ = load_matlab_instance(instance_to_viz, base_path=matlab_data_base_path)
    if node_pos_true_viz is not None:
        true_anchor_pos_viz = node_pos_true_viz[:num_anchors]
        anchor_rssi_matrix_viz = rssi_avg_viz[:num_anchors, :num_anchors]
        A_viz, n_viz = estimate_rssi_parameters(true_anchor_pos_viz, anchor_rssi_matrix_viz)
        if A_viz is None: A_viz, n_viz = -50.0, 3.0

        est_pos_viz_initial = np.copy(node_pos_true_viz) # Store initial state for the loop
        avg_anchor_viz = np.mean(true_anchor_pos_viz, axis=0)
        for i in range(num_anchors, total_nodes):
            est_pos_viz_initial[i] = avg_anchor_viz + np.random.randn(2) * 10
            
        current_est_pos_viz_iter = np.copy(est_pos_viz_initial)
        # Track per-node LS residual RMS for neighbor gating in visualization loop
        last_rms_errors_viz = np.full(total_nodes, np.inf)
        last_rms_errors_viz[:num_anchors] = 0.0
        
        print(f"\nSaving frames for GIF animation to: {GIF_FRAMES_PATH}")
        for iter_num_viz in tqdm(range(num_iterations), desc=f"Generating GIF frames for Instance {instance_to_viz}"):
            newly_est_pos_viz_this_iter = np.copy(current_est_pos_viz_iter)
            new_rms_errors_viz = np.copy(last_rms_errors_viz)
            for i_unknown in range(num_anchors, total_nodes):
                neighbors_p = []
                neighbors_r = []
                for j_known in range(total_nodes):
                    if i_unknown == j_known: continue
                    rssi_val = rssi_avg_viz[i_unknown, j_known]
                    if not np.isnan(rssi_val) and np.isfinite(rssi_val):
                        if j_known < num_anchors or last_rms_errors_viz[j_known] <= UNKNOWN_NEIGHBOR_RMS_THRESHOLD:
                            neighbors_p.append(current_est_pos_viz_iter[j_known])
                            neighbors_r.append(rssi_val)
                if len(neighbors_p) >= 3:
                    est_p, residual_rms_viz = estimate_unknown_position_ls(current_est_pos_viz_iter[i_unknown], neighbors_p, neighbors_r, A_viz, n_viz)
                    if est_p is not None:
                        newly_est_pos_viz_this_iter[i_unknown] = est_p
                        new_rms_errors_viz[i_unknown] = residual_rms_viz
            current_est_pos_viz_iter = newly_est_pos_viz_this_iter
            last_rms_errors_viz = new_rms_errors_viz
            
            # --- Plot and Save Frame for GIF ---
            plt.figure(figsize=(12, 10))
            plt.scatter(node_pos_true_viz[num_anchors:, 0], node_pos_true_viz[num_anchors:, 1], c='blue', label='True Unknown Positions', alpha=0.6, s=50)
            # Use current_est_pos_viz_iter for the estimated positions in this frame
            plt.scatter(current_est_pos_viz_iter[num_anchors:, 0], current_est_pos_viz_iter[num_anchors:, 1], c='red', marker='x', label="Predicted Unknown Positions", alpha=0.8, s=50)
            plt.scatter(node_pos_true_viz[:num_anchors, 0], node_pos_true_viz[:num_anchors, 1], c='green', marker='^', s=100, label='Anchor Nodes', edgecolors='black')

            for i_node_viz in range(num_anchors, total_nodes): # Renamed loop variable to avoid conflict
                plt.plot([node_pos_true_viz[i_node_viz,0], current_est_pos_viz_iter[i_node_viz,0]], [node_pos_true_viz[i_node_viz,1], current_est_pos_viz_iter[i_node_viz,1]], 'r--', alpha=0.3)
            
            plt.xlabel('X Position (m)', fontsize=SLIDE_AXIS_LABEL_FONT_SIZE)
            plt.ylabel('Y Position (m)', fontsize=SLIDE_AXIS_LABEL_FONT_SIZE)
            # plt.title(f'Instance {instance_to_viz} - Iteration {iter_num_viz + 1}/{num_iterations}', fontsize=SLIDE_TITLE_FONT_SIZE)
            plt.legend(fontsize=SLIDE_LEGEND_FONT_SIZE, loc='upper right')
            plt.xticks(fontsize=SLIDE_TICK_LABEL_FONT_SIZE)
            plt.yticks(fontsize=SLIDE_TICK_LABEL_FONT_SIZE)
            plt.grid(True)
            plt.axis('equal')
            plt.xlim(np.min(node_pos_true_viz[:,0]) - 500, np.max(node_pos_true_viz[:,0]) + 500) # Example: Adjust limits dynamically or set fixed ones
            plt.ylim(np.min(node_pos_true_viz[:,1]) - 500, np.max(node_pos_true_viz[:,1]) + 500)
            plt.tight_layout()
            frame_filename = os.path.join(GIF_FRAMES_PATH, f"frame_{iter_num_viz:03d}.png")
            plt.savefig(frame_filename)
            plt.close() # Close the figure to free memory
            # --- End Plot and Save Frame ---
        
        # The final state visualization is now effectively the last frame of the GIF series
        # So, the original final plot can be removed or commented out if num_iterations > 0
        # For now, let's keep it to ensure the single image is still produced as before if num_iterations=0 or for comparison
        final_estimated_pos_viz = current_est_pos_viz_iter # This is the final state

        # (Original final plot logic - this will plot the final state again)
        plt.figure(figsize=(12, 10))
        plt.scatter(node_pos_true_viz[num_anchors:, 0], node_pos_true_viz[num_anchors:, 1], c='blue', label='True Unknown Positions', alpha=0.6, s=50)
        plt.scatter(final_estimated_pos_viz[num_anchors:, 0], final_estimated_pos_viz[num_anchors:, 1], c='red', marker='x', label="Predicted Unknown Positions", alpha=0.8, s=50)
        plt.scatter(node_pos_true_viz[:num_anchors, 0], node_pos_true_viz[:num_anchors, 1], c='green', marker='^', s=100, label='Anchor Nodes', edgecolors='black')

        for i in range(num_anchors, total_nodes):
            plt.plot([node_pos_true_viz[i,0], final_estimated_pos_viz[i,0]], [node_pos_true_viz[i,1], final_estimated_pos_viz[i,1]], 'r--', alpha=0.3)
        
        plt.xlabel('X Position (m)', fontsize=SLIDE_AXIS_LABEL_FONT_SIZE)
        plt.ylabel('Y Position (m)', fontsize=SLIDE_AXIS_LABEL_FONT_SIZE)
        # plt.title(f'Instance {instance_to_viz}: Final State ({dataset_name} - {node_config_str}, {num_iterations} iter)', fontsize=SLIDE_TITLE_FONT_SIZE)
        plt.legend(fontsize=SLIDE_LEGEND_FONT_SIZE, loc='upper right')
        plt.xticks(fontsize=SLIDE_TICK_LABEL_FONT_SIZE)
        plt.yticks(fontsize=SLIDE_TICK_LABEL_FONT_SIZE)
        plt.grid(True)
        plt.axis('equal')
        plt.xlim(np.min(node_pos_true_viz[:,0]) - 500, np.max(node_pos_true_viz[:,0]) + 500) # Match limits with GIF frames for consistency
        plt.ylim(np.min(node_pos_true_viz[:,1]) - 500, np.max(node_pos_true_viz[:,1]) + 500)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_{node_config_str}_{RUN_TAG}_im_instance_{instance_to_viz}_visualization.png"))
        plt.show()

# %%



