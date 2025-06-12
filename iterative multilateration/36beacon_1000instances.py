# %%
import numpy as np
from scipy.io import loadmat
from scipy.optimize import least_squares, curve_fit
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import os

# %%
# Parameters (matching MATLAB and GCN script where applicable)
num_instances_to_load_for_param_est = 1 # Number of instances to use for global A, n estimation (max: num_instances)
num_instances_to_process = 100  # Total instances for localization evaluation

num_anchors = 16 # MODIFIED as per user change in attached file
num_unknowns = 48 # MODIFIED as per user change in attached file
total_nodes = num_anchors + num_unknowns

num_measurements = 10 # Number of RSSI measurements per node pair from MATLAB script

# Map origin from MATLAB code
mapOriginLat = 40.466198
mapOriginLon = 33.898610
earthRadius = 6378137.0
metersPerDegreeLat = (math.pi / 180) * earthRadius
metersPerDegreeLon = (math.pi / 180) * earthRadius * np.cos(np.deg2rad(mapOriginLat))

# Adjust this path if your data is elsewhere
matlab_data_base_path = "../matlab/data/64beacons_100instances/" # As per your current selection
output_dir = "output_visualizations_matlab_im"
os.makedirs(output_dir, exist_ok=True)

# --- Create unique identifiers for output files ---
dataset_name = os.path.basename(os.path.normpath(matlab_data_base_path))
node_config_str = f"{total_nodes}N_{num_anchors}A"
# --- End unique identifiers ---

# --- Debugging --- 
debug_instance_indices = [1, 2, 3] # Instances to generate debug prints for multilateration inputs
plot_global_rssi_fit = True # Whether to plot the global A, n fit
# --- End Debugging --- 

# --- Option to use hardcoded GCN parameters --- ADDED BLOCK
use_hardcoded_gcn_params = True # Set to True to use GCN learned, False to estimate from anchors
A_gcn_learned = -50.0086
n_gcn_learned = 3.0228
param_source_str = "gcn_learned_params" if use_hardcoded_gcn_params else "estimated_params"
# --- End Hardcoded GCN parameters --- ADDED BLOCK


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
    epsilon_rssi = 1e-6 
    if n_param == 0: 
        return float('inf') 
    
    power_val = (A_param - (rssi + epsilon_rssi)) / (10 * n_param)
    
    if power_val > 30: 
        return float('inf') 
        
    distance = 10**power_val
    return distance

# %%
# from scipy.optimize import curve_fit # Moved to top imports

def log_distance_model_for_fit(d, A, n):
    """Model function for curve_fit: RSSI = A - 10*n*log10(d)"""
    d_safe = np.maximum(d, 1e-9) 
    return A - 10 * n * np.log10(d_safe)

# MODIFIED: estimate_rssi_parameters now takes aggregated data for global fit
def estimate_global_rssi_parameters(all_anchor_distances, all_anchor_rssi_measurements, plot_fit=False):
    """
    Estimates global RSSI model parameters A and n using aggregated anchor-to-anchor measurements.
    Args:
        all_anchor_distances (list): List of all anchor-to-anchor distances.
        all_anchor_rssi_measurements (list): List of all corresponding RSSI measurements.
        plot_fit (bool): Whether to plot the fitted curve.
    Returns:
        tuple: (A_glob, n_glob) - Estimated global A and n parameters.
    """
    if len(all_anchor_distances) < 2: 
        print(f"  Global RSSI Param Est: Not enough valid anchor-to-anchor measurements ({len(all_anchor_distances)} points). Falling back.")
        return -50.0, 3.0 

    distances_np = np.array(all_anchor_distances)
    avg_rssis_np = np.array(all_anchor_rssi_measurements)   

    param_bounds = ([-120, 1.0], [30, 7.0]) # Adjusted bounds for potentially wider RSSI range and n
    # Initial guess can be more data-driven if many points are available
    p0_A = np.median(avg_rssis_np[distances_np < np.median(distances_np)]) if len(avg_rssis_np)>0 else -50.0
    p0_n = 3.0
    p0 = [p0_A, p0_n]

    A_glob, n_glob = -50.0, 3.0 
    fit_successful = False 

    try:
        params, covariance = curve_fit(log_distance_model_for_fit, distances_np, avg_rssis_np, p0=p0, bounds=param_bounds, maxfev=10000)
        A_est_cand, n_est_cand = params
        if (param_bounds[0][0] <= A_est_cand <= param_bounds[1][0] and \
                param_bounds[0][1] <= n_est_cand <= param_bounds[1][1]):
            A_glob, n_glob = A_est_cand, n_est_cand
            fit_successful = True 
        else:
            print(f"  Global RSSI Param Est: Estimated params {A_est_cand, n_est_cand} out of bounds. Falling back.")
    except RuntimeError:
        print(f"  Global RSSI Param Est: curve_fit RuntimeError. Falling back.")
    except ValueError as e:
        print(f"  Global RSSI Param Est: curve_fit ValueError: {e}. Falling back.")

    print(f"Global RSSI Param Est: Using A={A_glob:.2f}, n={n_glob:.2f} (Fit successful: {fit_successful})")
    
    if plot_fit:
        print(f"  Number of data points for global fit: {len(distances_np)}")
        plt.figure(figsize=(10,7))
        # Plot a subset of points if too many, for clarity
        sample_size_plot = min(len(distances_np), 1000)
        sample_indices = np.random.choice(len(distances_np), sample_size_plot, replace=False)
        
        plt.scatter(distances_np[sample_indices], avg_rssis_np[sample_indices], label='Anchor Measurements (Sampled)', color='blue', alpha=0.3, s=10)
        
        if len(distances_np) > 0:
            dist_range_for_plot = np.logspace(np.log10(max(1e-1, np.min(distances_np[distances_np > 0]))), np.log10(np.max(distances_np)), 100)
            rssi_fitted_curve = log_distance_model_for_fit(dist_range_for_plot, A_glob, n_glob)
            plt.plot(dist_range_for_plot, rssi_fitted_curve, label=f'Global Fitted Model (A={A_glob:.2f}, n={n_glob:.2f})', color='red', linewidth=2)
        
        plt.xlabel('Distance (m) - Log Scale')
        plt.ylabel('RSSI (dBm)')
        plt.xscale('log') # Plot distance on log scale for better visualization of log-distance model
        plt.title(f'Global RSSI vs. Distance (All Anchor Links & Fit) - {dataset_name} {node_config_str}')
        plt.legend()
        plt.grid(True, which="both", ls="--")
        plot_filename = os.path.join(output_dir, f"{dataset_name}_{node_config_str}_global_rssi_fit.png")
        plt.savefig(plot_filename)
        plt.show()

    return A_glob, n_glob

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
    signal_strength_matrix_all_sims = mat_data['signal_strength_matrix']
    rssi_matrix_avg = np.nanmean(signal_strength_matrix_all_sims, axis=2)

    num_nodes_in_instance = len(nodeLatitudes)
    if num_nodes_in_instance != total_nodes:
        print(f"Warning: Instance {instance_idx} has {num_nodes_in_instance} nodes, expected {total_nodes}.")
    
    node_positions_xy = np.zeros((num_nodes_in_instance, 2))
    for i in range(num_nodes_in_instance):
        node_positions_xy[i, 0], node_positions_xy[i, 1] = latlon_to_xy(
            nodeLatitudes[i], nodeLongitudes[i], mapOriginLat, mapOriginLon
        )
        
    return node_positions_xy, rssi_matrix_avg, signal_strength_matrix_all_sims

# %%
def multilateration_error_func(position_guess, known_positions, measured_distances):
    errors = []
    for i in range(known_positions.shape[0]):
        calculated_dist = np.linalg.norm(position_guess - known_positions[i])
        errors.append(calculated_dist - measured_distances[i])
    return np.array(errors)

def estimate_unknown_position_ls(initial_guess_xy, neighbor_positions_xy, neighbor_rssi_values, A_param, n_param):
    if len(neighbor_positions_xy) < 2: 
        return None 

    measured_distances = [rssi_to_distance_log_model(rssi, A_param, n_param) for rssi in neighbor_rssi_values]
    valid_indices = [i for i, d in enumerate(measured_distances) if np.isfinite(d) and d < 1e4] 
    
    if len(valid_indices) < 2:
        return None
        
    known_pos_filtered = np.array([neighbor_positions_xy[i] for i in valid_indices])
    measured_dist_filtered = np.array([measured_distances[i] for i in valid_indices])

    if known_pos_filtered.shape[0] < 2: 
        return None

    bounds = ([-1000, -1000], [5000, 5000]) 

    result = least_squares(
        multilateration_error_func,
        initial_guess_xy,
        args=(known_pos_filtered, measured_dist_filtered),
        bounds=bounds,
        method='trf', 
        ftol=1e-5, xtol=1e-5, gtol=1e-5 
    )

    if result.success : 
        return result.x
    else:
        return None

# %% 
# --- Step 1 & 2: Estimate or Set global A, n parameters --- MODIFIED BLOCK
if use_hardcoded_gcn_params:
    A_glob = A_gcn_learned
    n_glob = n_gcn_learned
    print(f"Using hardcoded GCN-learned parameters: A={A_glob:.4f}, n={n_glob:.4f}")
else:
    all_anchor_distances_for_fit = []
    all_anchor_rssi_measurements_for_fit = []
    print("Aggregating anchor data for RSSI model parameter estimation...")
    
    node_positions_xy_sample, _, _ = load_matlab_instance(1) 
    if node_positions_xy_sample is None:
        raise ValueError("Could not load sample instance 1 to get anchor positions for parameter estimation.")
    true_anchor_positions_xy_global = node_positions_xy_sample[:num_anchors]

    for idx_instance in tqdm(range(1, num_instances_to_load_for_param_est + 1), desc="Loading Anchor Data"):
        _, _, signal_strength_all_sims_inst = load_matlab_instance(idx_instance)
        if signal_strength_all_sims_inst is None:
            continue
        
        anchor_to_anchor_rssi_all_sims = signal_strength_all_sims_inst[:num_anchors, :num_anchors, :]

        for i in range(num_anchors):
            for j in range(i + 1, num_anchors):
                dist_ij = np.linalg.norm(true_anchor_positions_xy_global[i] - true_anchor_positions_xy_global[j])
                if dist_ij == 0:
                    continue
                
                rssi_ij_all = anchor_to_anchor_rssi_all_sims[i, j, :]
                rssi_ji_all = anchor_to_anchor_rssi_all_sims[j, i, :]
                
                valid_rssi_ij = rssi_ij_all[~np.isnan(rssi_ij_all)]
                valid_rssi_ji = rssi_ji_all[~np.isnan(rssi_ji_all)]
                
                for rssi_val in valid_rssi_ij:
                    all_anchor_distances_for_fit.append(dist_ij)
                    all_anchor_rssi_measurements_for_fit.append(rssi_val)
                for rssi_val in valid_rssi_ji:
                    all_anchor_distances_for_fit.append(dist_ij)
                    all_anchor_rssi_measurements_for_fit.append(rssi_val)
    
    A_glob, n_glob = estimate_global_rssi_parameters(all_anchor_distances_for_fit, all_anchor_rssi_measurements_for_fit, plot_fit=plot_global_rssi_fit)

print(f"Proceeding with A={A_glob:.4f}, n={n_glob:.4f} for localization.")
# --- End Step 1 & 2 -- MODIFIED BLOCK

# --- Step 3: Main Iterative Multilateration Loop using global A, n ---
all_instance_errors = []
num_iterations = 10 

for idx_instance in tqdm(range(1, num_instances_to_process + 1), desc="Processing Instances for Localization"):
    node_positions_xy_true, rssi_matrix_avg_true, _ = load_matlab_instance(idx_instance)
    if node_positions_xy_true is None:
        continue

    # Anchor positions are known and true
    true_anchor_positions_xy_inst = node_positions_xy_true[:num_anchors]
    
    estimated_positions_xy = np.copy(node_positions_xy_true) 
    avg_anchor_pos = np.mean(true_anchor_positions_xy_inst, axis=0)
    for i in range(num_anchors, total_nodes):
        estimated_positions_xy[i] = avg_anchor_pos + np.random.randn(2) * 10 
    
    current_estimated_positions_for_iter = np.copy(estimated_positions_xy)

    for iteration in range(num_iterations):
        newly_estimated_positions_this_iter = np.copy(current_estimated_positions_for_iter)
        
        for i_unknown in range(num_anchors, total_nodes):
            neighbor_pos_for_mlat = []
            neighbor_rssi_for_mlat = []
            
            for j_known in range(total_nodes):
                if i_unknown == j_known:
                    continue
                
                rssi_val = rssi_matrix_avg_true[i_unknown, j_known] 
                
                if not np.isnan(rssi_val) and np.isfinite(rssi_val):
                    neighbor_pos_for_mlat.append(current_estimated_positions_for_iter[j_known])
                    neighbor_rssi_for_mlat.append(rssi_val)
            
            if len(neighbor_pos_for_mlat) >= 3: 
                initial_guess = current_estimated_positions_for_iter[i_unknown] 
                
                if idx_instance in debug_instance_indices and i_unknown == num_anchors and iteration == 0: 
                    print(f"  Instance {idx_instance}, Unknown Node {i_unknown}, Iteration {iteration}: Multilateration Inputs")
                    for k_neighbor in range(len(neighbor_pos_for_mlat)):
                        rssi_val_debug = neighbor_rssi_for_mlat[k_neighbor]
                        dist_derived = rssi_to_distance_log_model(rssi_val_debug, A_glob, n_glob) # Use global A, n
                        print(f"    Neighbor {k_neighbor} (Pos: {neighbor_pos_for_mlat[k_neighbor]}): RSSI={rssi_val_debug:.2f} -> DerivedDist={dist_derived:.2f}m")

                estimated_pos = estimate_unknown_position_ls(
                    initial_guess,
                    neighbor_pos_for_mlat,
                    neighbor_rssi_for_mlat,
                    A_glob, # Use global A
                    n_glob  # Use global n
                )
                if estimated_pos is not None:
                    newly_estimated_positions_this_iter[i_unknown] = estimated_pos
        
        current_estimated_positions_for_iter = np.copy(newly_estimated_positions_this_iter)

    instance_node_errors = []
    for i in range(num_anchors, total_nodes):
        true_pos = node_positions_xy_true[i]
        est_pos = current_estimated_positions_for_iter[i]
        error = np.linalg.norm(true_pos - est_pos)
        instance_node_errors.append(error)
    if instance_node_errors: # Ensure list is not empty before extending
        all_instance_errors.extend(instance_node_errors) 

# --- Plotting and Final Evaluation ---
if all_instance_errors: # Check if list is not empty
    all_instance_errors_np = np.array(all_instance_errors) # MODIFIED: rename to avoid conflict

    plt.figure(figsize=(10, 6))
    plt.hist(all_instance_errors_np, bins=50, alpha=0.7, color='coral', label=f'Iterative Multilateration ({dataset_name} - {node_config_str})') 
    plt.xlabel('Localization Error (meters)')
    plt.ylabel('Number of Unknown Nodes')
    plt.title(f'Error Distribution ({dataset_name} - {node_config_str}, {num_iterations} Iterations, Params: {param_source_str})') 
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{dataset_name}_{node_config_str}_im_error_histogram_{param_source_str}.png")) # MODIFIED
    plt.show()

    mean_error_im_matlab = np.mean(all_instance_errors_np) 
    median_error_im_matlab = np.median(all_instance_errors_np) 
    std_error_im_matlab = np.std(all_instance_errors_np) 
else:
    print("No errors recorded, possibly due to all instances failing or no unknown nodes processed.")
    mean_error_im_matlab = median_error_im_matlab = std_error_im_matlab = float('nan')

print(f"Iterative Multilateration ({dataset_name} - {node_config_str} - {num_iterations} iterations, Params: {param_source_str}):") 
print(f"  Mean Error: {mean_error_im_matlab:.4f} m")
print(f"  Median Error: {median_error_im_matlab:.4f} m")
print(f"  Std Dev Error: {std_error_im_matlab:.4f} m")

# Optional: Visualize one instance
if num_instances_to_process > 0 and len(debug_instance_indices)>0: 
    instance_to_viz = debug_instance_indices[0] 
    node_pos_true_viz, rssi_avg_viz, _ = load_matlab_instance(instance_to_viz, base_path=matlab_data_base_path) 
    if node_pos_true_viz is not None:
        A_viz, n_viz = A_glob, n_glob # Use global parameters for visualization consistency

        est_pos_viz = np.copy(node_pos_true_viz)
        avg_anchor_viz = np.mean(node_pos_true_viz[:num_anchors], axis=0)
        for i in range(num_anchors, total_nodes):
            est_pos_viz[i] = avg_anchor_viz + np.random.randn(2) * 10
            
        current_est_pos_viz_iter = np.copy(est_pos_viz)
        for _ in range(num_iterations):
            newly_est_pos_viz_this_iter = np.copy(current_est_pos_viz_iter)
            for i_unknown in range(num_anchors, total_nodes):
                neighbors_p = []
                neighbors_r = []
                for j_known in range(total_nodes):
                    if i_unknown == j_known: continue
                    rssi_val = rssi_avg_viz[i_unknown, j_known]
                    if not np.isnan(rssi_val) and np.isfinite(rssi_val):
                        neighbors_p.append(current_est_pos_viz_iter[j_known])
                        neighbors_r.append(rssi_val)
                if len(neighbors_p) >= 3:
                    est_p = estimate_unknown_position_ls(current_est_pos_viz_iter[i_unknown], neighbors_p, neighbors_r, A_viz, n_viz)
                    if est_p is not None:
                        newly_est_pos_viz_this_iter[i_unknown] = est_p
            current_est_pos_viz_iter = newly_est_pos_viz_this_iter
        
        final_estimated_pos_viz = current_est_pos_viz_iter

        plt.figure(figsize=(12, 10))
        plt.scatter(node_pos_true_viz[num_anchors:, 0], node_pos_true_viz[num_anchors:, 1], c='blue', label='True Unknown Positions', alpha=0.6, s=50)
        plt.scatter(final_estimated_pos_viz[num_anchors:, 0], final_estimated_pos_viz[num_anchors:, 1], c='red', marker='x', label=f'Est. Unknown ({num_iterations} iter, Params: {param_source_str})', alpha=0.8, s=50) # MODIFIED
        plt.scatter(node_pos_true_viz[:num_anchors, 0], node_pos_true_viz[:num_anchors, 1], c='green', marker='^', s=100, label='Anchor Nodes', edgecolors='black')

        for i in range(num_anchors, total_nodes):
            plt.plot([node_pos_true_viz[i,0], final_estimated_pos_viz[i,0]], [node_pos_true_viz[i,1], final_estimated_pos_viz[i,1]], 'r--', alpha=0.3)
        
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title(f'Instance {instance_to_viz}: True vs. Estimated ({dataset_name} - {node_config_str}, Params: {param_source_str})') # MODIFIED
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.savefig(os.path.join(output_dir, f"{dataset_name}_{node_config_str}_im_instance_{instance_to_viz}_visualization_{param_source_str}.png")) # MODIFIED
        plt.show()

# %%



