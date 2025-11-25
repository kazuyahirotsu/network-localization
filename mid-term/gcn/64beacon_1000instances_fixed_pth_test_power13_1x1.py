# %%
import torch
import torch.nn.functional as F
from torch.nn import Sequential, Linear, ReLU, Parameter
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import NNConv
from torch.utils.data import random_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from geopy.distance import geodesic
import numpy as np
from scipy.io import loadmat
from tqdm import tqdm
import math
import os

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Using device:', device)

#######################################
# Parameters and Utility Functions (Should match training)
#######################################
num_instances = 1000  # Or fewer for a quick test, e.g., 100
num_anchors = 16
num_unknowns = 48
num_measurements = 10

# Origin from MATLAB code
mapOriginLat = 40.466198
mapOriginLon = 33.898610
earthRadius = 6378137.0
metersPerDegreeLat = (math.pi / 180) * earthRadius
metersPerDegreeLon = (math.pi / 180) * earthRadius * np.cos(np.deg2rad(mapOriginLat))

def latlon_to_xy(lat, lon, originLat, originLon):
    x = (lon - originLon) * metersPerDegreeLon
    y = (lat - originLat) * metersPerDegreeLat
    return x, y

# --- Slide Font Sizes ---
SLIDE_SUPTITLE_FONT_SIZE = 30
SLIDE_TITLE_FONT_SIZE = 30
SLIDE_AXIS_LABEL_FONT_SIZE = 30
SLIDE_TICK_LABEL_FONT_SIZE = 30
SLIDE_LEGEND_FONT_SIZE = 30
# --- End Slide Font Sizes ---

#######################################
# Model Definition (Must match the saved model's architecture)
#######################################
class MainEdgeNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MainEdgeNet, self).__init__()
        self.mlp = Sequential(
            Linear(in_channels, 64),
            ReLU(),
            Linear(64, out_channels)
        )
    def forward(self, x):
        return self.mlp(x)

class MainGNN(torch.nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, output_dim):
        super(MainGNN, self).__init__()
        # These dimensions must match those used during training when the model was saved
        self.edge_nn1 = MainEdgeNet(edge_in_dim, node_in_dim * hidden_dim)
        self.conv1 = NNConv(node_in_dim, hidden_dim, self.edge_nn1, aggr='mean')

        self.edge_nn2 = MainEdgeNet(edge_in_dim, hidden_dim * hidden_dim)
        self.conv2 = NNConv(hidden_dim, hidden_dim, self.edge_nn2, aggr='mean')

        self.fc = Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.fc(x)
        return x

# These dimensions must match the training script
# Node features: x, y, is_anchor (3)
node_in_dim_loaded = 3
# Edge features: num_measurements + estimated_distance (10 + 1 = 11)
edge_in_dim_loaded = num_measurements + 1
hidden_dim_loaded = 64
output_dim_loaded = 2

main_gnn_loaded = MainGNN(node_in_dim=node_in_dim_loaded,
                          edge_in_dim=edge_in_dim_loaded,
                          hidden_dim=hidden_dim_loaded,
                          output_dim=output_dim_loaded).to(device)

# For the fixed-power model, Pt is saved as a float (dBm). Path loss exponent and offset are Parameters.
PT_DBM_TENSOR_LOADED = torch.tensor(13.0, device=device)  # will be overwritten by checkpoint value
path_loss_exponent_loaded = Parameter(torch.tensor(0.0, device=device))
offset_loaded = Parameter(torch.tensor(0.0, device=device))

# Initialize scalers - their attributes will be loaded
feature_scaler_loaded = StandardScaler()
y_scaler_loaded = StandardScaler()

#######################################
# Load Model and Parameters
#######################################
load_path = 'trained_localization_model_64beacons_1000instances_fixed_power13_1x1.pth'
print(f'Loading model and parameters from {load_path}...')
if torch.cuda.is_available() and device.type == 'cuda':
    checkpoint = torch.load(load_path)
else:
    checkpoint = torch.load(load_path, map_location=torch.device('cpu'))

main_gnn_loaded.load_state_dict(checkpoint['main_gnn_state_dict'])

# Load Pt in dBm (float) and convert to tensor on device
pt_dbm_value = checkpoint.get('Pt_dBm', 13.0)
PT_DBM_TENSOR_LOADED = torch.tensor(float(pt_dbm_value), device=device)

# For Parameter types, assign their .data attribute
path_loss_exponent_loaded.data = checkpoint['path_loss_exponent'].data.to(device)
offset_loaded.data = checkpoint['offset'].data.to(device)

# Load scaler parameters
feature_scaler_loaded.mean_ = checkpoint['feature_scaler_params']['mean_']
feature_scaler_loaded.scale_ = checkpoint['feature_scaler_params']['scale_']

y_scaler_loaded.mean_ = checkpoint['y_scaler_params']['mean_']
y_scaler_loaded.scale_ = checkpoint['y_scaler_params']['scale_']

print('Model and parameters loaded successfully.')

#######################################
# Display Loaded Path Loss Parameters
#######################################
print("\nLoaded Path Loss Model Parameters (Fixed Power):")
print(f"  Pt_dBm (Fixed Transmit Power): {PT_DBM_TENSOR_LOADED.item():.4f} dBm")
print(f"  Path Loss Exponent: {path_loss_exponent_loaded.item():.4f}")
print(f"  Offset: {offset_loaded.item():.4f}")

# Set the model to evaluation mode
main_gnn_loaded.eval()

#######################################
# Data Loading and Preprocessing for Testing (Should mirror training script)
#######################################
print("Loading and preprocessing test data...")
data_list_test = []

for instance_idx in tqdm(range(1, num_instances + 1), desc="Loading MATLAB data for testing"):
    filename = f"../../matlab/data/64beacons_1000instances_1x1/data_instance_{instance_idx}.mat"
    mat_data = loadmat(filename)
    nodeLatitudes = mat_data['nodeLatitudes'].flatten()
    nodeLongitudes = mat_data['nodeLongitudes'].flatten()
    signal_strength_matrix = mat_data['signal_strength_matrix']
    num_nodes = len(nodeLatitudes)
    node_x_coords = np.zeros(num_nodes)
    node_y_coords = np.zeros(num_nodes)
    for i in range(num_nodes):
        node_x_coords[i], node_y_coords[i] = latlon_to_xy(nodeLatitudes[i], nodeLongitudes[i], mapOriginLat, mapOriginLon)

    edge_index_list = []
    edge_attr_list = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and not np.isnan(signal_strength_matrix[i, j, 0]):
                edge_index_list.append([i, j])
                edge_attr_list.append(signal_strength_matrix[i, j, :])

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    anchor_mask = torch.zeros(num_nodes, dtype=torch.bool)
    anchor_mask[:num_anchors] = True
    unknown_mask = ~anchor_mask
    avg_anchor_x = node_x_coords[anchor_mask].mean()
    avg_anchor_y = node_y_coords[anchor_mask].mean()

    node_features_raw_list = []
    for i in range(num_nodes):
        is_anchor = 1 if i < num_anchors else 0
        if is_anchor:
            node_features_raw_list.append([node_x_coords[i], node_y_coords[i], is_anchor])
        else:
            node_features_raw_list.append([avg_anchor_x + np.random.randn()*10,
                                           avg_anchor_y + np.random.randn()*10,
                                           is_anchor])

    x_raw = torch.tensor(node_features_raw_list, dtype=torch.float)
    x_scaled = torch.tensor(feature_scaler_loaded.transform(x_raw.numpy()), dtype=torch.float)

    y_true_raw = torch.tensor(np.column_stack((node_x_coords, node_y_coords)), dtype=torch.float)
    y_true_scaled = torch.tensor(y_scaler_loaded.transform(y_true_raw.numpy()), dtype=torch.float)

    data = Data(x=x_scaled, edge_index=edge_index, edge_attr=edge_attr, y=y_true_scaled)
    data.anchor_mask = anchor_mask
    data.unknown_mask = unknown_mask
    data.orig_positions = y_true_raw
    data_list_test.append(data)

# Use a consistent way to get the test set
total_instances_test = len(data_list_test)
train_size_dummy = int(0.8 * total_instances_test)
test_size_final = total_instances_test - train_size_dummy
_, test_dataset_loaded = random_split(
    data_list_test,
    [train_size_dummy, test_size_final],
    generator=torch.Generator().manual_seed(42)
)

test_loader_loaded = DataLoader(test_dataset_loaded, batch_size=1, shuffle=False)
print(f"Test data loaded. Number of test samples: {len(test_loader_loaded)}")

#######################################
# Helper function from training (needed for inference)
#######################################
def estimate_distance_from_rssi(measured_rssi, Pt_param, ple_param, offset_param, epsilon=1e-6):
    ple_val = ple_param + epsilon if torch.abs(ple_param) < epsilon else ple_param
    divisor = 10 * ple_val
    if torch.abs(divisor) < epsilon:
        divisor = epsilon if divisor >= 0 else -epsilon
    exponent_val = (Pt_param + offset_param - measured_rssi) / divisor
    exponent_val = torch.clamp(exponent_val, -2.0, 5.0)
    dist = torch.pow(10.0, exponent_val)
    dist = torch.clamp(dist, min=epsilon)
    return dist

#######################################
# Evaluation with Loaded Model
#######################################
errors_gcn_loaded = []

# For visualization of one sample
plotted_sample = False

# Initialize list to store true and GCN estimated distances for edges
all_true_and_gcn_estimated_distances = []

# Store first-sample data for quick re-plotting without re-evaluation
first_sample_true_positions = None
first_sample_pred_positions = None
first_sample_anchor_mask = None
first_sample_unknown_mask = None

with torch.no_grad():
    for i, data in enumerate(tqdm(test_loader_loaded, desc="Evaluating with loaded model")):
        data = data.to(device)

        measured_RSSI = data.edge_attr.mean(dim=1)
        dist_estimated = estimate_distance_from_rssi(
            measured_RSSI,
            PT_DBM_TENSOR_LOADED.to(data.edge_index.device),
            path_loss_exponent_loaded.to(data.edge_index.device),
            offset_loaded.to(data.edge_index.device)
        )
        new_edge_attr = torch.cat([data.edge_attr.to(data.edge_index.device), dist_estimated.unsqueeze(1)], dim=1)

        # Collect true and GCN estimated distances for each edge
        true_node_positions_numpy = data.orig_positions.cpu().numpy()
        for k in range(data.edge_index.shape[1]):
            node_u_idx = data.edge_index[0, k].item()
            node_v_idx = data.edge_index[1, k].item()
            pos_u = true_node_positions_numpy[node_u_idx]
            pos_v = true_node_positions_numpy[node_v_idx]
            true_dist_uv = np.sqrt(np.sum((pos_u - pos_v)**2))
            gcn_estimated_dist_uv = dist_estimated[k].item()
            all_true_and_gcn_estimated_distances.append((true_dist_uv, gcn_estimated_dist_uv))

        out = main_gnn_loaded(data.x, data.edge_index, new_edge_attr)

        predicted_scaled = out.cpu().numpy()
        predicted_positions = y_scaler_loaded.inverse_transform(predicted_scaled)

        true_positions = data.orig_positions.cpu().numpy()

        predicted_positions[data.anchor_mask.cpu().numpy()] = true_positions[data.anchor_mask.cpu().numpy()]

        for idx in range(true_positions.shape[0]):
            if data.unknown_mask[idx]:
                true_pos = true_positions[idx]
                pred_pos = predicted_positions[idx]
                error = np.sqrt((true_pos[0] - pred_pos[0])**2 + (true_pos[1] - pred_pos[1])**2)
                errors_gcn_loaded.append(error)

        # Plot the first sample in the test set for visualization
        if i == 0 and not plotted_sample:
            # Save arrays for later plotting without recomputation
            first_sample_true_positions = true_positions.copy()
            first_sample_pred_positions = predicted_positions.copy()
            first_sample_anchor_mask = data.anchor_mask.cpu().numpy().copy()
            first_sample_unknown_mask = data.unknown_mask.cpu().numpy().copy()
            plt.figure(figsize=(12,10))
            # Plot true unknown positions
            plt.scatter(true_positions[data.unknown_mask.cpu(),0],
                        true_positions[data.unknown_mask.cpu(),1],
                        c='blue', label='True Unknown Positions', alpha=0.6, s=50)
            # Plot predicted unknown positions
            plt.scatter(predicted_positions[data.unknown_mask.cpu(),0],
                        predicted_positions[data.unknown_mask.cpu(),1],
                        c='red', marker='x', label='Predicted Unknown Positions', alpha=0.8, s=50)
            # Plot anchor nodes
            plt.scatter(true_positions[data.anchor_mask.cpu(),0],
                        true_positions[data.anchor_mask.cpu(),1],
                        c='green', marker='^', s=100, label='Anchor Nodes', edgecolors='black')

            # Draw red dashed lines between true and predicted positions for unknown nodes
            for node_idx in range(true_positions.shape[0]):
                if data.unknown_mask[node_idx]:
                    tp = true_positions[node_idx]
                    pp = predicted_positions[node_idx]
                    plt.plot([tp[0], pp[0]], [tp[1], pp[1]], 'r--', alpha=0.3)

            plt.xlabel('X Position (m)', fontsize=SLIDE_AXIS_LABEL_FONT_SIZE)
            plt.ylabel('Y Position (m)', fontsize=SLIDE_AXIS_LABEL_FONT_SIZE)
            plt.legend(fontsize=SLIDE_LEGEND_FONT_SIZE, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
            plt.xticks(fontsize=SLIDE_TICK_LABEL_FONT_SIZE)
            plt.yticks(fontsize=SLIDE_TICK_LABEL_FONT_SIZE)
            plt.grid(True)
            plt.axis('equal')
            os.makedirs("new_results", exist_ok=True)
            sample_viz_filename = f'new_results/sample_visualization_loaded_model_{load_path.split("/")[-1].replace(".pth","")}.png'
            plt.tight_layout()
            plt.savefig(sample_viz_filename, dpi=300, bbox_inches='tight')
            print(f'Sample visualization saved to {sample_viz_filename}')
            plt.show()
            plotted_sample = True

if errors_gcn_loaded:
    errors_gcn_loaded = np.array(errors_gcn_loaded)

    # Compute metrics
    mean_error_loaded = errors_gcn_loaded.mean()
    median_error_loaded = np.median(errors_gcn_loaded)
    p90_error_loaded = np.percentile(errors_gcn_loaded, 90)
    p95_error_loaded = np.percentile(errors_gcn_loaded, 95)

    print(
        f"Loaded GCN Errors - Mean: {mean_error_loaded:.4f} m, "
        f"Median: {median_error_loaded:.4f} m, "
        f"P90: {p90_error_loaded:.4f} m, P95: {p95_error_loaded:.4f} m"
    )

    # Ensure results directory exists
    os.makedirs("new_results", exist_ok=True)

    # Save metrics to a text file
    metrics_filename = f'new_results/metrics_loaded_model_{load_path.split("/")[-1].replace(".pth","")}.txt'
    with open(metrics_filename, 'w') as f:
        f.write("Localization Error Metrics (meters)\n")
        f.write(f"Model: {load_path.split('/')[-1]}\n")
        f.write(f"Mean: {mean_error_loaded:.6f}\n")
        f.write(f"Median: {median_error_loaded:.6f}\n")
        f.write(f"P90: {p90_error_loaded:.6f}\n")
        f.write(f"P95: {p95_error_loaded:.6f}\n")
        f.write(f"Num Samples: {errors_gcn_loaded.size}\n")
    print(f'Metrics saved to {metrics_filename}')

    # Plot histogram
    plt.figure(figsize=(12, 6))
    plt.hist(
        errors_gcn_loaded,
        bins=30,
        alpha=0.8,
        color='lightcoral',
        label='GCN Errors',
        edgecolor='black',
        linewidth=0.5
    )
    plt.xlabel('Localization Error (meters)', fontsize=SLIDE_AXIS_LABEL_FONT_SIZE)
    plt.ylabel('Number of Nodes', fontsize=SLIDE_AXIS_LABEL_FONT_SIZE)
    # No title per requirement
    plt.legend(fontsize=SLIDE_LEGEND_FONT_SIZE)
    plt.xticks(fontsize=SLIDE_TICK_LABEL_FONT_SIZE)
    plt.yticks(fontsize=SLIDE_TICK_LABEL_FONT_SIZE)
    plt.grid(True, alpha=0.3)

    # Overlay metrics text box
    metrics_text = (
        f"Mean: {mean_error_loaded:.2f} m\n"
        f"Median: {median_error_loaded:.2f} m\n"
        f"P90: {p90_error_loaded:.2f} m\n"
        f"P95: {p95_error_loaded:.2f} m"
    )
    ax = plt.gca()
    ax.text(
        0.98, 0.98, metrics_text,
        transform=ax.transAxes,
        va='top', ha='right',
        fontsize=SLIDE_LEGEND_FONT_SIZE,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    )

    error_hist_filename = f'new_results/error_histogram_loaded_model_{load_path.split("/")[-1].replace(".pth","")}.png'
    plt.tight_layout()
    plt.savefig(error_hist_filename, dpi=300, bbox_inches='tight')
    print(f'Error histogram saved to {error_hist_filename}')
    plt.show()

    # Plot Error CDF
    sorted_errors = np.sort(errors_gcn_loaded)
    cdf = np.arange(1, sorted_errors.size + 1) / sorted_errors.size

    plt.figure(figsize=(12, 6))
    plt.plot(sorted_errors, cdf, color='navy', linewidth=2, label='Empirical CDF')
    plt.xlabel('Localization Error (meters)', fontsize=SLIDE_AXIS_LABEL_FONT_SIZE)
    plt.ylabel('Cumulative Probability', fontsize=SLIDE_AXIS_LABEL_FONT_SIZE)
    # No title per requirement
    plt.xticks(fontsize=SLIDE_TICK_LABEL_FONT_SIZE)
    plt.yticks(fontsize=SLIDE_TICK_LABEL_FONT_SIZE)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.legend(fontsize=SLIDE_LEGEND_FONT_SIZE)
    error_cdf_filename = f'new_results/error_cdf_loaded_model_{load_path.split("/")[-1].replace(".pth","")}.png'
    plt.tight_layout()
    plt.savefig(error_cdf_filename, dpi=300, bbox_inches='tight')
    print(f'Error CDF saved to {error_cdf_filename}')
    plt.show()

    # Save the errors to a .npy file
    errors_filename = f'new_results/gcn_errors_{load_path.split("/")[-1].replace(".pth","")}.npy'
    np.save(errors_filename, errors_gcn_loaded)
    print(f'GCN errors saved to {errors_filename}')

    # Save the collected true vs. GCN estimated distances
    true_vs_estimated_dist_filename = f'new_results/gcn_true_vs_estimated_distances_{load_path.split("/")[-1].replace(".pth","")}.npy'
    np.save(true_vs_estimated_dist_filename, np.array(all_true_and_gcn_estimated_distances))
    print(f'GCN true vs. estimated distances saved to {true_vs_estimated_dist_filename}')

    # Save a comprehensive evaluation dump for fast re-plotting
    eval_dump_filename = f'new_results/eval_dump_{load_path.split("/")[-1].replace(".pth","")}.npz'
    np.savez_compressed(
        eval_dump_filename,
        model_file=np.array(load_path),
        pt_dbm=np.array(PT_DBM_TENSOR_LOADED.item(), dtype=np.float32),
        path_loss_exponent=np.array(path_loss_exponent_loaded.item(), dtype=np.float32),
        offset=np.array(offset_loaded.item(), dtype=np.float32),
        errors=errors_gcn_loaded.astype(np.float32),
        edge_true_vs_estimated_distances=np.array(all_true_and_gcn_estimated_distances, dtype=np.float32),
        first_true_positions=(first_sample_true_positions.astype(np.float32) if first_sample_true_positions is not None else np.array([])),
        first_pred_positions=(first_sample_pred_positions.astype(np.float32) if first_sample_pred_positions is not None else np.array([])),
        first_anchor_mask=(first_sample_anchor_mask.astype(np.bool_) if first_sample_anchor_mask is not None else np.array([], dtype=np.bool_)),
        first_unknown_mask=(first_sample_unknown_mask.astype(np.bool_) if first_sample_unknown_mask is not None else np.array([], dtype=np.bool_)),
        mean=np.array(mean_error_loaded, dtype=np.float32),
        median=np.array(median_error_loaded, dtype=np.float32),
        p90=np.array(p90_error_loaded, dtype=np.float32),
        p95=np.array(p95_error_loaded, dtype=np.float32)
    )
    print(f'Evaluation dump saved to {eval_dump_filename}')
else:
    print("No unknown nodes found in the test set to calculate errors or test_loader_loaded is empty.")

# %% [markdown]
# **How to use the .pth file (64beacon_1000instances_fixed_power13_pth_test.py)**
#
# This script demonstrates loading and utilizing the trained model and parameters saved in
# `trained_localization_model_64beacons_1000instances_fixed_power13.pth`, where the transmit
# power is fixed at 13 dBm and only the path loss exponent and offset are learned.
#
# Key points:
# - The GNN architecture and input dimensions must match the training configuration.
# - `Pt_dBm` is loaded as a float and used as a constant in the RSSI distance estimate.
# - `path_loss_exponent` and `offset` are restored as torch.nn.Parameter values.
# - The `StandardScaler` parameters for features and targets are also restored and used.
# - Data loading mirrors training; new edge attributes concatenate the estimated distance.
# - Evaluation saves a sample visualization, an error histogram, the raw error values, and
#   a NumPy array of (true_distance, estimated_distance) for each edge in the test graphs.


