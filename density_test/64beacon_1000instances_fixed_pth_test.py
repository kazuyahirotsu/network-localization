# %% # type: ignore
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

#######################################
# Parameters and Utility Functions (Should match training)
#######################################
num_instances = 1000 # Or fewer for a quick test, e.g., 100
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

Pt_loaded = Parameter(torch.tensor(0.0, device=device)) # Initial value doesn't matter before loading
path_loss_exponent_loaded = Parameter(torch.tensor(0.0, device=device))
offset_loaded = Parameter(torch.tensor(0.0, device=device))

# Initialize scalers - their attributes will be loaded
feature_scaler_loaded = StandardScaler()
y_scaler_loaded = StandardScaler()

#######################################
# Load Model and Parameters
#######################################
load_path = 'trained_localization_model_64beacons_1000instances_fixed.pth'
print(f'Loading model and parameters from {load_path}...')
if torch.cuda.is_available():
    checkpoint = torch.load(load_path)
else:
    checkpoint = torch.load(load_path, map_location=torch.device('cpu'))

main_gnn_loaded.load_state_dict(checkpoint['main_gnn_state_dict'])

# For Parameter types, assign their .data attribute
Pt_loaded.data = checkpoint['Pt'].data.to(device)
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
print("\nLoaded Path Loss Model Parameters:")
print(f"  Pt (Transmitted Power related): {Pt_loaded.item():.4f}")
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
    filename = f"../matlab/data/64beacons_100instances/data_instance_{instance_idx}.mat"
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

# Use a consistent way to get the test set, e.g., by splitting all loaded data
# If the original training script used a fixed random seed for splitting, use that here too for exact match.
# Otherwise, this new split will be a random 20% of the reloaded 1000 instances.
total_instances_test = len(data_list_test)
train_size_dummy = int(0.8 * total_instances_test) 
test_size_final = total_instances_test - train_size_dummy
_, test_dataset_loaded = random_split(data_list_test, [train_size_dummy, test_size_final], generator=torch.Generator().manual_seed(42)) # Added generator for reproducibility

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

with torch.no_grad():
    for i, data in enumerate(tqdm(test_loader_loaded, desc="Evaluating with loaded model")):
        data = data.to(device)

        measured_RSSI = data.edge_attr.mean(dim=1)
        dist_estimated = estimate_distance_from_rssi(
            measured_RSSI,
            Pt_loaded.to(data.edge_index.device),
            path_loss_exponent_loaded.to(data.edge_index.device),
            offset_loaded.to(data.edge_index.device)
        )
        new_edge_attr = torch.cat([data.edge_attr.to(data.edge_index.device), dist_estimated.unsqueeze(1)], dim=1)

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
        if i == 0 and not plotted_sample: # Check plotted_sample to ensure it runs only once if loader is re-iterated
            plt.figure(figsize=(12,6))
            # Plot true unknown positions
            plt.scatter(true_positions[data.unknown_mask.cpu(),0],
                        true_positions[data.unknown_mask.cpu(),1],
                        c='blue', label='True Unknown Positions', alpha=0.6, s=50)
            # Plot predicted unknown positions
            plt.scatter(predicted_positions[data.unknown_mask.cpu(),0],
                        predicted_positions[data.unknown_mask.cpu(),1],
                        c='red', label='Predicted Unknown Positions', alpha=0.6, s=50)
            # Plot anchor nodes
            plt.scatter(true_positions[data.anchor_mask.cpu(),0],
                        true_positions[data.anchor_mask.cpu(),1],
                        c='green', marker='^', s=100, label='Anchor Nodes')

            # Draw lines between true and predicted positions for unknown nodes
            for node_idx in range(true_positions.shape[0]):
                if data.unknown_mask[node_idx]:
                    tp = true_positions[node_idx]
                    pp = predicted_positions[node_idx]
                    plt.plot([tp[0], pp[0]], [tp[1], pp[1]], 'k--', alpha=0.4, linewidth=0.8)
            
            plt.xlabel('X Position (m)')
            plt.ylabel('Y Position (m)')
            plt.title(f'Sample Visualization (Test Sample 0) - Loaded Model ({load_path.split("/")[-1]})')
            plt.legend()
            plt.grid(True)
            os.makedirs("results", exist_ok=True)
            sample_viz_filename = f'results/sample_visualization_loaded_model_{load_path.split("/")[-1].replace(".pth","")}.png'
            plt.savefig(sample_viz_filename)
            print(f'Sample visualization saved to {sample_viz_filename}')
            plt.show()
            plotted_sample = True # Mark as plotted

if errors_gcn_loaded:
    errors_gcn_loaded = np.array(errors_gcn_loaded)
    plt.figure(figsize=(12, 6))
    plt.hist(errors_gcn_loaded, bins=20, alpha=0.7, color='lightcoral', label=f'GCN Errors ({load_path.split("/")[-1]})')
    plt.xlabel('Localization Error (meters)')
    plt.ylabel('Number of Nodes')
    plt.title(f'Error Distribution (Loaded Model: {load_path.split("/")[-1]})')
    plt.legend()
    error_hist_filename = f'results/error_histogram_loaded_model_{load_path.split("/")[-1].replace(".pth","")}.png'
    plt.savefig(error_hist_filename)
    print(f'Error histogram saved to {error_hist_filename}')
    plt.show()

    mean_error_loaded = errors_gcn_loaded.mean()
    median_error_loaded = np.median(errors_gcn_loaded)
    print(f"Loaded GCN Mean Error: {mean_error_loaded:.4f} m, Median Error: {median_error_loaded:.4f} m")
else:
    print("No unknown nodes found in the test set to calculate errors or test_loader_loaded is empty.")

# %% [markdown]
# **Explanation of How to Use the .pth File (`36beacon_1000instances_fixed_pth_test.py`)**
# 
# This script demonstrates loading and utilizing the trained model and parameters saved in `trained_localization_model_36beacons_1000instances_fixed.pth`.
# 
# **Key Steps:**
# 
# 1.  **Environment and Definitions:**
#     *   Imports necessary libraries (torch, sklearn, numpy, etc.).
#     *   Sets the `device` (CPU or CUDA).
#     *   Re-defines the model architecture: `MainEdgeNet` and `MainGNN`. **Crucially, these definitions must be identical to the ones used during the training phase when the `.pth` file was created.**
#     *   Defines parameters like `num_measurements` which are needed for model input dimensions.
# 
# 2.  **Initialize Model and Scalers (Placeholders):**
#     *   Instances of `MainGNN`, `Pt`, `path_loss_exponent`, `offset`, `feature_scaler`, and `y_scaler` are created. Their initial values are not critical as they will be overwritten by the loaded checkpoint.
#     *   The dimensions for `MainGNN` (`node_in_dim_loaded`, `edge_in_dim_loaded`, etc.) must match the configuration of the saved model.
# 
# 3.  **Load Checkpoint:**
#     *   `torch.load(load_path)` is used to load the dictionary containing all saved states from the specified `.pth` file.
#     *   `map_location` is used to correctly load models trained on GPU onto a CPU if necessary.
# 
# 4.  **Restore States:**
#     *   `main_gnn_loaded.load_state_dict(checkpoint['main_gnn_state_dict'])` restores the GNN model weights.
#     *   For `torch.nn.Parameter` objects (`Pt_loaded`, `path_loss_exponent_loaded`, `offset_loaded`), their `.data` attribute is updated from the checkpoint. They are also moved to the correct `device`.
#     *   The `mean_` and `scale_` attributes of the `StandardScaler` instances (`feature_scaler_loaded`, `y_scaler_loaded`) are restored from the checkpoint. This is vital for consistent data transformation.
# 
# 5.  **Set to Evaluation Mode:**
#     *   `main_gnn_loaded.eval()` is called. This is important as it sets layers like Dropout and BatchNorm to evaluation mode.
# 
# 6.  **Data Preparation for Testing:**
#     *   The script re-loads the dataset (or a relevant subset for testing). This step needs to mirror the data loading and initial feature creation process of the training script.
#     *   **Crucially, the loaded `feature_scaler_loaded` and `y_scaler_loaded` are used to transform the raw features and target coordinates of the test data.** This ensures that the data fed to the loaded model is in the same scale and format as the data it was trained on.
#     *   A `test_loader_loaded` is created. To ensure a consistent test set for comparison with the original training run, a `generator` with a fixed `manual_seed` is used for `random_split`.
# 
# 7.  **Helper Function:**
#     *   The `estimate_distance_from_rssi` function is redefined as it's needed to compute the edge features for the loaded GNN.
# 
# 8.  **Evaluation with Loaded Model:**
#     *   The script iterates through the `test_loader_loaded`.
#     *   For each data batch:
#         *   `measured_RSSI` is calculated.
#         *   `dist_estimated` is computed using `estimate_distance_from_rssi` with the **loaded** physical parameters (`Pt_loaded`, `path_loss_exponent_loaded`, `offset_loaded`).
#         *   `new_edge_attr` is created by concatenating original edge attributes (ensured to be on the correct device) with the `dist_estimated`.
#         *   The `main_gnn_loaded` model makes predictions (`out`).
#         *   Predicted positions are inverse-transformed using the **loaded** `y_scaler_loaded`.
#         *   True positions are obtained from `data.orig_positions` (which stores raw, unscaled coordinates).
#         *   Errors are calculated for unknown nodes and collected.
#         *   **The first sample from the test set is visualized, including lines connecting true and predicted positions for unknown nodes. This plot is also saved.**
# 
# 9.  **Results:**
#     *   An error histogram and mean/median errors are printed for the loaded model's performance on the test set. **The error histogram is saved to a file.** An additional check is added to handle cases where `errors_gcn_loaded` might be empty (e.g., if the test set had no unknown nodes).
# 
# **To Run This Script:**
# 
# 1.  Ensure the `trained_localization_model_64beacons_1000instances_fixed.pth` (or the corresponding 36-beacon file name if you are testing that one by changing `load_path`) file is in the same directory as this script (or provide the correct path to it).
# 2.  Make sure the data path (`../matlab/data/...`) is correct relative to where you run this script and matches the beacon configuration (e.g., `64beacons_100instances` or `36beacons_100instances`).
# 3.  The required Python packages (PyTorch, PyTorch Geometric, scikit-learn, NumPy, Matplotlib, tqdm, geopy) must be installed.
# 
# This script provides a comprehensive way to test your saved model, ensuring that all components (model weights, physical parameters, data scalers) are correctly restored and applied.



