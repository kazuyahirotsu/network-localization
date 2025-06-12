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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

#######################################
# Parameters and Utility Functions
#######################################
num_instances = 1000
num_anchors = 9
num_unknowns = 27
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
# Load Data
#######################################
data_list = []
for instance_idx in tqdm(range(1, num_instances + 1), desc="Loading MATLAB data"):
    filename = f"../matlab/data/36beacons_100instances/data_instance_{instance_idx}.mat"
    mat_data = loadmat(filename)

    nodeLatitudes = mat_data['nodeLatitudes'].flatten()
    nodeLongitudes = mat_data['nodeLongitudes'].flatten()
    signal_strength_matrix = mat_data['signal_strength_matrix']

    num_nodes = len(nodeLatitudes)
    node_x = np.zeros(num_nodes)
    node_y = np.zeros(num_nodes)
    for i in range(num_nodes):
        node_x[i], node_y[i] = latlon_to_xy(nodeLatitudes[i], nodeLongitudes[i], mapOriginLat, mapOriginLon)

    # Construct edges
    edge_index_list = []
    edge_attr_list = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and not np.isnan(signal_strength_matrix[i, j, 0]):
                edge_index_list.append([i, j])
                # We will handle RSSI and delta_RSSI later. For now just store.
                edge_attr_list.append(signal_strength_matrix[i, j, :])

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    # Identify anchors and unknowns
    anchor_mask = torch.zeros(num_nodes, dtype=torch.bool)
    anchor_mask[:num_anchors] = True
    unknown_mask = ~anchor_mask

    # Initialize unknowns near average anchor position
    avg_anchor_x = node_x[anchor_mask].mean()
    avg_anchor_y = node_y[anchor_mask].mean()

    node_features = []
    for i in range(num_nodes):
        is_anchor = 1 if i < num_anchors else 0
        if is_anchor:
            node_features.append([node_x[i], node_y[i], is_anchor])
        else:
            node_features.append([avg_anchor_x + np.random.randn()*10,
                                  avg_anchor_y + np.random.randn()*10,
                                  is_anchor])

    x = torch.tensor(node_features, dtype=torch.float)
    y = torch.tensor(np.column_stack((node_x, node_y)), dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    data.anchor_mask = anchor_mask
    data.unknown_mask = unknown_mask

    # Store original positions (not scaled) for distance calculation
    data.orig_positions = torch.tensor(np.column_stack((node_x, node_y)), dtype=torch.float)

    data_list.append(data)

#######################################
# Scale Features and Targets
#######################################
all_features = torch.cat([d.x for d in data_list], dim=0).numpy()
feature_scaler = StandardScaler()
feature_scaler.fit(all_features)
for d in data_list:
    d.x = torch.tensor(feature_scaler.transform(d.x.numpy()), dtype=torch.float)

all_y = torch.cat([d.y for d in data_list], dim=0).numpy()
y_scaler = StandardScaler()
y_scaler.fit(all_y)
for d in data_list:
    d.y = torch.tensor(y_scaler.transform(d.y.numpy()), dtype=torch.float)

# Assume data_list contains 1000 instances
total_instances = len(data_list)
train_size = int(0.8 * total_instances)
test_size = total_instances - train_size

# Randomly split the data_list into training and testing sets
train_dataset, test_dataset = random_split(data_list, [train_size, test_size])

# Create DataLoaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


# train_loader = DataLoader(data_list, batch_size=1, shuffle=True)
# test_loader = DataLoader(data_list, batch_size=1, shuffle=False)

#######################################
# Physically Inspired Parameters
#######################################
# Learnable parameters for the RSSI model
Pt = Parameter(torch.tensor(0.0, requires_grad=True, device=device))
path_loss_exponent = Parameter(torch.tensor(3.0, requires_grad=True, device=device))
offset = Parameter(torch.tensor(-50.0, requires_grad=True, device=device))

#######################################
# Model Definition
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

node_in_dim = data_list[0].x.shape[1]
# We will add delta_RSSI as one additional feature to the existing edge attributes.
# Currently, edge_attr: shape = (num_edges, num_measurements)
# After adding delta_RSSI: shape = (num_edges, num_measurements + 1)
edge_in_dim = num_measurements + 1
hidden_dim = 64
output_dim = 2

main_gnn = MainGNN(node_in_dim=node_in_dim,
                   edge_in_dim=edge_in_dim,
                   hidden_dim=hidden_dim,
                   output_dim=output_dim).to(device)

params = list(main_gnn.parameters()) + [Pt, path_loss_exponent, offset]
optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-5)  # Lower LR for stability

#######################################
# Training
#######################################
num_epochs = 50
loss_history = []

def compute_distances(orig_positions, edge_index):
    # orig_positions: [num_nodes, 2]
    # edge_index: [2, num_edges]
    # return distances [num_edges, 1]
    src = edge_index[0]
    dst = edge_index[1]
    pos_src = orig_positions[src]
    pos_dst = orig_positions[dst]
    dist = torch.sqrt(torch.sum((pos_src - pos_dst)**2, dim=1))
    return dist

for epoch in range(num_epochs):
    main_gnn.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()

        # Compute distances in original space (not scaled)
        dist = compute_distances(data.orig_positions, data.edge_index)

        # Compute expected RSSI
        # RSSI_expected = Pt - 10 * path_loss_exponent * log10(dist) + offset
        # Avoid log10(0), add a small epsilon
        epsilon = 1e-6
        RSSI_expected = Pt - 10.0 * path_loss_exponent * torch.log10(dist + epsilon) + offset

        # measured_RSSI: take mean over the 10 measurements for simplicity
        measured_RSSI = data.edge_attr.mean(dim=1)
        delta_RSSI = (measured_RSSI - RSSI_expected).unsqueeze(1)

        # New edge_attr: original + delta_RSSI
        new_edge_attr = torch.cat([data.edge_attr, delta_RSSI], dim=1)

        out = main_gnn(data.x, data.edge_index, new_edge_attr)

        # Use Smooth L1 Loss for robustness
        loss = F.smooth_l1_loss(out[data.unknown_mask], data.y[data.unknown_mask])
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(main_gnn.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_([Pt, path_loss_exponent, offset], 1.0)

        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    loss_history.append(avg_loss)
    print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

plt.figure()
plt.plot(loss_history)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

#######################################
# Evaluation
#######################################
main_gnn.eval()
errors_gcn = []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)

        dist = compute_distances(data.orig_positions, data.edge_index)
        epsilon = 1e-6
        RSSI_expected = Pt - 10.0 * path_loss_exponent * torch.log10(dist + epsilon) + offset
        measured_RSSI = data.edge_attr.mean(dim=1)
        delta_RSSI = (measured_RSSI - RSSI_expected).unsqueeze(1)
        new_edge_attr = torch.cat([data.edge_attr, delta_RSSI], dim=1)

        out = main_gnn(data.x, data.edge_index, new_edge_attr)

        predicted_scaled = out.cpu().numpy()
        predicted_positions = y_scaler.inverse_transform(predicted_scaled)
        true_positions = y_scaler.inverse_transform(data.y.cpu().numpy())

        predicted_positions[data.anchor_mask.cpu()] = true_positions[data.anchor_mask.cpu()]

        for idx in range(true_positions.shape[0]):
            if data.unknown_mask[idx]:
                true_pos = true_positions[idx]
                pred_pos = predicted_positions[idx]
                error = np.sqrt((true_pos[0] - pred_pos[0])**2 + (true_pos[1] - pred_pos[1])**2)
                errors_gcn.append(error)

errors_gcn = np.array(errors_gcn)
plt.figure(figsize=(12, 6))
plt.hist(errors_gcn, bins=20, alpha=0.7, color='skyblue', label='GCN Errors')
plt.xlabel('Localization Error (meters)')
plt.ylabel('Number of Nodes')
plt.title('Error Distribution for GCN on Test Data')
plt.legend()
plt.show()

mean_error = errors_gcn.mean()
median_error = np.median(errors_gcn)
print(f"GCN Mean Error: {mean_error:.4f} m, Median Error: {median_error:.4f} m")

# Visualization of one sample
sample_data = data_list[0].to(device)
with torch.no_grad():
    dist = compute_distances(sample_data.orig_positions, sample_data.edge_index)
    epsilon = 1e-6
    RSSI_expected = Pt - 10.0 * path_loss_exponent * torch.log10(dist + epsilon) + offset
    measured_RSSI = sample_data.edge_attr.mean(dim=1)
    delta_RSSI = (measured_RSSI - RSSI_expected).unsqueeze(1)
    new_edge_attr = torch.cat([sample_data.edge_attr, delta_RSSI], dim=1)

    out = main_gnn(sample_data.x, sample_data.edge_index, new_edge_attr)
    predicted_scaled = out.cpu().numpy()
    predicted_positions = y_scaler.inverse_transform(predicted_scaled)
    true_positions = y_scaler.inverse_transform(sample_data.y.cpu().numpy())

predicted_positions[sample_data.anchor_mask.cpu()] = true_positions[sample_data.anchor_mask.cpu()]

plt.figure(figsize=(12,6))
plt.scatter(true_positions[sample_data.unknown_mask.cpu(),0],
            true_positions[sample_data.unknown_mask.cpu(),1],
            c='blue', label='True Unknown Positions', alpha=0.6)
plt.scatter(predicted_positions[sample_data.unknown_mask.cpu(),0],
            predicted_positions[sample_data.unknown_mask.cpu(),1],
            c='red', label='Predicted Unknown Positions', alpha=0.6)
plt.scatter(true_positions[sample_data.anchor_mask.cpu(),0],
            true_positions[sample_data.anchor_mask.cpu(),1],
            c='green', marker='^', s=100, label='Anchor Nodes')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.title('Comparison of True vs. Predicted Node Positions (Sample Graph)')
plt.legend()
plt.grid(True)
plt.show()

# %% 
# Added section for visualizing delta_RSSI to infer learned terrain/propagation info

print("\n#######################################")
print("# Visualizing Delta_RSSI              #")
print("#######################################")

# Ensure model is in eval mode and parameters are on the correct device
main_gnn.eval()
Pt_learned = Pt.to(device)
path_loss_exponent_learned = path_loss_exponent.to(device)
offset_learned = offset.to(device)

num_instances_to_visualize = 3 # Visualize for the first few instances
visualized_instances_count = 0

# Store all delta_RSSI values for global statistics if needed later
all_delta_rssi_values_collected = []

with torch.no_grad():
    # Iterate through the full data_list to use original (unshuffled) instances for consistency
    for i, data_instance in enumerate(tqdm(data_list, desc="Calculating Delta_RSSI for Visualization")):
        if visualized_instances_count >= num_instances_to_visualize:
            break

        data_instance = data_instance.to(device)
        
        true_positions_xy = data_instance.orig_positions # These are already (N, 2) tensors
        edge_index_viz = data_instance.edge_index
        edge_attr_viz = data_instance.edge_attr # (num_edges, num_measurements)

        dist_viz = compute_distances(true_positions_xy, edge_index_viz) # Pass true positions
        
        epsilon = 1e-9 # Ensure consistency with training epsilon, or slightly smaller if dist can be very small
        RSSI_expected_simple_model = Pt_learned - 10.0 * path_loss_exponent_learned * torch.log10(dist_viz + epsilon) + offset_learned
        
        measured_RSSI_viz = edge_attr_viz.mean(dim=1)
        delta_RSSI_viz = measured_RSSI_viz - RSSI_expected_simple_model
        
        all_delta_rssi_values_collected.extend(delta_RSSI_viz.cpu().numpy())

        # --- Plotting for this instance ---
        plt.figure(figsize=(14, 10))
        true_positions_np = true_positions_xy.cpu().numpy()
        anchor_mask_np = data_instance.anchor_mask.cpu().numpy()
        unknown_mask_np = data_instance.unknown_mask.cpu().numpy()

        # Plot nodes
        plt.scatter(true_positions_np[anchor_mask_np, 0], true_positions_np[anchor_mask_np, 1], 
                    c='green', marker='^', s=150, label='Anchor Nodes', edgecolors='black', zorder=5)
        plt.scatter(true_positions_np[unknown_mask_np, 0], true_positions_np[unknown_mask_np, 1], 
                    c='blue', marker='o', s=70, label='Unknown Nodes (True Pos)', alpha=0.7, zorder=4)

        # Plot links colored by delta_RSSI
        delta_rssi_np = delta_RSSI_viz.cpu().numpy()
        
        # Determine a consistent color scale if desired, or normalize per plot
        # For now, let vmin and vmax be determined by the current instance's data for more contrast
        vmin_delta = np.percentile(delta_rssi_np, 5) if len(delta_rssi_np)>0 else -10
        vmax_delta = np.percentile(delta_rssi_np, 95) if len(delta_rssi_np)>0 else 10
        # Ensure vmin is less than vmax, and they are not equal to avoid issues with Normalize
        if vmin_delta >= vmax_delta:
            vmin_delta = vmax_delta -1 # a small difference
            if vmin_delta == vmax_delta: # if still equal (e.g. vmax_delta was 0)
                 vmin_delta = -1
                 vmax_delta = 1
            

        norm = plt.Normalize(vmin=vmin_delta, vmax=vmax_delta)
        cmap = plt.cm.get_cmap('coolwarm') # Blue (negative delta) to Red (positive delta)

        for k in range(edge_index_viz.shape[1]):
            node_i_idx = edge_index_viz[0, k].item()
            node_j_idx = edge_index_viz[1, k].item()
            
            start_pos = true_positions_np[node_i_idx]
            end_pos = true_positions_np[node_j_idx]
            
            delta_val = delta_rssi_np[k]
            line_color = cmap(norm(delta_val))
            
            plt.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]], color=line_color, linewidth=1.5, alpha=0.6)

        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title(f'Instance {i+1}: Links Colored by Delta_RSSI (Measured - SimpleModel)')
        
        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([]) # You need this for the colorbar to work with line plots
        cbar = plt.colorbar(sm, ax=plt.gca())
        cbar.set_label('Delta_RSSI (dB)')
        
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        # plt.savefig(f"output_delta_rssi_instance_{i+1}.png") # Optionally save the figure
        plt.show()
        
        visualized_instances_count += 1

# --- Optional: Global statistics or histogram of all_delta_rssi_values ---
if all_delta_rssi_values_collected:
    plt.figure(figsize=(10,6))
    plt.hist(all_delta_rssi_values_collected, bins=100, alpha=0.7, color='purple')
    plt.xlabel('Delta_RSSI (dB)')
    plt.ylabel('Frequency of Links')
    plt.title('Global Distribution of Delta_RSSI Values (All Visualized Instances)')
    plt.grid(True)
    plt.show()
    print(f"Overall Delta_RSSI Stats: Min={np.min(all_delta_rssi_values_collected):.2f}, Max={np.max(all_delta_rssi_values_collected):.2f}, Mean={np.mean(all_delta_rssi_values_collected):.2f}, Std={np.std(all_delta_rssi_values_collected):.2f}")


# %%



