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

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print('Using device:', device)

#######################################
# Parameters and Utility Functions
#######################################
num_instances = 1000
num_anchors = 16
num_unknowns = 48
num_measurements = 10

# Minimum RSSI (dBm) to keep a link
RSSI_MIN_DBM = -130.0

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
    filename = f"../../matlab/data/64beacons_100instances/data_instance_{instance_idx}.mat"
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
            rssi_values = signal_strength_matrix[i, j, :]
            rssi_mean = np.nanmean(rssi_values)
            if i != j \
               and not np.isnan(signal_strength_matrix[i, j, 0]) \
               and not np.isnan(rssi_mean) \
               and rssi_mean >= RSSI_MIN_DBM:
                edge_index_list.append([i, j])
                # We will handle RSSI and delta_RSSI later. For now just store.
                edge_attr_list.append(rssi_values)

    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)

    # Identify anchors and unknowns
    anchor_mask = torch.zeros(num_nodes, dtype=torch.bool)
    anchor_mask[:num_anchors] = True
    unknown_mask = ~anchor_mask

    # Initialize unknowns near average anchor position
    avg_anchor_x = node_x[:num_anchors].mean()
    avg_anchor_y = node_y[:num_anchors].mean()

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
# Fixed transmit power at 13 dBm
PT_DBM = 13.0
PT_DBM_TENSOR = torch.tensor(PT_DBM, device=device)

# Learnable parameters for the RSSI model (excluding power)
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
# We will add estimated_distance as one additional feature to the existing edge attributes.
# Currently, edge_attr: shape = (num_edges, num_measurements)
# After adding estimated_distance: shape = (num_edges, num_measurements + 1)
edge_in_dim = num_measurements + 1
hidden_dim = 64
output_dim = 2

main_gnn = MainGNN(node_in_dim=node_in_dim,
                   edge_in_dim=edge_in_dim,
                   hidden_dim=hidden_dim,
                   output_dim=output_dim).to(device)

params = list(main_gnn.parameters()) + [path_loss_exponent, offset]
optimizer = torch.optim.Adam(params, lr=1e-4, weight_decay=1e-5)  # Lower LR for stability

#######################################
# Training
#######################################
num_epochs = 50
loss_history = []

def estimate_distance_from_rssi(measured_rssi, Pt_param, ple_param, offset_param, epsilon=1e-6):
    """
    Estimates distance from measured RSSI using the path loss model.
    RSSI = Pt - 10 * PLE * log10(d) + offset
    log10(d) = (Pt + offset - RSSI) / (10 * PLE)
    d = 10**((Pt + offset - RSSI) / (10 * PLE))
    """
    # Ensure PLE is positive and avoid division by zero
    ple_val = ple_param + epsilon if torch.abs(ple_param) < epsilon else ple_param
    # Ensure the divisor (10 * ple_val) is not zero
    divisor = 10 * ple_val
    if torch.abs(divisor) < epsilon: # if divisor is too close to zero
        divisor = epsilon if divisor >= 0 else -epsilon

    exponent_val = (Pt_param + offset_param - measured_rssi) / divisor
    # Clamp exponent to avoid potential overflow/underflow with 10**exponent_val
    exponent_val = torch.clamp(exponent_val, -2.0, 5.0) # Clamping log10(distance) roughly between 0.01m and 100km
    
    dist = torch.pow(10.0, exponent_val)
    # Clamp distance to a minimum positive value
    dist = torch.clamp(dist, min=epsilon)
    return dist

# This function calculates true distances and is NOT used for GNN feature input
def compute_true_distances(orig_positions, edge_index):
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

        # measured_RSSI: take mean over the 10 measurements
        measured_RSSI = data.edge_attr.mean(dim=1) # Shape: [num_edges]

        # Estimate distance from RSSI model using fixed power and learnable parameters
        dist_estimated = estimate_distance_from_rssi(
            measured_RSSI,
            PT_DBM_TENSOR,
            path_loss_exponent.to(data.edge_index.device),
            offset.to(data.edge_index.device)
        ) # Shape: [num_edges]

        # New edge_attr: original features + estimated_distance (needs to be [num_edges, 1])
        new_edge_attr = torch.cat([data.edge_attr, dist_estimated.unsqueeze(1)], dim=1)

        out = main_gnn(data.x, data.edge_index, new_edge_attr)

        # Use Smooth L1 Loss for robustness
        loss = F.smooth_l1_loss(out[data.unknown_mask], data.y[data.unknown_mask])
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(main_gnn.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_([path_loss_exponent, offset], 1.0)

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

        # measured_RSSI: take mean over the 10 measurements
        measured_RSSI = data.edge_attr.mean(dim=1) # Shape: [num_edges]

        # Estimate distance from RSSI model using fixed power and learned parameters
        dist_estimated = estimate_distance_from_rssi(
            measured_RSSI,
            PT_DBM_TENSOR,
            path_loss_exponent.to(data.edge_index.device),
            offset.to(data.edge_index.device)
        ) # Shape: [num_edges]
        
        # New edge_attr: original features + estimated_distance
        new_edge_attr = torch.cat([data.edge_attr, dist_estimated.unsqueeze(1)], dim=1)

        out = main_gnn(data.x, data.edge_index, new_edge_attr)

        predicted_scaled = out.cpu().numpy()
        predicted_positions = y_scaler.inverse_transform(predicted_scaled)
        true_positions = y_scaler.inverse_transform(data.y.cpu().numpy())

        anchor_mask_np = data.anchor_mask.detach().cpu().numpy().astype(bool)
        predicted_positions[anchor_mask_np] = true_positions[anchor_mask_np]

        unknown_mask_np = data.unknown_mask.detach().cpu().numpy().astype(bool)
        diffs = predicted_positions[unknown_mask_np] - true_positions[unknown_mask_np]
        node_errors = np.sqrt(np.sum(diffs**2, axis=1))
        errors_gcn.extend(node_errors.tolist())

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
    # measured_RSSI: take mean over the 10 measurements
    measured_RSSI = sample_data.edge_attr.mean(dim=1) # Shape: [num_edges]

    # Estimate distance from RSSI model using fixed power and learned parameters
    dist_estimated = estimate_distance_from_rssi(
        measured_RSSI,
        PT_DBM_TENSOR,
        path_loss_exponent.to(sample_data.edge_index.device),
        offset.to(sample_data.edge_index.device)
    ) # Shape: [num_edges]

    # New edge_attr: original features + estimated_distance
    new_edge_attr = torch.cat([sample_data.edge_attr, dist_estimated.unsqueeze(1)], dim=1)

    out = main_gnn(sample_data.x, sample_data.edge_index, new_edge_attr)
    predicted_scaled = out.cpu().numpy()
    predicted_positions = y_scaler.inverse_transform(predicted_scaled)
    true_positions = y_scaler.inverse_transform(sample_data.y.cpu().numpy())

anchor_mask_np = sample_data.anchor_mask.detach().cpu().numpy().astype(bool)
predicted_positions[anchor_mask_np] = true_positions[anchor_mask_np]

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

#######################################
# Save Model and Parameters
#######################################
save_path = 'trained_localization_model_64beacons_1000instances_fixed_power13_2.pth'
torch.save({
    'main_gnn_state_dict': main_gnn.state_dict(),
    'Pt_dBm': PT_DBM,
    'path_loss_exponent': float(path_loss_exponent.detach().cpu().item()),
    'offset': float(offset.detach().cpu().item()),
    'optimizer_state_dict': optimizer.state_dict(), # Optional: if you want to resume training
    'feature_scaler_params': {
        'mean_': feature_scaler.mean_,
        'scale_': feature_scaler.scale_
    },
    'y_scaler_params': {
        'mean_': y_scaler.mean_,
        'scale_': y_scaler.scale_
    }
    # You might also want to save epoch, loss_history if resuming training
    # 'epoch': epoch,
    # 'loss_history': loss_history
}, save_path)
print(f'Model and parameters saved to {save_path}')





