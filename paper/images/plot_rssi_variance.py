import os
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import math

# --- Font Sizes ---
SLIDE_AXIS_LABEL_FONT_SIZE = 22
SLIDE_TICK_LABEL_FONT_SIZE = 18
SLIDE_TITLE_FONT_SIZE = 24
SLIDE_LEGEND_FONT_SIZE = 18

# --- Constants ---
TERRAIN_DATA_DIR = 'matlab/data/64beacons_100instances'
FREE_SPACE_DATA_DIR = 'matlab/data/64beacons_1000instances_free'
NUM_SAMPLES_TO_LOAD = 10  # Load 10 random instances to keep it fast
P_T = 13  # Transmit power in dBm

# --- Geographic Constants from MATLAB script ---
MAP_ORIGIN_LAT = 40.466198
EARTH_RADIUS = 6378137
METERS_PER_DEG_LAT = (math.pi / 180) * EARTH_RADIUS
METERS_PER_DEG_LON = (math.pi / 180) * EARTH_RADIUS * math.cos(math.radians(MAP_ORIGIN_LAT))

def latlon_to_xy(lats, lons):
    """Converts arrays of latitude/longitude to local XY coordinates in meters."""
    y = (lats - MAP_ORIGIN_LAT) * METERS_PER_DEG_LAT
    x = (lons - (lons.mean() - MAP_ORIGIN_LAT * METERS_PER_DEG_LON / METERS_PER_DEG_LAT)) * METERS_PER_DEG_LON
    return np.stack([x, y], axis=-1)

def get_distances_and_rssi(data_dir: str):
    """Loads a sample of .mat files and extracts all pairwise distances and RSSI values."""
    all_distances = []
    all_rssi = []
    
    file_list = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
    sample_files = random.sample(file_list, min(len(file_list), NUM_SAMPLES_TO_LOAD))

    for filename in sample_files:
        filepath = os.path.join(data_dir, filename)
        try:
            data = loadmat(filepath)
            lats = data['nodeLatitudes'].flatten()
            lons = data['nodeLongitudes'].flatten()
            rssi_matrix = data['signal_strength_matrix']  # Shape: (nodes, nodes, samples)

            coords = latlon_to_xy(lats, lons)
            dist_matrix = np.sqrt(np.sum((coords[:, np.newaxis, :] - coords[np.newaxis, :, :])**2, axis=-1))
            
            # Flatten matrices to get pairs of (distance, rssi)
            num_nodes = len(lats)
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j:
                        d = dist_matrix[i, j]
                        rssi_samples = rssi_matrix[i, j, :]
                        # Filter out invalid RSSI values if any
                        valid_rssi = rssi_samples[~np.isnan(rssi_samples)]
                        all_distances.extend([d] * len(valid_rssi))
                        all_rssi.extend(valid_rssi)
        except Exception as e:
            print(f"Could not process {filepath}: {e}")

    return np.array(all_distances), np.array(all_rssi)

def plot_rssi_comparison():
    """Generates and saves the side-by-side RSSI variance plot."""
    print("Loading terrain-aware data...")
    terrain_dist, terrain_rssi = get_distances_and_rssi(TERRAIN_DATA_DIR)
    print("Loading free-space data...")
    free_dist, free_rssi = get_distances_and_rssi(FREE_SPACE_DATA_DIR)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

    # --- Plot 1: Terrain-Aware ---
    ax1.scatter(terrain_dist, terrain_rssi, alpha=0.05, s=10)
    ax1.set_title('(a) Terrain-Aware', fontsize=SLIDE_TITLE_FONT_SIZE)
    ax1.set_xlabel('Distance (m)', fontsize=SLIDE_AXIS_LABEL_FONT_SIZE)
    ax1.set_ylabel('RSSI (dBm)', fontsize=SLIDE_AXIS_LABEL_FONT_SIZE)

    # --- Plot 2: Free-Space ---
    ax2.scatter(free_dist, free_rssi, alpha=0.05, s=10)
    ax2.set_title('(b) Free-Space', fontsize=SLIDE_TITLE_FONT_SIZE)
    ax2.set_xlabel('Distance (m)', fontsize=SLIDE_AXIS_LABEL_FONT_SIZE)

    # --- Formatting for Both ---
    # Determine global y-axis limits across both datasets for a fair comparison
    global_ymin = min(np.min(terrain_rssi) if len(terrain_rssi) > 0 else 0, 
                      np.min(free_rssi) if len(free_rssi) > 0 else 0)
    global_ymax = max(np.max(terrain_rssi) if len(terrain_rssi) > 0 else -100, 
                      np.max(free_rssi) if len(free_rssi) > 0 else -100)
    y_padding = (global_ymax - global_ymin) * 0.05
    
    max_dist = max(np.max(terrain_dist) if len(terrain_dist) > 0 else 5000, 
                   np.max(free_dist) if len(free_dist) > 0 else 5000)

    for ax in [ax1, ax2]:
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=SLIDE_TICK_LABEL_FONT_SIZE)
        ax.set_xlim(0, max_dist)
        ax.set_ylim(global_ymin - y_padding, global_ymax + y_padding)

    plt.tight_layout(pad=3.0)
    output_path = "paper/images/rssi_variance_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"RSSI comparison plot saved to: {output_path}")

if __name__ == "__main__":
    plot_rssi_comparison()
