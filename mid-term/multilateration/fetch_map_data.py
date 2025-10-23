import requests
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from tqdm import tqdm

# --- Configuration ---
# This script requires the 'requests' and 'tqdm' libraries
# pip install requests tqdm
OUTPUT_DIR = "investigation_results_multi_instance_dense"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Map origin from MATLAB code
MAP_ORIGIN_LAT = 40.466198
MAP_ORIGIN_LON = 33.898610
MAP_SIZE_METERS = 4000.0

# Earth constants for coordinate conversion
EARTH_RADIUS = 6378137.0
METERS_PER_DEGREE_LAT = (math.pi / 180) * EARTH_RADIUS
METERS_PER_DEGREE_LON = (math.pi / 180) * EARTH_RADIUS * np.cos(np.deg2rad(MAP_ORIGIN_LAT))
DEG_PER_METER_LAT = 1 / METERS_PER_DEGREE_LAT
DEG_PER_METER_LON = 1 / METERS_PER_DEGREE_LON

# Grid density for fetching elevation data
GRID_DENSITY = 250 # 250x250 grid for high density
BATCH_SIZE = 10000 # Number of points to fetch in each API call

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid')
FONT_SIZE_TITLE = 20
FONT_SIZE_LABEL = 16
FONT_SIZE_TICKS = 14
# --- End Configuration ---

def get_elevation_data(locations):
    """
    Fetches elevation data from the Open-Elevation API in batches.
    """
    all_elevations = []
    
    # Use tqdm for a progress bar
    for i in tqdm(range(0, len(locations), BATCH_SIZE), desc="Fetching elevation batches"):
        batch = locations[i:i+BATCH_SIZE]
        try:
            response = requests.post(
                'https://api.open-elevation.com/api/v1/lookup',
                json={'locations': batch},
                headers={'Content-type': 'application/json'}
            )
            response.raise_for_status()
            data = response.json()
            all_elevations.extend([item['elevation'] for item in data['results']])
        except requests.exceptions.RequestException as e:
            print(f"\nError fetching batch: {e}")
            return None
            
    return all_elevations

def main():
    print("--- Fetching Real-World Elevation Data (High Density) ---")

    # 1. Create a grid of lat/lon coordinates
    x_coords_meters = np.linspace(0, MAP_SIZE_METERS, GRID_DENSITY)
    y_coords_meters = np.linspace(0, MAP_SIZE_METERS, GRID_DENSITY)

    locations_to_fetch = []
    for y in y_coords_meters:
        for x in x_coords_meters:
            lat = MAP_ORIGIN_LAT + (y * DEG_PER_METER_LAT)
            lon = MAP_ORIGIN_LON + (x * DEG_PER_METER_LON)
            locations_to_fetch.append({'latitude': lat, 'longitude': lon})

    # 2. Fetch elevation data from the API
    print(f"Fetching {len(locations_to_fetch)} elevation points in batches of {BATCH_SIZE}...")
    elevations = get_elevation_data(locations_to_fetch)
    
    if elevations is None:
        print("Could not retrieve elevation data. Aborting.")
        return

    # 3. Reshape the flat list of elevations into a 2D grid for plotting
    elevation_grid = np.array(elevations).reshape((GRID_DENSITY, GRID_DENSITY))

    # 4. Visualize and save the elevation map
    print("Generating elevation heatmap...")
    plt.figure(figsize=(12, 10))
    
    # Use 'plasma' or 'viridis' colormap as they are perceptually uniform
    im = plt.imshow(
        elevation_grid,
        cmap='plasma',
        origin='lower',
        extent=[0, MAP_SIZE_METERS, 0, MAP_SIZE_METERS],
        interpolation='bilinear' # Add interpolation for a smoother look
    )
    
    cbar = plt.colorbar(im)
    cbar.set_label('Elevation (m)', fontsize=FONT_SIZE_LABEL)
    cbar.ax.tick_params(labelsize=FONT_SIZE_TICKS)
    
    plt.title('Real-World Terrain Elevation Map', fontsize=FONT_SIZE_TITLE)
    plt.xlabel('X Position (m)', fontsize=FONT_SIZE_LABEL)
    plt.ylabel('Y Position (m)', fontsize=FONT_SIZE_LABEL)
    plt.xticks(fontsize=FONT_SIZE_TICKS)
    plt.yticks(fontsize=FONT_SIZE_TICKS)
    plt.grid(False)

    output_path = os.path.join(OUTPUT_DIR, "elevation_map.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nHigh-density elevation map saved to {output_path}")
    print("You can now compare 'elevation_map.png' with 'agg_task_3_1_spatial_error_heatmap.png' side-by-side.")
    print("\n--- Script finished. ---")

if __name__ == '__main__':
    main() 