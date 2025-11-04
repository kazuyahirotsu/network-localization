import contextily as cx
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math

# --- Font Sizes from plot_eval_dump_power13.py ---
SLIDE_AXIS_LABEL_FONT_SIZE = 22
SLIDE_TICK_LABEL_FONT_SIZE = 18
SLIDE_TITLE_FONT_SIZE = 24

def fetch_satellite_image():
    """
    Fetches a satellite image for the experiment area and places it 
    side-by-side with the corresponding elevation map.
    The combined image is saved to 'paper/images/experiment_area_comparison.png'.
    """
    # --- Parameters from MATLAB script ---
    # Map origin (bottom-left corner)
    map_origin_lat = 40.466198
    map_origin_lon = 33.898610

    # Map size in meters
    map_size_meters = 4000

    # Earth's radius in meters
    earth_radius = 6378137

    # --- Calculate bounding box ---
    # Degrees per meter calculation
    deg_per_meter_lat = 1 / ((math.pi / 180) * earth_radius)
    deg_per_meter_lon = 1 / ((math.pi / 180) * earth_radius * math.cos(math.radians(map_origin_lat)))

    # Calculate the latitude and longitude deltas for the map size
    delta_lat = map_size_meters * deg_per_meter_lat
    delta_lon = map_size_meters * deg_per_meter_lon

    # Calculate the end coordinates (top-right corner)
    map_end_lat = map_origin_lat + delta_lat
    map_end_lon = map_origin_lon + delta_lon

    # Bounding box for contextily (west, south, east, north)
    bbox = (map_origin_lon, map_origin_lat, map_end_lon, map_end_lat)

    print(f"Fetching map for bounding box: {bbox}")

    # --- Create Figure with Two Subplots ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # --- Subplot 1: Satellite Image ---
    try:
        # Fetch the satellite image
        img, ext = cx.bounds2img(
            bbox[0], bbox[1], bbox[2], bbox[3],
            ll=True,  # Source is lon/lat
            source=cx.providers.Esri.WorldImagery,  # Satellite imagery provider
            zoom='auto'  # Auto-determine zoom level
        )

        ax1.imshow(img, extent=ext)

        # --- Customize and save plot ---
        ax1.set_xlabel("Longitude", fontsize=SLIDE_AXIS_LABEL_FONT_SIZE)
        ax1.set_ylabel("Latitude", fontsize=SLIDE_AXIS_LABEL_FONT_SIZE)
        ax1.set_title("(a) Satellite View", fontsize=SLIDE_TITLE_FONT_SIZE)

        # Manually set ticks and labels to show lon/lat values
        xticks = [ext[0], (ext[0] + ext[1]) / 2, ext[1]]
        xlabels = [f'{bbox[0]:.3f}', f'{(bbox[0] + bbox[2]) / 2:.3f}', f'{bbox[2]:.3f}']
        ax1.set_xticks(xticks)
        ax1.set_xticklabels(xlabels, rotation=30, ha='right', fontsize=SLIDE_TICK_LABEL_FONT_SIZE)

        yticks = [ext[2], (ext[2] + ext[3]) / 2, ext[3]]
        ylabels = [f'{bbox[1]:.3f}', f'{(bbox[1] + bbox[3]) / 2:.3f}', f'{bbox[3]:.3f}']
        ax1.set_yticks(yticks)
        ax1.set_yticklabels(ylabels, fontsize=SLIDE_TICK_LABEL_FONT_SIZE)

    except Exception as e:
        print(f"An error occurred while fetching the satellite map: {e}")
        ax1.text(0.5, 0.5, 'Satellite image fetch failed.', ha='center', va='center')

    # --- Subplot 2: Elevation Map ---
    try:
        # Fetch the terrain map for the same bounding box
        terrain_img, terrain_ext = cx.bounds2img(
            bbox[0], bbox[1], bbox[2], bbox[3],
            ll=True,
            source=cx.providers.Esri.WorldTopoMap,
            zoom='auto'
        )
        ax2.imshow(terrain_img, extent=terrain_ext)
        ax2.set_title("(b) Terrain Map", fontsize=SLIDE_TITLE_FONT_SIZE)
        ax2.set_xlabel("Longitude", fontsize=SLIDE_AXIS_LABEL_FONT_SIZE)
        ax2.set_ylabel("Latitude", fontsize=SLIDE_AXIS_LABEL_FONT_SIZE)

        # Use the same ticks and labels as the satellite map for alignment
        ax2.set_xticks(xticks)
        ax2.set_xticklabels(xlabels, rotation=30, ha='right', fontsize=SLIDE_TICK_LABEL_FONT_SIZE)
        ax2.set_yticks(yticks)
        ax2.set_yticklabels(ylabels, fontsize=SLIDE_TICK_LABEL_FONT_SIZE)

    except Exception as e:
        print(f"An error occurred while fetching the terrain map: {e}")
        ax2.text(0.5, 0.5, 'Terrain map fetch failed.', ha='center', va='center')

    # --- Finalize and Save Plot ---
    plt.tight_layout(pad=3.0)
    output_path = "paper/images/experiment_area_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

    print(f"Combined image saved to: {output_path}")

if __name__ == "__main__":
    fetch_satellite_image()
