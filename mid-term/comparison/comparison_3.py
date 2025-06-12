import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
GCN_RSSI_DIST_ERRORS_PATH = "../gcn/results/gcn_errors_trained_localization_model_64beacons_1000instances_fixed.npy"
GCN_RAW_RSSI_ERRORS_PATH = "../gcn/results/gcn_errors_trained_localization_model_64beacons_1000instances_fixed_no_rssi2dist_no_rssi2dist.npy"
MULTILATERATION_ERRORS_PATH = "../multilateration/output_visualizations_matlab_im/64beacons_100instances_64N_16A_im_simple_errors_raw.npy"
OUTPUT_DIR = "comparison_results_three_methods"
OUTPUT_FILENAME = "error_comparison_histogram_three_methods.png"

# Font sizes for paper quality - INCREASED FOR SLIDES
TITLE_FONT_SIZE = 28        # Was 25
AXIS_LABEL_FONT_SIZE = 26   # Was 25
TICK_LABEL_FONT_SIZE = 22     # Was 25
LEGEND_FONT_SIZE = 22       # Was 25
SUPTITLE_FONT_SIZE = 30     # Was 25
BOXPLOT_TEXT_FONT_SIZE = 18 # For mean/median text on boxplot, was LEGEND_FONT_SIZE - 7 => 18

# --- Create output directory ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load data ---
try:
    gcn_rssi_dist_errors = np.load(GCN_RSSI_DIST_ERRORS_PATH)
    print(f"Loaded GCN (RSSI-Dist Est.) errors from: {GCN_RSSI_DIST_ERRORS_PATH}")
    print(f"GCN (RSSI-Dist Est.) errors shape: {gcn_rssi_dist_errors.shape}, Mean: {np.mean(gcn_rssi_dist_errors):.2f} m, Median: {np.median(gcn_rssi_dist_errors):.2f} m")
except FileNotFoundError:
    print(f"Error: GCN (RSSI-Dist Est.) error file not found at {GCN_RSSI_DIST_ERRORS_PATH}")
    gcn_rssi_dist_errors = None
except Exception as e:
    print(f"Error loading GCN (RSSI-Dist Est.) errors: {e}")
    gcn_rssi_dist_errors = None

try:
    gcn_raw_rssi_errors = np.load(GCN_RAW_RSSI_ERRORS_PATH)
    print(f"Loaded GCN (Raw RSSI) errors from: {GCN_RAW_RSSI_ERRORS_PATH}")
    print(f"GCN (Raw RSSI) errors shape: {gcn_raw_rssi_errors.shape}, Mean: {np.mean(gcn_raw_rssi_errors):.2f} m, Median: {np.median(gcn_raw_rssi_errors):.2f} m")
except FileNotFoundError:
    print(f"Error: GCN (Raw RSSI) error file not found at {GCN_RAW_RSSI_ERRORS_PATH}")
    gcn_raw_rssi_errors = None
except Exception as e:
    print(f"Error loading GCN (Raw RSSI) errors: {e}")
    gcn_raw_rssi_errors = None

try:
    multilateration_errors = np.load(MULTILATERATION_ERRORS_PATH)
    if np.iscomplexobj(multilateration_errors):
        multilateration_errors = np.abs(multilateration_errors)
    if multilateration_errors.dtype.names and 'error' in multilateration_errors.dtype.names:
        multilateration_errors = multilateration_errors['error']
    multilateration_errors = multilateration_errors.flatten()
    print(f"Loaded Multilateration errors from: {MULTILATERATION_ERRORS_PATH}")
    print(f"Multilateration errors shape: {multilateration_errors.shape}, Mean: {np.mean(multilateration_errors):.2f} m, Median: {np.median(multilateration_errors):.2f} m")
except FileNotFoundError:
    print(f"Error: Multilateration error file not found at {MULTILATERATION_ERRORS_PATH}")
    multilateration_errors = None
except Exception as e:
    print(f"Error loading Multilateration errors: {e}")
    multilateration_errors = None

# --- Plotting ---
if gcn_rssi_dist_errors is not None and gcn_raw_rssi_errors is not None and multilateration_errors is not None:
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(16, 9)) # Adjusted for potentially more data in histogram

    all_errors_for_bins = []
    if gcn_rssi_dist_errors is not None:
        all_errors_for_bins.append(gcn_rssi_dist_errors)
    if gcn_raw_rssi_errors is not None:
        all_errors_for_bins.append(gcn_raw_rssi_errors)
    if multilateration_errors is not None:
        all_errors_for_bins.append(multilateration_errors)
    
    combined_errors = np.concatenate(all_errors_for_bins)
    min_val = np.min(combined_errors)
    max_val = np.max(combined_errors)
    bin_width = 125
    start_bin = np.floor(min_val / bin_width) * bin_width
    end_bin_limit = max_val + bin_width
    bins = np.arange(start_bin, end_bin_limit, bin_width)

    if gcn_rssi_dist_errors is not None:
        plt.hist(gcn_rssi_dist_errors, bins=bins, alpha=0.7, label='GCN (Proposed) Errors', color='skyblue', edgecolor='black')
    if gcn_raw_rssi_errors is not None:
        plt.hist(gcn_raw_rssi_errors, bins=bins, alpha=0.7, label='GCN (No RSSI-Dist Est.) Errors', color='lightgreen', edgecolor='black')
    if multilateration_errors is not None:
        print(f"Multilateration errors shape: {multilateration_errors.shape}")
        plt.hist(multilateration_errors, bins=bins, alpha=0.7, label='Iterative Multilateration Errors', color='salmon', edgecolor='black')

    plt.xlabel('Localization Error (meters)', fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel('Number of Unknown Nodes', fontsize=AXIS_LABEL_FONT_SIZE)
    plt.xticks(fontsize=TICK_LABEL_FONT_SIZE)
    plt.yticks(fontsize=TICK_LABEL_FONT_SIZE)
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    plt.savefig(output_path, dpi=300)
    print(f"Comparison histogram saved to: {output_path}")
    plt.show()

    # --- Add Box Plot ---
    plt.figure(figsize=(12, 8)) # Adjusted for three boxes
    data_to_plot = []
    labels = []
    colors = []

    if gcn_rssi_dist_errors is not None:
        data_to_plot.append(gcn_rssi_dist_errors)
        labels.append('GCN (Proposed)')
        colors.append('skyblue')
    if gcn_raw_rssi_errors is not None:
        data_to_plot.append(gcn_raw_rssi_errors)
        labels.append('GCN (No RSSI-Dist Est.)')
        colors.append('lightgreen')
    if multilateration_errors is not None:
        data_to_plot.append(multilateration_errors)
        labels.append('Iterative Multilateration')
        colors.append('salmon')

    if data_to_plot:
        bp = plt.boxplot(data_to_plot, patch_artist=True, labels=labels, notch=True, vert=True, meanline=True, showmeans=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for median in bp['medians']:
            median.set(color='black', linewidth=2)
        for mean in bp['means']:
            mean.set(color='black', linestyle='--', linewidth=1.5)

        plt.title('Error Distribution Comparison', fontsize=TITLE_FONT_SIZE)
        plt.ylabel('Localization Error (meters)', fontsize=AXIS_LABEL_FONT_SIZE)
        plt.xticks(fontsize=TICK_LABEL_FONT_SIZE -2) # Adjust tick label if too crowded
        plt.yticks(fontsize=TICK_LABEL_FONT_SIZE)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')

        y_min, y_max = plt.ylim()
        text_y_pos_factor = 0.08 # Percentage of y-range for offset
        
        # Adjust text placement to avoid overlap, maybe reduce font or place differently
        for i, data_series in enumerate(data_to_plot):
            mean_val = np.mean(data_series)
            median_val = np.median(data_series)
            # Position text below boxes
            plt.text(i + 1, y_min - (y_max-y_min) * text_y_pos_factor, f'Mean: {mean_val:.1f}m\\nMedian: {median_val:.1f}m',
                     horizontalalignment='center', verticalalignment='top', fontsize=BOXPLOT_TEXT_FONT_SIZE -2, # Smaller text
                     bbox=dict(boxstyle='round,pad=0.2', fc=colors[i], alpha=0.5))

        plt.tight_layout(rect=[0, 0.08, 1, 0.95]) # Adjust bottom margin for text

        boxplot_output_filename = OUTPUT_FILENAME.replace("histogram", "boxplot")
        boxplot_output_path = os.path.join(OUTPUT_DIR, boxplot_output_filename)
        plt.savefig(boxplot_output_path, dpi=300)
        print(f"Comparison boxplot saved to: {boxplot_output_path}")
        plt.show()
    else:
        print("No data available to generate box plot.")

    # --- Add Side-by-Side Error Map Visualization ---
    print("\nAttempting to load images for side-by-side error map comparison...")

    MULTILATERATION_SAMPLE_IMG_PATH = "../multilateration/output_visualizations_matlab_im/64beacons_100instances_64N_16A_im_instance_1_visualization.png"
    
    # GCN (RSSI-Dist Est.) Image Path
    gcn_rssi_dist_model_name_part = os.path.basename(GCN_RSSI_DIST_ERRORS_PATH).replace("gcn_errors_", "").replace(".npy", "")
    GCN_RSSI_DIST_SAMPLE_VIS_FILENAME = f"sample_visualization_loaded_model_{gcn_rssi_dist_model_name_part}.png"
    GCN_RSSI_DIST_SAMPLE_IMG_PATH = os.path.join("../gcn/results/", GCN_RSSI_DIST_SAMPLE_VIS_FILENAME)

    # GCN (Raw RSSI) Image Path
    gcn_raw_rssi_model_name_part = os.path.basename(GCN_RAW_RSSI_ERRORS_PATH).replace("gcn_errors_", "").replace(".npy", "")
    GCN_RAW_RSSI_SAMPLE_VIS_FILENAME = f"sample_visualization_loaded_model_{gcn_raw_rssi_model_name_part}.png"
    GCN_RAW_RSSI_SAMPLE_IMG_PATH = os.path.join("../gcn/results/", GCN_RAW_RSSI_SAMPLE_VIS_FILENAME)


    img_ml = None
    img_gcn_rssi_dist = None
    img_gcn_raw_rssi = None
    
    # Load Multilateration Image
    if os.path.exists(MULTILATERATION_SAMPLE_IMG_PATH):
        try:
            img_ml = plt.imread(MULTILATERATION_SAMPLE_IMG_PATH)
            print(f"Loaded Multilateration sample image from: {MULTILATERATION_SAMPLE_IMG_PATH}")
        except Exception as e:
            print(f"Error loading Multilateration sample image: {e}")
    else:
        print(f"Multilateration sample image not found at: {MULTILATERATION_SAMPLE_IMG_PATH}")

    # Load GCN (RSSI-Dist Est.) Image
    if os.path.exists(GCN_RSSI_DIST_SAMPLE_IMG_PATH):
        try:
            img_gcn_rssi_dist = plt.imread(GCN_RSSI_DIST_SAMPLE_IMG_PATH)
            print(f"Loaded GCN (RSSI-Dist Est.) sample image from: {GCN_RSSI_DIST_SAMPLE_IMG_PATH}")
        except Exception as e:
            print(f"Error loading GCN (RSSI-Dist Est.) sample image: {e}")
    else:
        print(f"GCN (RSSI-Dist Est.) sample image not found at: {GCN_RSSI_DIST_SAMPLE_IMG_PATH}")
        print(f"(Expected GCN model part: {gcn_rssi_dist_model_name_part})")
        
    # Load GCN (Raw RSSI) Image
    if os.path.exists(GCN_RAW_RSSI_SAMPLE_IMG_PATH):
        try:
            img_gcn_raw_rssi = plt.imread(GCN_RAW_RSSI_SAMPLE_IMG_PATH)
            print(f"Loaded GCN (Raw RSSI) sample image from: {GCN_RAW_RSSI_SAMPLE_IMG_PATH}")
        except Exception as e:
            print(f"Error loading GCN (Raw RSSI) sample image: {e}")
    else:
        print(f"GCN (Raw RSSI) sample image not found at: {GCN_RAW_RSSI_SAMPLE_IMG_PATH}")
        print(f"(Expected GCN model part: {gcn_raw_rssi_model_name_part})")


    if img_ml is not None and img_gcn_rssi_dist is not None and img_gcn_raw_rssi is not None:
        fig, axes = plt.subplots(1, 3, figsize=(30, 10)) 
        fig.subplots_adjust(wspace=0.05, hspace=0.05)
        
        axes[0].imshow(img_ml)
        axes[0].set_title("Multilateration", fontsize=TITLE_FONT_SIZE)
        axes[0].axis('off')
        
        axes[1].imshow(img_gcn_rssi_dist)
        axes[1].set_title("GCN (RSSI-Dist Est.)", fontsize=TITLE_FONT_SIZE)
        axes[1].axis('off') 
        
        axes[2].imshow(img_gcn_raw_rssi)
        axes[2].set_title("GCN (Raw RSSI)", fontsize=TITLE_FONT_SIZE)
        axes[2].axis('off')
        
        side_by_side_filename = "error_map_side_by_side_comparison_three_methods.png"
        side_by_side_output_path = os.path.join(OUTPUT_DIR, side_by_side_filename)
        plt.savefig(side_by_side_output_path, dpi=300)
        print(f"Side-by-side error map comparison saved to: {side_by_side_output_path}")
        plt.show()
    else:
        missing_images = []
        if img_ml is None: missing_images.append("Multilateration")
        if img_gcn_rssi_dist is None: missing_images.append("GCN (RSSI-Dist Est.)")
        if img_gcn_raw_rssi is None: missing_images.append("GCN (Raw RSSI)")
        print(f"Skipping side-by-side plot: The following images are missing: {', '.join(missing_images)}.")

elif gcn_rssi_dist_errors is None or gcn_raw_rssi_errors is None or multilateration_errors is None:
    missing_data_sources = []
    if gcn_rssi_dist_errors is None: missing_data_sources.append("GCN (RSSI-Dist Est.)")
    if gcn_raw_rssi_errors is None: missing_data_sources.append("GCN (Raw RSSI)")
    if multilateration_errors is None: missing_data_sources.append("Multilateration")
    print(f"Cannot create plots because error data for the following sources failed to load: {', '.join(missing_data_sources)}.")
else: # Should not be reached if the above elif covers all cases
    print("Cannot create plots because one or more error datasets failed to load.")

print("\nScript finished.")
