import numpy as np
import matplotlib.pyplot as plt
import os

# --- Configuration ---
GCN_ERRORS_PATH = "../gcn/results/gcn_errors_trained_localization_model_64beacons_1000instances_fixed.npy"
MULTILATERATION_ERRORS_PATH = "../multilateration/output_visualizations_matlab_im/64beacons_100instances_64N_16A_im_simple_errors_raw.npy"
OUTPUT_DIR = "comparison_results"
OUTPUT_FILENAME = "error_comparison_histogram.png"

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
    gcn_errors = np.load(GCN_ERRORS_PATH)
    print(f"Loaded GCN errors from: {GCN_ERRORS_PATH}")
    print(f"GCN errors shape: {gcn_errors.shape}, Mean: {np.mean(gcn_errors):.2f} m, Median: {np.median(gcn_errors):.2f} m")
except FileNotFoundError:
    print(f"Error: GCN error file not found at {GCN_ERRORS_PATH}")
    gcn_errors = None
except Exception as e:
    print(f"Error loading GCN errors: {e}")
    gcn_errors = None

try:
    multilateration_errors = np.load(MULTILATERATION_ERRORS_PATH)
    # Assuming multilateration errors might be stored differently, e.g., complex numbers or structured array.
    # We need to ensure it's a flat array of magnitudes if that's the case.
    # For now, assuming it's a simple array of error values.
    # If it's complex (e.g., error vectors), take the magnitude:
    if np.iscomplexobj(multilateration_errors):
        multilateration_errors = np.abs(multilateration_errors)
    # If it's a structured array, try to extract a relevant field, e.g., 'error'
    # This is a placeholder; you might need to adjust based on actual structure.
    if multilateration_errors.dtype.names and 'error' in multilateration_errors.dtype.names:
        multilateration_errors = multilateration_errors['error']
    
    # Flatten the array in case it's multi-dimensional for some reason
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
if gcn_errors is not None and multilateration_errors is not None:
    plt.style.use('seaborn-v0_8-whitegrid') # Using a seaborn style for better aesthetics
    plt.figure(figsize=(14, 8))

    # Determine common bins with a width of 250m
    # Combine all errors to find overall min and max for bin calculation
    combined_errors = np.concatenate((gcn_errors, multilateration_errors))
    min_val = np.min(combined_errors)
    max_val = np.max(combined_errors)

    bin_width = 250  # meters

    # Calculate bin edges starting from a multiple of bin_width <= min_val,
    # and ending at a multiple of bin_width that covers max_val.
    start_bin = np.floor(min_val / bin_width) * bin_width
    # The 'stop' parameter in np.arange is exclusive, so ensure the range covers max_val.
    end_bin_limit = max_val + bin_width

    bins = np.arange(start_bin, end_bin_limit, bin_width)

    plt.hist(gcn_errors, bins=bins, alpha=0.7, label='GCN Errors', color='skyblue', edgecolor='black')
    plt.hist(multilateration_errors, bins=bins, alpha=0.7, label='Iterative Multilateration Errors', color='salmon', edgecolor='black')

    # plt.suptitle('Comparison of Localization Errors', fontsize=SUPTITLE_FONT_SIZE, fontweight='bold')
    # plt.title('GCN vs. Multilateration (64 Beacons, 1000 Instances)', fontsize=TITLE_FONT_SIZE)
    plt.xlabel('Localization Error (meters)', fontsize=AXIS_LABEL_FONT_SIZE)
    plt.ylabel('Number of Unknown Nodes', fontsize=AXIS_LABEL_FONT_SIZE)
    
    plt.xticks(fontsize=TICK_LABEL_FONT_SIZE)
    plt.yticks(fontsize=TICK_LABEL_FONT_SIZE)
    
    plt.legend(fontsize=LEGEND_FONT_SIZE)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # # Add text for mean and median errors
    # text_y_position = 0.95
    # plt.text(0.98, text_y_position, f"GCN: Mean={np.mean(gcn_errors):.2f}m, Median={np.median(gcn_errors):.2f}m",
    #          horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes,
    #          fontsize=LEGEND_FONT_SIZE - 2, bbox=dict(boxstyle='round,pad=0.3', fc='skyblue', alpha=0.5))
    
    # text_y_position -= 0.07 # Adjust spacing
    # plt.text(0.98, text_y_position, f"Multilateration: Mean={np.mean(multilateration_errors):.2f}m, Median={np.median(multilateration_errors):.2f}m",
    #          horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes,
    #          fontsize=LEGEND_FONT_SIZE - 2, bbox=dict(boxstyle='round,pad=0.3', fc='salmon', alpha=0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle

    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    plt.savefig(output_path, dpi=300) # Save with high DPI for paper quality
    print(f"Comparison histogram saved to: {output_path}")
    
    plt.show()

    # --- Add Box Plot ---
    plt.figure(figsize=(10, 8)) # Adjusted figure size for box plot
    data_to_plot = []
    labels = []
    colors = []

    if gcn_errors is not None:
        data_to_plot.append(gcn_errors)
        labels.append('GCN Errors')
        colors.append('skyblue')
    if multilateration_errors is not None:
        data_to_plot.append(multilateration_errors)
        labels.append('Iterative Multilateration Errors')
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
        
        # plt.suptitle('Comparison of Localization Errors (Box Plot)', fontsize=SUPTITLE_FONT_SIZE, fontweight='bold')
        plt.title('Error Distribution Comparison', fontsize=TITLE_FONT_SIZE)
        plt.ylabel('Localization Error (meters)', fontsize=AXIS_LABEL_FONT_SIZE)
        plt.xticks(fontsize=TICK_LABEL_FONT_SIZE) # For x-axis category labels
        plt.yticks(fontsize=TICK_LABEL_FONT_SIZE)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y') # Grid for y-axis only

        # Add text for mean and median to the side or legend if needed, 
        # but boxplot inherently shows median, and optionally mean.
        # For clarity, let's add mean/median text below each box if possible, or as a summary.
        y_min, y_max = plt.ylim()
        text_y_offset = (y_max - y_min) * 0.05 # Adjust as needed

        for i, data_series in enumerate(data_to_plot):
            mean_val = np.mean(data_series)
            median_val = np.median(data_series)
            plt.text(i + 1, y_min - text_y_offset, f'Mean: {mean_val:.2f}m\\nMedian: {median_val:.2f}m',
                     horizontalalignment='center', verticalalignment='top', fontsize=BOXPLOT_TEXT_FONT_SIZE,
                     bbox=dict(boxstyle='round,pad=0.3', fc=colors[i], alpha=0.5))

        plt.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout to make space for suptitle and bottom text

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
    
    # Derive GCN sample image path from GCN_ERRORS_PATH
    gcn_model_name_part = os.path.basename(GCN_ERRORS_PATH).replace("gcn_errors_", "").replace(".npy", "")
    GCN_SAMPLE_VIS_FILENAME = f"sample_visualization_loaded_model_{gcn_model_name_part}.png"
    GCN_SAMPLE_IMG_PATH = os.path.join("../gcn/results/", GCN_SAMPLE_VIS_FILENAME)

    img_ml = None
    img_gcn = None
    
    if os.path.exists(MULTILATERATION_SAMPLE_IMG_PATH):
        try:
            img_ml = plt.imread(MULTILATERATION_SAMPLE_IMG_PATH)
            print(f"Loaded Multilateration sample image from: {MULTILATERATION_SAMPLE_IMG_PATH}")
        except Exception as e:
            print(f"Error loading Multilateration sample image: {e}")
    else:
        print(f"Multilateration sample image not found at: {MULTILATERATION_SAMPLE_IMG_PATH}")

    if os.path.exists(GCN_SAMPLE_IMG_PATH):
        try:
            img_gcn = plt.imread(GCN_SAMPLE_IMG_PATH)
            print(f"Loaded GCN sample image from: {GCN_SAMPLE_IMG_PATH}")
        except Exception as e:
            print(f"Error loading GCN sample image: {e}")
    else:
        print(f"GCN sample image not found at: {GCN_SAMPLE_IMG_PATH}")
        print(f"(Expected GCN model part: {gcn_model_name_part})")

    if img_ml is not None and img_gcn is not None:
        fig, axes = plt.subplots(1, 2, figsize=(20, 10)) 
        fig.subplots_adjust(wspace=0.05, hspace=0.05) # Add/adjust wspace for tighter gap
        
        axes[0].imshow(img_ml)
        axes[0].set_title("Multilateration", fontsize=TITLE_FONT_SIZE) # Use updated TITLE_FONT_SIZE
        axes[0].axis('off')
        
        axes[1].imshow(img_gcn)
        axes[1].set_title("GCN", fontsize=TITLE_FONT_SIZE) # Use updated TITLE_FONT_SIZE
        axes[1].axis('off') 
        
        # fig.suptitle('Side-by-Side Error Map Comparison', fontsize=SUPTITLE_FONT_SIZE, fontweight='bold') # Keep suptitle if desired
        # plt.tight_layout(rect=[0, 0, 1, 0.95]) # tight_layout might fight with subplots_adjust, often one is enough or needs careful coordination
        # If using suptitle, tight_layout with rect is good. Otherwise, subplots_adjust might be cleaner for just image spacing.
        
        side_by_side_filename = "error_map_side_by_side_comparison.png"
        side_by_side_output_path = os.path.join(OUTPUT_DIR, side_by_side_filename)
        plt.savefig(side_by_side_output_path, dpi=300)
        print(f"Side-by-side error map comparison saved to: {side_by_side_output_path}")
        plt.show()
    elif img_ml is None:
        print("Skipping side-by-side plot: Multilateration image missing.")
    elif img_gcn is None:
        print("Skipping side-by-side plot: GCN image missing.")

elif gcn_errors is None:
    print("Cannot create plot because GCN errors failed to load.")
elif multilateration_errors is None:
    print("Cannot create plot because Multilateration errors failed to load.")
else:
    print("Cannot create plot because both GCN and Multilateration errors failed to load.")

print("\nScript finished.")
