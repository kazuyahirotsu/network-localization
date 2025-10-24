# %%
import os
import numpy as np
import matplotlib.pyplot as plt

# --- Publication Font Sizes (smaller for paper layout) ---
SLIDE_SUPTITLE_FONT_SIZE = 24
SLIDE_TITLE_FONT_SIZE = 24
SLIDE_AXIS_LABEL_FONT_SIZE = 22
SLIDE_TICK_LABEL_FONT_SIZE = 18
SLIDE_LEGEND_FONT_SIZE = 18

def load_eval_dump(eval_dump_path: str):
    data = np.load(eval_dump_path, allow_pickle=True)
    return data

def ensure_results_dir():
    os.makedirs('new_results', exist_ok=True)

def plot_histogram(errors: np.ndarray, model_name: str, metrics: dict):
    plt.figure(figsize=(10, 4.5))
    plt.hist(
        errors,
        bins=30,
        alpha=0.8,
        color='lightcoral',
        edgecolor='black',
        linewidth=0.5
    )
    plt.xlabel('Localization Error (meters)', fontsize=SLIDE_AXIS_LABEL_FONT_SIZE)
    plt.ylabel('Number of Nodes', fontsize=SLIDE_AXIS_LABEL_FONT_SIZE)
    # No title per requirement
    # No legend per requirement
    plt.xticks(fontsize=SLIDE_TICK_LABEL_FONT_SIZE)
    plt.yticks(fontsize=SLIDE_TICK_LABEL_FONT_SIZE)
    plt.grid(True, alpha=0.3)

    # metrics_text = (
    #     f"Mean: {metrics['mean']:.2f} m\n"
    #     f"Median: {metrics['median']:.2f} m\n"
    #     f"P90: {metrics['p90']:.2f} m\n"
    #     f"P95: {metrics['p95']:.2f} m"
    # )
    # ax = plt.gca()
    # ax.text(
    #     0.98, 0.98, metrics_text,
    #     transform=ax.transAxes,
    #     va='top', ha='right',
    #     fontsize=SLIDE_LEGEND_FONT_SIZE,
    #     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray')
    # )

    png = f'new_results/error_histogram_loaded_model_{model_name}.png'
    plt.tight_layout()
    plt.savefig(png, dpi=300, bbox_inches='tight')
    print(f'Saved: {png}')
    plt.show()

def plot_cdf(errors: np.ndarray, model_name: str):
    sorted_errors = np.sort(errors)
    cdf = np.arange(1, sorted_errors.size + 1) / sorted_errors.size

    plt.figure(figsize=(10, 4.5))
    plt.plot(sorted_errors, cdf, color='navy', linewidth=2, label='Empirical CDF')
    plt.xlabel('Localization Error (meters)', fontsize=SLIDE_AXIS_LABEL_FONT_SIZE)
    plt.ylabel('Cumulative Probability', fontsize=SLIDE_AXIS_LABEL_FONT_SIZE)
    # No title per requirement
    plt.xticks(fontsize=SLIDE_TICK_LABEL_FONT_SIZE)
    plt.yticks(fontsize=SLIDE_TICK_LABEL_FONT_SIZE)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    # plt.legend(fontsize=SLIDE_LEGEND_FONT_SIZE, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
    png = f'new_results/error_cdf_loaded_model_{model_name}.png'
    plt.tight_layout()
    plt.savefig(png, dpi=300, bbox_inches='tight')
    print(f'Saved: {png}')
    plt.show()

def plot_sample(first_true_positions: np.ndarray,
                first_pred_positions: np.ndarray,
                anchor_mask: np.ndarray,
                unknown_mask: np.ndarray,
                model_name: str):
    if first_true_positions.size == 0 or first_pred_positions.size == 0:
        print('No stored first-sample positions in dump; skipping sample plot.')
        return

    plt.figure(figsize=(10, 8))
    plt.scatter(first_true_positions[unknown_mask,0],
                first_true_positions[unknown_mask,1],
                c='blue', label='True Unknown Positions', alpha=0.6, s=50)
    plt.scatter(first_pred_positions[unknown_mask,0],
                first_pred_positions[unknown_mask,1],
                c='red', marker='x', label='Predicted Unknown Positions', alpha=0.8, s=50)
    plt.scatter(first_true_positions[anchor_mask,0],
                first_true_positions[anchor_mask,1],
                c='green', marker='^', s=100, label='Anchor Nodes', edgecolors='black')

    for node_idx in np.where(unknown_mask)[0]:
        tp = first_true_positions[node_idx]
        pp = first_pred_positions[node_idx]
        plt.plot([tp[0], pp[0]], [tp[1], pp[1]], 'r--', alpha=0.3)

    plt.xlabel('X Position (m)', fontsize=SLIDE_AXIS_LABEL_FONT_SIZE)
    plt.ylabel('Y Position (m)', fontsize=SLIDE_AXIS_LABEL_FONT_SIZE)
    plt.legend(fontsize=SLIDE_LEGEND_FONT_SIZE, loc='upper right')
    plt.xticks(fontsize=SLIDE_TICK_LABEL_FONT_SIZE)
    plt.yticks(fontsize=SLIDE_TICK_LABEL_FONT_SIZE)
    plt.grid(True)
    # Set axis limits tightly around data to avoid excessive whitespace
    all_x = np.concatenate([
        first_true_positions[:,0], first_pred_positions[:,0]
    ])
    all_y = np.concatenate([
        first_true_positions[:,1], first_pred_positions[:,1]
    ])
    x_min, x_max = float(np.min(all_x)), float(np.max(all_x))
    y_min, y_max = float(np.min(all_y)), float(np.max(all_y))
    x_pad = 0.05 * (x_max - x_min if x_max > x_min else 1.0)
    y_pad = 0.05 * (y_max - y_min if y_max > y_min else 1.0)
    plt.xlim(x_min - x_pad, x_max + x_pad)
    plt.ylim(y_min - y_pad, y_max + y_pad)
    plt.gca().set_aspect('equal', adjustable='box')
    png = f'new_results/sample_visualization_loaded_model_{model_name}.png'
    plt.tight_layout()
    plt.savefig(png, dpi=300, bbox_inches='tight')
    print(f'Saved: {png}')
    plt.show()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot evaluation results from saved .npz dump')
    parser.add_argument('--dump', required=True, help='Path to eval_dump_*.npz')
    args = parser.parse_args()

    ensure_results_dir()
    dump = load_eval_dump(args.dump)

    model_file = str(dump['model_file'])
    model_name = os.path.basename(model_file).replace('.pth', '')
    errors = dump['errors']
    metrics = {
        'mean': float(dump['mean']),
        'median': float(dump['median']),
        'p90': float(dump['p90']),
        'p95': float(dump['p95']),
    }

    # show metrics
    print(f"Mean: {metrics['mean']:.2f} m")
    print(f"Median: {metrics['median']:.2f} m")
    print(f"P90: {metrics['p90']:.2f} m")
    print(f"P95: {metrics['p95']:.2f} m")
    print(f"Num Samples: {errors.size}")

    plot_histogram(errors, model_name, metrics)
    plot_cdf(errors, model_name)
    plot_sample(
        dump['first_true_positions'],
        dump['first_pred_positions'],
        dump['first_anchor_mask'],
        dump['first_unknown_mask'],
        model_name
    )

if __name__ == '__main__':
    main()


