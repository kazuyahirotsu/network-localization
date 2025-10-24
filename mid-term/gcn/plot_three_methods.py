# %%
import os
import numpy as np
import matplotlib.pyplot as plt

# --- Publication Font Sizes (smaller for paper layout) ---
SLIDE_AXIS_LABEL_FONT_SIZE = 22
SLIDE_TICK_LABEL_FONT_SIZE = 18
SLIDE_LEGEND_FONT_SIZE = 18


def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)


def load_errors(npz_path: str) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    return np.asarray(data['errors']).astype(float)


def compute_bins(arrays, num_bins: int = 40, clip_percentile: float = 99.5):
    max_val = 0.0
    for a in arrays:
        if a.size == 0:
            continue
        max_val = max(max_val, float(np.percentile(a, clip_percentile)))
    if max_val <= 0:
        max_val = 1.0
    bins = np.linspace(0.0, max_val, num_bins)
    return bins


def plot_histogram_three(errors_list, labels, colors, out_png):
    plt.figure(figsize=(10, 4.5))
    bins = compute_bins(errors_list, num_bins=40)
    for errs, lab, col in zip(errors_list, labels, colors):
        if errs.size == 0:
            continue
        plt.hist(errs, bins=bins, histtype='step', linewidth=2.0, color=col, label=lab)

    plt.xlabel('Localization Error (meters)', fontsize=SLIDE_AXIS_LABEL_FONT_SIZE)
    plt.ylabel('Number of Nodes', fontsize=SLIDE_AXIS_LABEL_FONT_SIZE)
    plt.xticks(fontsize=SLIDE_TICK_LABEL_FONT_SIZE)
    plt.yticks(fontsize=SLIDE_TICK_LABEL_FONT_SIZE)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=SLIDE_LEGEND_FONT_SIZE, loc='upper right')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f'Saved: {out_png}')
    plt.close()


def plot_cdf_three(errors_list, labels, colors, out_png):
    plt.figure(figsize=(10, 4.5))
    for errs, lab, col in zip(errors_list, labels, colors):
        if errs.size == 0:
            continue
        s = np.sort(errs)
        cdf = np.arange(1, s.size + 1) / s.size
        plt.plot(s, cdf, color=col, linewidth=2.0, label=lab)

    plt.xlabel('Localization Error (meters)', fontsize=SLIDE_AXIS_LABEL_FONT_SIZE)
    plt.ylabel('Cumulative Probability', fontsize=SLIDE_AXIS_LABEL_FONT_SIZE)
    plt.xticks(fontsize=SLIDE_TICK_LABEL_FONT_SIZE)
    plt.yticks(fontsize=SLIDE_TICK_LABEL_FONT_SIZE)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    plt.legend(fontsize=SLIDE_LEGEND_FONT_SIZE, loc='lower right')
    plt.tight_layout()
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    print(f'Saved: {out_png}')
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot Histogram and CDF for three methods from their eval dumps (.npz)')
    parser.add_argument('--proposed', required=True, help='Path to Proposed Method eval_dump .npz (e.g., power13)')
    parser.add_argument('--plain', required=True, help='Path to Plain GCN eval_dump .npz (e.g., no_rssi2dist)')
    parser.add_argument('--mlat', required=True, help='Path to Iterative Multilateration eval_dump .npz')
    parser.add_argument('--outdir', default='new_results', help='Output directory')
    parser.add_argument('--tag', default='comparison', help='Tag appended to output filenames (e.g., fixed, free)')
    args = parser.parse_args()

    ensure_outdir(args.outdir)

    errors_proposed = load_errors(args.proposed)
    errors_plain = load_errors(args.plain)
    errors_mlat = load_errors(args.mlat)

    labels = ['Proposed Method', 'Plain GCN', 'Iterative Multilateration']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    errors_list = [errors_proposed, errors_plain, errors_mlat]

    hist_png = os.path.join(args.outdir, f'three_methods_hist_{args.tag}.png')
    cdf_png = os.path.join(args.outdir, f'three_methods_cdf_{args.tag}.png')

    plot_histogram_three(errors_list, labels, colors, hist_png)
    plot_cdf_three(errors_list, labels, colors, cdf_png)


if __name__ == '__main__':
    main()


