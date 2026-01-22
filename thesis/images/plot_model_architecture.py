"""
Generate model architecture diagram as PDF - matching original layout
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.lines as mlines

# Figure setup - wide aspect ratio like original
fig, ax = plt.subplots(1, 1, figsize=(22, 6))
ax.set_xlim(0, 22)
ax.set_ylim(0, 6)
ax.axis('off')

# Colors
BOX_COLOR = 'white'
BORDER_COLOR = 'black'
GROUP_COLOR = '#f5f5f5'
GROUP_BORDER = '#cccccc'

def draw_box(ax, x, y, w, h, text, fontsize=9):
    """Draw a box with centered text"""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                          facecolor=BOX_COLOR, edgecolor=BORDER_COLOR, linewidth=1)
    ax.add_patch(box)
    # Split text by newlines and center each line
    lines = text.split('\n')
    line_height = 0.22
    total_height = len(lines) * line_height
    start_y = y + h/2 + total_height/2 - line_height/2
    for i, line in enumerate(lines):
        ax.text(x + w/2, start_y - i*line_height, line, ha='center', va='center', fontsize=fontsize)

def draw_group(ax, x, y, w, h, title):
    """Draw a group box with title"""
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                          facecolor=GROUP_COLOR, edgecolor=GROUP_BORDER, linewidth=1)
    ax.add_patch(box)
    ax.text(x + 0.1, y + h - 0.15, title, ha='left', va='top', fontsize=9, fontweight='normal')

def draw_arrow(ax, start, end, color='black'):
    """Draw an arrow between two points"""
    arrow = FancyArrowPatch(start, end, connectionstyle="arc3,rad=0",
                            arrowstyle='->', mutation_scale=12, color=color, linewidth=1)
    ax.add_patch(arrow)

# === Draw Groups ===

# Inputs group
draw_group(ax, 0.2, 2.8, 2.0, 2.8, "Inputs")

# RSSI to Distance Module group
draw_group(ax, 2.8, 2.5, 2.8, 1.5, "RSSI to Distance Module")

# Edge Features group
draw_group(ax, 6.0, 2.2, 2.8, 2.0, "Edge Features")

# Edge-conditioned kernels group
draw_group(ax, 9.2, 1.8, 2.8, 2.8, "Edge-conditioned kernels")

# Edge-conditioned GCN group
draw_group(ax, 12.4, 1.8, 5.0, 2.8, "Edge-conditioned GCN")

# === Draw Boxes ===

# Inputs
draw_box(ax, 0.4, 4.3, 1.6, 1.0, "Node Features\n[x, y, is_anchor]\nR^3", fontsize=8)
draw_box(ax, 0.4, 3.0, 1.6, 0.9, "RSSI time series\nK=10", fontsize=8)

# RSSI to Distance Module
draw_box(ax, 3.0, 2.8, 1.2, 0.6, "RSSI to Distance", fontsize=8)
draw_box(ax, 4.4, 2.8, 1.0, 0.6, "d_hat", fontsize=8)

# Edge Features
draw_box(ax, 6.2, 2.6, 2.4, 1.2, "Edge Features\nR^11 = RSSI(10) + d_hat", fontsize=8)

# Edge-conditioned kernels
draw_box(ax, 9.4, 3.4, 2.4, 0.9, "EdgeNet1 MLP\n11 -> 64 -> (3 x H)", fontsize=8)
draw_box(ax, 9.4, 2.1, 2.4, 0.9, "EdgeNet2 MLP\n11 -> 64 -> (H x H)", fontsize=8)

# Edge-conditioned GCN
draw_box(ax, 12.6, 3.4, 2.0, 0.9, "NNConv1\nR^3 -> R^H\naggr=mean + ReLU", fontsize=8)
draw_box(ax, 12.6, 2.1, 2.0, 0.9, "NNConv2\nR^H -> R^H\naggr=mean + ReLU", fontsize=8)
draw_box(ax, 15.0, 2.6, 1.6, 0.9, "Linear\nR^H -> R^2", fontsize=8)

# Output (outside GCN group)
draw_box(ax, 18.0, 2.6, 1.8, 0.9, "Estimated Position\nR^2", fontsize=8)

# Anchors note (below)
draw_box(ax, 15.5, 0.5, 2.8, 0.9, "Anchors fixed\nexcluded from loss;\noverwritten at eval", fontsize=8)

# === Draw Arrows ===

# Node Features -> NNConv1 (long curved arrow at top)
ax.annotate('', xy=(12.6, 3.85), xytext=(2.0, 4.8),
            arrowprops=dict(arrowstyle='->', color='black', connectionstyle="arc3,rad=-0.15", lw=1))

# RSSI time series -> RSSI to Distance
draw_arrow(ax, (2.0, 3.45), (3.0, 3.1))

# RSSI to Distance -> d_hat
draw_arrow(ax, (4.2, 3.1), (4.4, 3.1))

# d_hat -> Edge Features
draw_arrow(ax, (5.4, 3.1), (6.2, 3.2))

# RSSI time series -> Edge Features (curved down)
ax.annotate('', xy=(6.2, 2.8), xytext=(1.2, 3.0),
            arrowprops=dict(arrowstyle='->', color='black', connectionstyle="arc3,rad=0.3", lw=1))

# Edge Features -> EdgeNet1
ax.annotate('', xy=(9.4, 3.85), xytext=(8.6, 3.4),
            arrowprops=dict(arrowstyle='->', color='black', connectionstyle="arc3,rad=-0.15", lw=1))

# Edge Features -> EdgeNet2
ax.annotate('', xy=(9.4, 2.55), xytext=(8.6, 2.9),
            arrowprops=dict(arrowstyle='->', color='black', connectionstyle="arc3,rad=0.15", lw=1))

# EdgeNet1 -> NNConv1 (weights)
ax.annotate('', xy=(12.6, 3.85), xytext=(11.8, 3.85),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1))
ax.text(12.2, 4.05, 'weights', fontsize=7, color='gray', ha='center')

# EdgeNet2 -> NNConv2 (weights)
ax.annotate('', xy=(12.6, 2.55), xytext=(11.8, 2.55),
            arrowprops=dict(arrowstyle='->', color='gray', lw=1))
ax.text(12.2, 2.35, 'weights', fontsize=7, color='gray', ha='center')

# NNConv1 -> NNConv2
ax.annotate('', xy=(13.6, 3.0), xytext=(13.6, 3.4),
            arrowprops=dict(arrowstyle='->', color='black', lw=1))

# NNConv2 -> Linear
draw_arrow(ax, (14.6, 2.9), (15.0, 3.0))

# Linear -> Estimated Position
draw_arrow(ax, (16.6, 3.05), (18.0, 3.05))

# Anchors note -> Estimated Position (dashed)
ax.annotate('', xy=(18.9, 2.6), xytext=(17.8, 1.4),
            arrowprops=dict(arrowstyle='->', color='gray', linestyle='dashed', 
                           connectionstyle="arc3,rad=-0.2", lw=1))

plt.tight_layout()
plt.savefig('thesis/images/model_architecture.pdf', dpi=300, bbox_inches='tight')
plt.savefig('thesis/images/model_architecture_new.png', dpi=300, bbox_inches='tight')
print("Saved: thesis/images/model_architecture.pdf")
print("Saved: thesis/images/model_architecture_new.png")
