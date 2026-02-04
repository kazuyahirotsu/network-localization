import numpy as np
import matplotlib.pyplot as plt

# Error range
r = np.linspace(-10, 10, 500)

# Squared Loss (L2)
def squared_loss(r):
    return r**2

# Huber Loss (modified to match L2 up to delta)
# |r| <= delta: r^2 (same as L2)
# |r| > delta: 2*delta*|r| - delta^2 (linear, continuous)
def huber_loss(r, delta):
    return np.where(np.abs(r) <= delta, 
                    r**2, 
                    2 * delta * np.abs(r) - delta**2)

# Create figure
fig, ax = plt.subplots(figsize=(8, 5))

# Plot Squared Loss
ax.plot(r, squared_loss(r), 'r-', linewidth=2.5, label='Squared (L2)')

# Plot Huber Loss with multiple deltas
deltas = [2, 4, 6]
colors = ['#1f77b4', '#2ca02c', '#9467bd']  # blue, green, purple

for delta, color in zip(deltas, colors):
    ax.plot(r, huber_loss(r, delta), linewidth=2, color=color, label=f'Huber (δ={delta})')

# Labels and styling
ax.set_xlabel('Residual (r)', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Squared Loss vs Huber Loss', fontsize=14)
ax.legend(fontsize=11, loc='upper center', ncol=2)
ax.grid(True, alpha=0.3)
ax.set_xlim(-10, 10)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('loss_comparison.png', dpi=150, bbox_inches='tight')
print("Saved to loss_comparison.png")
