import matplotlib
matplotlib.use("Agg")

import numpy as np
import h5py
from scipy.optimize import minimize, curve_fit
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import json
import os
warnings.filterwarnings('ignore')

# Conversion factor: mean_2d_error = sigma * sqrt(pi/2)
RAYLEIGH_MEAN_FACTOR = np.sqrt(np.pi / 2)  # ≈ 1.2533

def sigma_to_mean_2d(sigma):
    """Convert per-axis σ to expected mean 2D error"""
    return sigma * RAYLEIGH_MEAN_FACTOR

def mean_2d_to_sigma(mean_2d):
    """Convert target mean 2D error to per-axis σ"""
    return mean_2d / RAYLEIGH_MEAN_FACTOR

print(f"Rayleigh mean factor: {RAYLEIGH_MEAN_FACTOR:.4f}")
print(f"Example: σ=100m → mean 2D error = {sigma_to_mean_2d(100):.1f}m")
print(f"Example: target 200m mean 2D → σ = {mean_2d_to_sigma(200):.1f}m")
print(f"Example: target 206m mean 2D → σ = {mean_2d_to_sigma(206):.1f}m (GCN result)")


print("Loading high-resolution RSSI data...")

with h5py.File('data/drone_beacon_simulation_data_large.mat', 'r') as f:
    beacon_lat = np.array(f['beaconLatitudes']).flatten()
    beacon_lon = np.array(f['beaconLongitudes']).flatten()
    beacon_x = np.array(f['beaconXMeters']).flatten()
    beacon_y = np.array(f['beaconYMeters']).flatten()
    beacon_height = float(np.array(f['beaconHeight']).flatten()[0])

    drone_x = np.array(f['droneXMeters']).flatten()
    drone_y = np.array(f['droneYMeters']).flatten()
    drone_altitude = float(np.array(f['droneAltitude']).flatten()[0])

    rssi_matrix = np.array(f['rssi_matrix']).T  # [num_positions x num_beacons]
    true_distances = np.array(f['true_distances']).T

    metadata_group = f['metadata']
    map_size_meters = float(np.array(metadata_group['mapSizeMeters']).flatten()[0])
    drone_grid_size = int(np.array(metadata_group['droneGridSize']).flatten()[0])

num_beacons = len(beacon_x)
num_positions = len(drone_x)
grid_spacing = map_size_meters / drone_grid_size

print(f"\n✓ Data loaded:")
print(f"  - Beacons: {num_beacons}")
print(f"  - Grid: {drone_grid_size}×{drone_grid_size} = {num_positions} positions")
print(f"  - Grid spacing: {grid_spacing:.1f}m")
print(f"  - Map size: {map_size_meters:.0f}×{map_size_meters:.0f}m")
print(f"  - Drone altitude: {drone_altitude}m")


# Fit path loss model on LOS signals
def log_distance_model(d, A, n):
    return A - 10 * n * np.log10(np.maximum(d, 1e-6))

def rssi_to_distance(rssi, A, n):
    return 10 ** ((A - rssi) / (10 * n))

LOS_THRESHOLD = -100
rssi_flat = rssi_matrix.flatten()
dist_flat = true_distances.flatten()
los_mask = rssi_flat > LOS_THRESHOLD

popt, _ = curve_fit(log_distance_model, dist_flat[los_mask], rssi_flat[los_mask], p0=[-30, 2.0])
A_fit, n_fit = popt
print(f"Path loss model: RSSI = {A_fit:.2f} - 10×{n_fit:.2f}×log₁₀(d)")


def huber_loss(residual, delta=10.0):
    abs_r = np.abs(residual)
    return np.where(abs_r <= delta, 0.5 * residual**2, delta * (abs_r - 0.5 * delta))

def objective_function(pos, beacon_locs, estimated_distances, z_drone, delta=10.0):
    x, y = pos
    total_loss = 0
    for i, (bx, by, bz) in enumerate(beacon_locs):
        calc_dist = np.sqrt((x - bx)**2 + (y - by)**2 + (z_drone - bz)**2)
        residual = calc_dist - estimated_distances[i]
        total_loss += huber_loss(residual, delta)
    return total_loss

def multilaterate_top_k(rssi_values, bx, by, bz, A, n, z_drone, k=10, delta=10.0):
    """Multilateration using top-K strongest signals with Huber loss"""
    top_k_idx = np.argsort(rssi_values)[-k:]
    top_k_rssi = rssi_values[top_k_idx]
    top_k_locs = [(bx[i], by[i], bz) for i in top_k_idx]
    est_distances = rssi_to_distance(top_k_rssi, A, n)
    
    init_x = np.mean([loc[0] for loc in top_k_locs])
    init_y = np.mean([loc[1] for loc in top_k_locs])
    
    result = minimize(
        objective_function, x0=[init_x, init_y],
        args=(top_k_locs, est_distances, z_drone, delta),
        method='L-BFGS-B',
        bounds=[(0, map_size_meters), (0, map_size_meters)]
    )
    return result.x[0], result.x[1]

print("Multilateration functions defined.")


def evaluate_localization_full_grid(beacon_mean_2d_error, num_samples=2000, k=10):
    """
    Evaluate localization accuracy on randomly sampled grid points.
    
    Args:
        beacon_mean_2d_error: Target mean 2D error for beacon positions (meters)
        num_samples: Number of random positions to test
        k: Number of top beacons to use
    
    Returns:
        dict with error statistics
    """
    # Convert mean 2D error to per-axis sigma
    sigma = mean_2d_to_sigma(beacon_mean_2d_error)
    
    # Sample random positions from the grid
    np.random.seed(42)  # Reproducibility
    sample_indices = np.random.choice(num_positions, size=min(num_samples, num_positions), replace=False)
    
    errors = []
    
    for idx in tqdm(sample_indices, desc=f"Beacon error={beacon_mean_2d_error}m", leave=False):
        true_x, true_y = drone_x[idx], drone_y[idx]
        rssi_values = rssi_matrix[idx]
        
        # Add noise to beacon positions
        if sigma > 0:
            noisy_bx = beacon_x + np.random.normal(0, sigma, num_beacons)
            noisy_by = beacon_y + np.random.normal(0, sigma, num_beacons)
        else:
            noisy_bx, noisy_by = beacon_x, beacon_y
        
        # Multilaterate
        est_x, est_y = multilaterate_top_k(
            rssi_values, noisy_bx, noisy_by, beacon_height,
            A_fit, n_fit, drone_altitude, k=k
        )
        
        error = np.sqrt((est_x - true_x)**2 + (est_y - true_y)**2)
        errors.append(error)
    
    errors = np.array(errors)
    
    return {
        'beacon_mean_2d_error': beacon_mean_2d_error,
        'sigma': sigma,
        'mean': np.mean(errors),
        'median': np.median(errors),
        'std': np.std(errors),
        'p90': np.percentile(errors, 90),
        'p95': np.percentile(errors, 95),
        'errors': errors
    }


# Define beacon error levels (mean 2D error in meters)
# Include 206m to match GCN result exactly
BEACON_ERROR_LEVELS = [0, 50, 100, 150, 200, 206, 250, 300]

print("="*70)
print("FULL-GRID LOCALIZATION ACCURACY VS BEACON POSITION ERROR")
print("="*70)
print(f"\nBeacon error levels (mean 2D): {BEACON_ERROR_LEVELS}")
print(f"Testing on 2000 random grid positions...\n")

localization_results = []

for beacon_error in BEACON_ERROR_LEVELS:
    result = evaluate_localization_full_grid(beacon_error, num_samples=2000, k=10)
    localization_results.append(result)
    print(f"  Beacon error {beacon_error:>3}m → UAV mean error: {result['mean']:.1f}m, median: {result['median']:.1f}m")

print("\n" + "="*70)


# Generate Figure: Sensitivity Analysis
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

beacon_errors = [r['beacon_mean_2d_error'] for r in localization_results]
uav_means = [r['mean'] for r in localization_results]
uav_medians = [r['median'] for r in localization_results]
uav_p90s = [r['p90'] for r in localization_results]
uav_stds = [r['std'] for r in localization_results]

# Panel 1: UAV Error vs Beacon Error
ax1 = axes[0]
ax1.fill_between(beacon_errors, np.array(uav_means)-np.array(uav_stds), 
                 np.array(uav_means)+np.array(uav_stds), alpha=0.2, color='blue')
ax1.plot(beacon_errors, uav_means, 'o-', lw=2, ms=8, color='blue', label='Mean ± std')
ax1.plot(beacon_errors, uav_medians, 's--', lw=2, ms=7, color='green', label='Median')
ax1.plot(beacon_errors, uav_p90s, '^:', lw=2, ms=7, color='red', label='90th percentile')

# Mark GCN result (206m)
gcn_idx = beacon_errors.index(206)
ax1.axvline(206, color='purple', ls='--', alpha=0.7, label='GCN result (206m)')
ax1.scatter([206], [uav_means[gcn_idx]], s=150, c='purple', zorder=5, marker='*')

ax1.set_xlabel('Beacon Position Error (mean 2D, m)', fontsize=11)
ax1.set_ylabel('UAV Localization Error (m)', fontsize=11)
ax1.set_title('Sensitivity: UAV Error vs Beacon Uncertainty', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Panel 2: Error Degradation (relative)
ax2 = axes[1]
baseline = localization_results[0]['mean']
degradation = [(r['mean'] - baseline) / baseline * 100 for r in localization_results]
colors = ['green' if d < 20 else 'orange' if d < 50 else 'red' for d in degradation]
bars = ax2.bar(range(len(beacon_errors)), degradation, color=colors, edgecolor='black')
ax2.set_xticks(range(len(beacon_errors)))
ax2.set_xticklabels([str(e) for e in beacon_errors], rotation=45)
ax2.set_xlabel('Beacon Position Error (mean 2D, m)', fontsize=11)
ax2.set_ylabel('UAV Error Increase (%)', fontsize=11)
ax2.set_title('Error Degradation from Baseline\n(Green<20%, Orange<50%, Red≥50%)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Panel 3: Summary Table
ax3 = axes[2]
ax3.axis('off')
table_data = [['Beacon Error\n(mean 2D)', 'UAV Mean\nError', 'UAV Median\nError', 'Degradation']]
for i, r in enumerate(localization_results):
    deg = (r['mean'] - baseline) / baseline * 100
    table_data.append([
        f"{r['beacon_mean_2d_error']}m",
        f"{r['mean']:.1f}m",
        f"{r['median']:.1f}m",
        f"+{deg:.1f}%" if deg > 0 else "baseline"
    ])

table = ax3.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)
for j in range(4):
    table[(0, j)].set_facecolor('#4472C4')
    table[(0, j)].set_text_props(color='white', fontweight='bold')
# Highlight GCN row
gcn_row = beacon_errors.index(206) + 1
for j in range(4):
    table[(gcn_row, j)].set_facecolor('#E6E6FA')

ax3.set_title('Quantitative Results', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('ch6_sensitivity_analysis.png', dpi=150, bbox_inches='tight')
print("Saved: ch6_sensitivity_analysis.png")
plt.show()


# Create RSSI interpolators for continuous flight
unique_x = np.unique(drone_x)
unique_y = np.unique(drone_y)
rssi_3d = rssi_matrix.reshape(drone_grid_size, drone_grid_size, num_beacons)

rssi_interpolators = []
for b in range(num_beacons):
    interp = RegularGridInterpolator(
        (unique_x, unique_y), rssi_3d[:, :, b],
        method='linear', bounds_error=False, fill_value=None
    )
    rssi_interpolators.append(interp)

def get_rssi_at_position(x, y):
    return np.array([interp((x, y)) for interp in rssi_interpolators])

print(f"Created {num_beacons} RSSI interpolators for continuous flight simulation.")


class DroneNavigator:
    """Closed-loop drone navigation using estimated position only."""
    
    def __init__(self, start_pos, speed=15.0, control_gain=0.3, 
                 waypoint_threshold=200, max_steps=2000):
        self.true_x, self.true_y = start_pos
        self.speed = speed
        self.control_gain = control_gain
        self.waypoint_threshold = waypoint_threshold
        self.max_steps = max_steps
        self.dt = 1.0
        
        self.true_path = [(self.true_x, self.true_y)]
        self.est_path = []
        self.errors = []
        self.headings_true = []
        self.headings_est = []
    
    def get_position_estimate(self, beacon_mean_2d_error=0, k=10):
        rssi = get_rssi_at_position(self.true_x, self.true_y)
        sigma = mean_2d_to_sigma(beacon_mean_2d_error)
        
        if sigma > 0:
            bx = beacon_x + np.random.normal(0, sigma, num_beacons)
            by = beacon_y + np.random.normal(0, sigma, num_beacons)
        else:
            bx, by = beacon_x, beacon_y
        
        return multilaterate_top_k(rssi, bx, by, beacon_height, 
                                   A_fit, n_fit, drone_altitude, k=k)
    
    def fly_to_waypoint(self, target_x, target_y, beacon_mean_2d_error=0, k=10):
        steps = 0
        while steps < self.max_steps:
            steps += 1
            
            # Get position estimate
            est_x, est_y = self.get_position_estimate(beacon_mean_2d_error, k)
            self.est_path.append((est_x, est_y))
            
            # Record error
            error = np.sqrt((est_x - self.true_x)**2 + (est_y - self.true_y)**2)
            self.errors.append(error)
            
            # Check arrival (using estimated position)
            est_dist = np.sqrt((target_x - est_x)**2 + (target_y - est_y)**2)
            if est_dist < self.waypoint_threshold:
                return {'reached': True, 'steps': steps, 
                        'true_final_dist': np.sqrt((target_x-self.true_x)**2 + (target_y-self.true_y)**2)}
            
            # Compute heading from estimated position
            dx, dy = target_x - est_x, target_y - est_y
            dist = np.sqrt(dx**2 + dy**2)
            heading_est = np.arctan2(dy, dx)
            
            # True heading (for analysis)
            true_dx, true_dy = target_x - self.true_x, target_y - self.true_y
            heading_true = np.arctan2(true_dy, true_dx)
            
            self.headings_est.append(heading_est)
            self.headings_true.append(heading_true)
            
            # Move drone
            vx = (dx / dist) * self.speed * self.control_gain
            vy = (dy / dist) * self.speed * self.control_gain
            self.true_x = np.clip(self.true_x + vx * self.dt, 200, map_size_meters - 200)
            self.true_y = np.clip(self.true_y + vy * self.dt, 200, map_size_meters - 200)
            self.true_path.append((self.true_x, self.true_y))
        
        return {'reached': False, 'steps': steps,
                'true_final_dist': np.sqrt((target_x-self.true_x)**2 + (target_y-self.true_y)**2)}
    
    def fly_mission(self, waypoints, beacon_mean_2d_error=0, k=10):
        results = []
        for wx, wy in waypoints:
            result = self.fly_to_waypoint(wx, wy, beacon_mean_2d_error, k)
            results.append(result)
        return results

print("DroneNavigator class defined.")


# Define flight patterns
def generate_flight_patterns(map_size=4000, margin=400):
    patterns = {}
    
    # Circular patrol
    center = map_size / 2
    radius = map_size / 3
    angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
    circular = [(center + radius*np.cos(a), center + radius*np.sin(a)) for a in angles]
    circular.append(circular[0])
    patterns['circular'] = circular
    
    # Diagonal transit
    patterns['diagonal'] = [(margin, margin), (map_size-margin, map_size-margin)]
    
    # Lawnmower coverage
    lawn = []
    y_vals = np.linspace(margin, map_size-margin, 5)
    for i, y in enumerate(y_vals):
        if i % 2 == 0:
            lawn.extend([(margin, y), (map_size-margin, y)])
        else:
            lawn.extend([(map_size-margin, y), (margin, y)])
    patterns['lawnmower'] = lawn
    
    # Figure-8
    t = np.linspace(0, 2*np.pi, 12)
    fig8_x = center + radius * np.sin(t)
    fig8_y = center + radius/2 * np.sin(2*t)
    patterns['figure8'] = [(fig8_x[i], fig8_y[i]) for i in range(len(t))]
    
    return patterns

FLIGHT_PATTERNS = generate_flight_patterns(map_size_meters)
print("Flight patterns defined:")
for name, pts in FLIGHT_PATTERNS.items():
    total = sum(np.sqrt((pts[i+1][0]-pts[i][0])**2 + (pts[i+1][1]-pts[i][1])**2) for i in range(len(pts)-1))
    print(f"  {name}: {len(pts)} waypoints, {total/1000:.1f} km")


# Geometric analysis: heading error vs distance for different position errors
distances = np.linspace(100, 3000, 100)
position_errors = [50, 100, 150, 200, 250]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Heading error vs distance
ax1 = axes[0]
for pos_err in position_errors:
    # Heading error ≈ arctan(pos_error / distance) for worst case
    heading_errors = np.degrees(np.arctan(pos_err / distances))
    ax1.plot(distances/1000, heading_errors, lw=2, label=f'{pos_err}m position error')

ax1.axhline(10, color='red', ls='--', alpha=0.7, label='10° threshold')
ax1.fill_between(distances/1000, 0, 10, alpha=0.1, color='green')
ax1.set_xlabel('Distance to Target (km)', fontsize=11)
ax1.set_ylabel('Heading Error (degrees)', fontsize=11)
ax1.set_title('Geometric Stability: Heading Error vs Distance', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0, 60])

# Panel 2: Safe operating distance for different position errors
ax2 = axes[1]
target_heading_error = 10  # degrees
safe_distances = []
pos_err_range = np.linspace(50, 300, 20)

for pe in pos_err_range:
    # distance where heading error = target
    safe_dist = pe / np.tan(np.radians(target_heading_error))
    safe_distances.append(safe_dist)

ax2.fill_between(pos_err_range, safe_distances, 3000, alpha=0.2, color='green', label='Safe zone (<10° heading error)')
ax2.fill_between(pos_err_range, 0, safe_distances, alpha=0.2, color='red', label='Unstable zone (>10° heading error)')
ax2.plot(pos_err_range, safe_distances, 'k-', lw=2)

# Mark GCN error level
ax2.axvline(206, color='purple', ls='--', lw=2, label='GCN beacon error (206m)')
gcn_safe_dist = 206 / np.tan(np.radians(10))
ax2.scatter([206], [gcn_safe_dist], s=100, c='purple', zorder=5)
ax2.annotate(f'{gcn_safe_dist:.0f}m', (206, gcn_safe_dist), 
             xytext=(230, gcn_safe_dist+200), fontsize=10)

ax2.set_xlabel('Position Error (m)', fontsize=11)
ax2.set_ylabel('Distance to Target (m)', fontsize=11)
ax2.set_title('Safe Operating Distance\n(for <10° heading error)', fontsize=12, fontweight='bold')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ch6_geometric_stability.png', dpi=150, bbox_inches='tight')
print("Saved: ch6_geometric_stability.png")
plt.show()


# Turning point analysis: close approach behavior
distances_close = np.linspace(50, 500, 100)
position_error = 150  # Use typical error from UAV localization

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Panel 1: Heading error near waypoint
ax1 = axes[0]
heading_errors_close = np.degrees(np.arctan(position_error / distances_close))
ax1.plot(distances_close, heading_errors_close, 'b-', lw=2)
ax1.fill_between(distances_close, 0, heading_errors_close, alpha=0.3)

# Mark critical thresholds
ax1.axhline(10, color='green', ls='--', lw=2, label='10° (stable)')
ax1.axhline(30, color='orange', ls='--', lw=2, label='30° (marginal)')
ax1.axhline(45, color='red', ls='--', lw=2, label='45° (unstable)')

# Mark recommended threshold
ax1.axvline(200, color='purple', ls='-', lw=2, alpha=0.7)
ax1.annotate('Recommended\nthreshold (200m)', (200, 50), fontsize=10, ha='center')

ax1.set_xlabel('Distance to Waypoint (m)', fontsize=11)
ax1.set_ylabel('Heading Error (degrees)', fontsize=11)
ax1.set_title(f'The Turning Point Problem\n(Position error = {position_error}m)', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim([50, 500])

# Panel 2: Probability of overshoot
ax2 = axes[1]
# At distance d with error e, probability that estimated distance < threshold when true distance > threshold
thresholds = [100, 150, 200, 250, 300]
true_distances = np.linspace(0, 500, 100)

for thresh in thresholds:
    # Simplified model: if true_dist > thresh but (true_dist - error) could be < thresh
    # This happens when position error pushes estimated position past the waypoint
    margin = true_distances - thresh
    # Probability that |error| > margin (when margin > 0)
    prob_premature = np.where(margin > 0, 
                              2 * (1 - 0.5 * (1 + np.clip(margin / position_error, -3, 3) / 3)),
                              1.0)
    prob_premature = np.clip(prob_premature, 0, 1)
    ax2.plot(true_distances, prob_premature * 100, lw=2, label=f'Threshold = {thresh}m')

ax2.set_xlabel('True Distance to Waypoint (m)', fontsize=11)
ax2.set_ylabel('Risk of Premature Arrival (%)', fontsize=11)
ax2.set_title('Waypoint Arrival Reliability\n(Higher threshold = lower risk)', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ch6_turning_point.png', dpi=150, bbox_inches='tight')
print("Saved: ch6_turning_point.png")
plt.show()


# Evaluate navigation on multiple patterns with different beacon errors
NAV_BEACON_ERRORS = [0, 100, 150, 200, 250]
NUM_TRIALS = 3

print("="*70)
print("MULTI-PATTERN NAVIGATION EVALUATION")
print("="*70)

navigation_results = {}

for pattern_name, waypoints in FLIGHT_PATTERNS.items():
    print(f"\nPattern: {pattern_name.upper()}")
    pattern_results = []
    
    for beacon_err in tqdm(NAV_BEACON_ERRORS, desc=f"  {pattern_name}"):
        trial_data = []
        
        for trial in range(NUM_TRIALS):
            nav = DroneNavigator(waypoints[0], waypoint_threshold=200)
            results = nav.fly_mission(waypoints[1:], beacon_mean_2d_error=beacon_err)
            
            trial_data.append({
                'wp_reached': sum(1 for r in results if r['reached']),
                'total_wp': len(results),
                'mean_error': np.mean(nav.errors),
                'median_error': np.median(nav.errors),
                'final_dists': [r['true_final_dist'] for r in results]
            })
        
        pattern_results.append({
            'beacon_error': beacon_err,
            'success_rate': np.mean([t['wp_reached']/t['total_wp']*100 for t in trial_data]),
            'mean_loc_error': np.mean([t['mean_error'] for t in trial_data]),
            'mean_final_dist': np.mean([np.mean(t['final_dists']) for t in trial_data])
        })
    
    navigation_results[pattern_name] = pattern_results

print("\n" + "="*70)


# Generate Figure: Multi-pattern navigation comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

colors = plt.cm.tab10(np.linspace(0, 1, len(FLIGHT_PATTERNS)))
pattern_names = list(FLIGHT_PATTERNS.keys())

# Panel 1: Success rate by pattern
ax1 = axes[0, 0]
for idx, name in enumerate(pattern_names):
    results = navigation_results[name]
    beacon_errs = [r['beacon_error'] for r in results]
    success = [r['success_rate'] for r in results]
    ax1.plot(beacon_errs, success, 'o-', lw=2, ms=8, color=colors[idx], label=name)

ax1.axvline(206, color='purple', ls='--', alpha=0.7, label='GCN (206m)')
ax1.set_xlabel('Beacon Position Error (mean 2D, m)', fontsize=11)
ax1.set_ylabel('Waypoint Success Rate (%)', fontsize=11)
ax1.set_title('Navigation Success by Pattern', fontsize=12, fontweight='bold')
ax1.legend(loc='lower left')
ax1.set_ylim([0, 110])
ax1.grid(True, alpha=0.3)

# Panel 2: Localization error during flight
ax2 = axes[0, 1]
for idx, name in enumerate(pattern_names):
    results = navigation_results[name]
    beacon_errs = [r['beacon_error'] for r in results]
    loc_errors = [r['mean_loc_error'] for r in results]
    ax2.plot(beacon_errs, loc_errors, 'o-', lw=2, ms=8, color=colors[idx], label=name)

ax2.axvline(206, color='purple', ls='--', alpha=0.7)
ax2.set_xlabel('Beacon Position Error (mean 2D, m)', fontsize=11)
ax2.set_ylabel('Mean UAV Localization Error (m)', fontsize=11)
ax2.set_title('Localization Error During Navigation', fontsize=12, fontweight='bold')
ax2.legend(loc='upper left')
ax2.grid(True, alpha=0.3)

# Panel 3: Final distance to waypoints
ax3 = axes[1, 0]
for idx, name in enumerate(pattern_names):
    results = navigation_results[name]
    beacon_errs = [r['beacon_error'] for r in results]
    final_dists = [r['mean_final_dist'] for r in results]
    ax3.plot(beacon_errs, final_dists, 'o-', lw=2, ms=8, color=colors[idx], label=name)

ax3.axhline(200, color='red', ls='--', lw=2, label='Threshold (200m)')
ax3.axvline(206, color='purple', ls='--', alpha=0.7)
ax3.set_xlabel('Beacon Position Error (mean 2D, m)', fontsize=11)
ax3.set_ylabel('Mean Final Distance to Waypoint (m)', fontsize=11)
ax3.set_title('Actual Waypoint Proximity', fontsize=12, fontweight='bold')
ax3.legend(loc='upper left')
ax3.grid(True, alpha=0.3)

# Panel 4: Summary at GCN error level (200m approximation)
ax4 = axes[1, 1]
ax4.axis('off')

# Find results at beacon_error = 200 (closest to GCN's 206m)
table_data = [['Pattern', 'Success\nRate', 'Mean\nLoc Error', 'Final WP\nDistance']]
for name in pattern_names:
    for r in navigation_results[name]:
        if r['beacon_error'] == 200:
            table_data.append([
                name.upper(),
                f"{r['success_rate']:.0f}%",
                f"{r['mean_loc_error']:.0f}m",
                f"{r['mean_final_dist']:.0f}m"
            ])
            break

table = ax4.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.3, 2.0)
for j in range(4):
    table[(0, j)].set_facecolor('#4472C4')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

ax4.set_title('Results at Beacon Error = 200m\n(Approximating GCN Performance)', 
              fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig('ch6_navigation_patterns.png', dpi=150, bbox_inches='tight')
print("Saved: ch6_navigation_patterns.png")
plt.show()


# Generate detailed flight visualization for circular pattern at GCN error level
pattern = 'circular'
waypoints = FLIGHT_PATTERNS[pattern]
beacon_err = 200  # Close to GCN's 206m

nav = DroneNavigator(waypoints[0], waypoint_threshold=200)
results = nav.fly_mission(waypoints[1:], beacon_mean_2d_error=beacon_err)

true_path = np.array(nav.true_path)
est_path = np.array(nav.est_path)
errors = np.array(nav.errors)
wp_array = np.array(waypoints)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# Panel 1: Flight path overview
ax1 = axes[0, 0]
ax1.scatter(beacon_x/1000, beacon_y/1000, c='green', s=50, marker='^', alpha=0.5, label='Beacons')
ax1.plot(true_path[:, 0]/1000, true_path[:, 1]/1000, 'b-', lw=2, label='True path')
ax1.plot(est_path[:, 0]/1000, est_path[:, 1]/1000, 'r--', lw=1.5, alpha=0.6, label='Estimated path')
for wx, wy in waypoints:
    circle = plt.Circle((wx/1000, wy/1000), 0.2, fill=False, color='orange', ls='--', alpha=0.6)
    ax1.add_patch(circle)
ax1.scatter(wp_array[:, 0]/1000, wp_array[:, 1]/1000, c='orange', s=200, marker='*', zorder=5, label='Waypoints')
ax1.set_xlabel('X (km)'); ax1.set_ylabel('Y (km)')
ax1.set_title(f'Flight Path: {pattern.upper()} (Beacon error = {beacon_err}m)', fontsize=12, fontweight='bold')
ax1.legend(loc='upper left')
ax1.set_aspect('equal'); ax1.grid(True, alpha=0.3)

# Panel 2: True vs Estimated positions
ax2 = axes[0, 1]
step = max(1, len(true_path) // 40)
for i in range(0, min(len(true_path)-1, len(est_path)), step):
    ax2.plot([true_path[i, 0]/1000, est_path[i, 0]/1000],
             [true_path[i, 1]/1000, est_path[i, 1]/1000], 'gray', lw=0.5, alpha=0.5)
ax2.scatter(true_path[::step, 0]/1000, true_path[::step, 1]/1000, c='blue', s=15, label='True')
ax2.scatter(est_path[::step, 0]/1000, est_path[::step, 1]/1000, c='red', s=15, label='Estimated')
ax2.scatter(wp_array[:, 0]/1000, wp_array[:, 1]/1000, c='orange', s=100, marker='*', zorder=5)
ax2.set_xlabel('X (km)'); ax2.set_ylabel('Y (km)')
ax2.set_title('True vs Estimated Positions', fontsize=12, fontweight='bold')
ax2.legend(); ax2.set_aspect('equal'); ax2.grid(True, alpha=0.3)

# Panel 3: Error over time
ax3 = axes[1, 0]
ax3.fill_between(range(len(errors)), 0, errors, alpha=0.3, color='blue')
ax3.plot(errors, 'b-', lw=1.5)
ax3.axhline(np.mean(errors), color='red', ls='--', lw=2, label=f'Mean: {np.mean(errors):.0f}m')
ax3.axhline(np.median(errors), color='green', ls='--', lw=2, label=f'Median: {np.median(errors):.0f}m')
ax3.set_xlabel('Time Step'); ax3.set_ylabel('Localization Error (m)')
ax3.set_title('Localization Error During Flight', fontsize=12, fontweight='bold')
ax3.legend(); ax3.grid(True, alpha=0.3)

# Panel 4: Error histogram
ax4 = axes[1, 1]
ax4.hist(errors, bins=30, edgecolor='black', alpha=0.7, color='steelblue', density=True)
ax4.axvline(np.mean(errors), color='red', ls='--', lw=2, label=f'Mean: {np.mean(errors):.0f}m')
ax4.axvline(np.median(errors), color='green', ls='--', lw=2, label=f'Median: {np.median(errors):.0f}m')
ax4.set_xlabel('Localization Error (m)'); ax4.set_ylabel('Density')
ax4.set_title('Error Distribution', fontsize=12, fontweight='bold')
ax4.legend(); ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ch6_detailed_flight.png', dpi=150, bbox_inches='tight')
print("Saved: ch6_detailed_flight.png")
plt.show()


# Create comprehensive summary
summary = {
    'experiment_config': {
        'num_beacons': int(num_beacons),
        'map_size_m': float(map_size_meters),
        'grid_size': int(drone_grid_size),
        'drone_altitude_m': float(drone_altitude),
        'localization_samples': 2000,
        'navigation_trials_per_condition': NUM_TRIALS,
        'waypoint_threshold_m': 200,
        'top_k_beacons': 10
    },
    'localization_results': [
        {
            'beacon_error_mean_2d_m': r['beacon_mean_2d_error'],
            'beacon_sigma_m': float(r['sigma']),
            'uav_mean_error_m': float(r['mean']),
            'uav_median_error_m': float(r['median']),
            'uav_p90_error_m': float(r['p90'])
        }
        for r in localization_results
    ],
    'navigation_results': {
        name: [
            {
                'beacon_error_m': r['beacon_error'],
                'success_rate_pct': float(r['success_rate']),
                'mean_loc_error_m': float(r['mean_loc_error']),
                'mean_final_dist_m': float(r['mean_final_dist'])
            }
            for r in results
        ]
        for name, results in navigation_results.items()
    },
    'key_findings': {
        'baseline_uav_error_m': float(localization_results[0]['mean']),
        'uav_error_at_gcn_level_m': float(next(r['mean'] for r in localization_results if r['beacon_mean_2d_error'] == 206)),
        'error_degradation_at_gcn_pct': float((next(r['mean'] for r in localization_results if r['beacon_mean_2d_error'] == 206) - localization_results[0]['mean']) / localization_results[0]['mean'] * 100),
    }
}

with open('ch6_results_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print("Results saved to ch6_results_summary.json")
print("\n" + "="*70)
print("KEY FINDINGS FOR CHAPTER 6")
print("="*70)
print(f"\nBaseline UAV localization error (perfect beacons): {summary['key_findings']['baseline_uav_error_m']:.1f}m")
print(f"UAV error at GCN beacon accuracy (206m): {summary['key_findings']['uav_error_at_gcn_level_m']:.1f}m")
print(f"Error degradation: +{summary['key_findings']['error_degradation_at_gcn_pct']:.1f}%")
print("\n" + "="*70)


print("\nFigures generated for thesis:")
print("  1. ch6_sensitivity_analysis.png - UAV error vs beacon uncertainty")
print("  2. ch6_geometric_stability.png - Heading error analysis")
print("  3. ch6_turning_point.png - Close-approach behavior")
print("  4. ch6_navigation_patterns.png - Multi-pattern comparison")
print("  5. ch6_detailed_flight.png - Detailed flight visualization")
print("\nAll figures saved to ./matlab/")
print("Copy to thesis/images/ for use in the paper.")
