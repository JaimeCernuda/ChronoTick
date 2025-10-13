#!/usr/bin/env python3
"""
Analyze and compare all four NTP correction methods.
Calculate metrics and create visualizations based on prediction stability.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 16)

# Load all four datasets
results_dir = Path("results/ntp_correction_experiment")
methods = ["none", "linear", "drift_aware", "advanced"]
data = {}

for method in methods:
    csv_path = results_dir / f"ntp_correction_{method}.csv"
    df = pd.read_csv(csv_path)
    data[method] = df
    print(f"Loaded {method}: {len(df)} samples")

# Calculate metrics for each method
print("\n=== PREDICTION STABILITY ANALYSIS ===")
results = {}

for method in methods:
    df = data[method]

    # Analyze prediction stability
    offset_mean = df['chronotick_offset_ms'].mean()
    offset_std = df['chronotick_offset_ms'].std()
    offset_range = df['chronotick_offset_ms'].max() - df['chronotick_offset_ms'].min()

    drift_mean = df['chronotick_drift_us_per_s'].mean()
    drift_std = df['chronotick_drift_us_per_s'].std()
    drift_range = df['chronotick_drift_us_per_s'].max() - df['chronotick_drift_us_per_s'].min()

    # Stability score (lower is better): weighted combination of std and range
    stability_score = (offset_std * 0.5 + offset_range * 0.5)

    results[method] = {
        'offset_mean': offset_mean,
        'offset_std': offset_std,
        'offset_range': offset_range,
        'drift_mean': drift_mean,
        'drift_std': drift_std,
        'drift_range': drift_range,
        'stability_score': stability_score
    }

    print(f"\n{method.upper()}:")
    print(f"  Offset: mean={offset_mean:.2f}ms, std={offset_std:.2f}ms, range={offset_range:.2f}ms")
    print(f"  Drift: mean={drift_mean:.2f}μs/s, std={drift_std:.2f}μs/s, range={drift_range:.2f}μs/s")
    print(f"  Stability Score: {stability_score:.2f} (lower is better)")

# Create comprehensive comparison plots
fig = plt.figure(figsize=(20, 16))

# Plot 1: Chronotick offset predictions over time
ax1 = plt.subplot(4, 2, 1)
for method in methods:
    df = data[method]
    ax1.plot(df['elapsed_seconds'], df['chronotick_offset_ms'],
             label=method, alpha=0.7, linewidth=1.5)

ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('ChronoTick Offset (ms)')
ax1.set_title('ChronoTick Offset Predictions Over Time')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Chronotick drift predictions over time
ax2 = plt.subplot(4, 2, 2)
for method in methods:
    df = data[method]
    ax2.plot(df['elapsed_seconds'], df['chronotick_drift_us_per_s'],
             label=method, alpha=0.7, linewidth=1.5)

ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('ChronoTick Drift (μs/s)')
ax2.set_title('ChronoTick Drift Predictions Over Time')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Offset stability comparison (box plots)
ax3 = plt.subplot(4, 2, 3)
offset_data = [data[m]['chronotick_offset_ms'].values for m in methods]
bp = ax3.boxplot(offset_data, labels=methods, patch_artist=True, showmeans=True)
colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax3.set_ylabel('Offset (ms)')
ax3.set_title('Offset Distribution Comparison')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Drift stability comparison (box plots)
ax4 = plt.subplot(4, 2, 4)
drift_data = [data[m]['chronotick_drift_us_per_s'].values for m in methods]
bp = ax4.boxplot(drift_data, labels=methods, patch_artist=True, showmeans=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

ax4.set_ylabel('Drift (μs/s)')
ax4.set_title('Drift Distribution Comparison')
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: Stability score comparison
ax5 = plt.subplot(4, 2, 5)
stability_scores = [results[m]['stability_score'] for m in methods]
bars = ax5.bar(methods, stability_scores, color=colors)
ax5.set_ylabel('Stability Score (lower is better)')
ax5.set_title('Overall Stability Comparison')
ax5.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, val in zip(bars, stability_scores):
    height = bar.get_height()
    ax5.text(bar.get_x() + bar.get_width()/2., height,
            f'{val:.1f}', ha='center', va='bottom', fontweight='bold')

# Plot 6: Standard deviation comparison
ax6 = plt.subplot(4, 2, 6)
x = np.arange(len(methods))
width = 0.35
offset_stds = [results[m]['offset_std'] for m in methods]
drift_stds = [r['drift_std']/1000 for r in [results[m] for m in methods]]  # Scale drift for visibility

ax6.bar(x - width/2, offset_stds, width, label='Offset Std (ms)', color='skyblue')
ax6.bar(x + width/2, drift_stds, width, label='Drift Std (ms equiv)', color='lightcoral')
ax6.set_ylabel('Standard Deviation')
ax6.set_title('Prediction Variability (Standard Deviation)')
ax6.set_xticks(x)
ax6.set_xticklabels(methods)
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

# Plot 7: Method comparison - zoomed in on last 5 minutes
ax7 = plt.subplot(4, 2, 7)
for method in methods:
    df = data[method]
    mask = df['elapsed_seconds'] >= 1200  # Last 5 minutes
    ax7.plot(df[mask]['elapsed_seconds'], df[mask]['chronotick_offset_ms'],
             label=method, alpha=0.7, linewidth=2)

ax7.set_xlabel('Time (seconds)')
ax7.set_ylabel('ChronoTick Offset (ms)')
ax7.set_title('Offset Predictions - Last 5 Minutes (Zoomed)')
ax7.legend()
ax7.grid(True, alpha=0.3)

# Plot 8: Summary metrics table
ax8 = plt.subplot(4, 2, 8)
ax8.axis('off')

# Create summary table
summary_data = []
for method in methods:
    r = results[method]
    summary_data.append([
        method.upper(),
        f"{r['offset_mean']:.2f}",
        f"{r['offset_std']:.2f}",
        f"{r['offset_range']:.2f}",
        f"{r['stability_score']:.1f}"
    ])

table = ax8.table(cellText=summary_data,
                  colLabels=['Method', 'Offset Mean\n(ms)', 'Offset Std\n(ms)',
                            'Offset Range\n(ms)', 'Stability\nScore'],
                  cellLoc='center',
                  loc='center',
                  colWidths=[0.2, 0.2, 0.2, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Color code the best values (lowest offset std and stability score are best)
for col in [2, 4]:  # Offset std and stability score
    values = [float(row[col]) for row in summary_data]
    best_idx = np.argmin(values)
    table[(best_idx + 1, col)].set_facecolor('lightgreen')

ax8.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(results_dir / 'ntp_correction_comparison_full.png', dpi=150, bbox_inches='tight')
print(f"\n✅ Saved comparison plot to {results_dir / 'ntp_correction_comparison_full.png'}")

# Determine best method (lowest stability score)
print("\n=== RECOMMENDATION ===")
best_method = min(methods, key=lambda m: results[m]['stability_score'])
print(f"🏆 BEST METHOD: {best_method.upper()}")
print(f"   Offset Std Dev: {results[best_method]['offset_std']:.2f}ms")
print(f"   Stability Score: {results[best_method]['stability_score']:.2f}")
print(f"\n   This method shows the most stable predictions with lowest variability.")

# Save recommendation
with open(results_dir / 'best_method.txt', 'w') as f:
    f.write(f"{best_method}\n")
print(f"\n✅ Saved best method to {results_dir / 'best_method.txt'}")

print("\n=== ANALYSIS COMPLETE ===")
