#!/usr/bin/env python3
"""
Plot detailed comparison of NTP correction methods showing:
1. ChronoTick predictions vs NTP ground truth over time
2. How well each method tracks real NTP measurements
3. Error evolution over time
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Find most recent test results
results_dir = Path('results/ntp_correction_experiment')
csv_files = {}
for method in ['none', 'linear', 'drift_aware', 'advanced']:
    pattern = f"ntp_correction_{method}_test_*.csv"
    files = sorted(results_dir.glob(pattern))
    if files:
        csv_files[method] = files[-1]
        print(f"Found {method}: {csv_files[method].name}")

# Load data
data = {}
for method, csv_file in csv_files.items():
    df = pd.read_csv(csv_file)
    data[method] = df
    print(f"  {method}: {len(df)} samples, {df['has_ntp'].sum()} with NTP ground truth")

print(f"\n{'='*80}")
print("DETAILED CORRECTION ANALYSIS")
print(f"{'='*80}\n")

# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('NTP Correction Methods: Detailed Comparison', fontsize=14, fontweight='bold')

colors = {'none': 'blue', 'linear': 'green', 'drift_aware': 'orange', 'advanced': 'red'}

for idx, method in enumerate(['none', 'linear', 'drift_aware', 'advanced']):
    if method not in data:
        continue

    row = idx // 2
    col = idx % 2
    ax = axes[row, col]

    df = data[method]

    # Plot 1: ChronoTick offset vs NTP ground truth over time
    ax.plot(df['elapsed_seconds'], df['chronotick_offset_ms'],
           color=colors[method], linewidth=2, label='ChronoTick prediction', alpha=0.8)

    # Overlay NTP ground truth measurements
    df_with_ntp = df[df['has_ntp'] == True].copy()
    if len(df_with_ntp) > 0:
        ax.scatter(df_with_ntp['elapsed_seconds'], df_with_ntp['ntp_ground_truth_offset_ms'],
                  color='red', s=100, marker='x', linewidth=3,
                  label='NTP ground truth', zorder=10)

        # Show error bars for NTP uncertainty
        ax.errorbar(df_with_ntp['elapsed_seconds'], df_with_ntp['ntp_ground_truth_offset_ms'],
                   yerr=df_with_ntp['ntp_ground_truth_uncertainty_ms'],
                   fmt='none', ecolor='red', alpha=0.3, capsize=5)

    # Calculate MAE for this method
    mae = df_with_ntp['chronotick_error_ms'].mean() if len(df_with_ntp) > 0 else float('nan')

    ax.set_xlabel('Time (seconds)', fontsize=10)
    ax.set_ylabel('Offset (ms)', fontsize=10)
    ax.set_title(f'{method.upper()}: MAE={mae:.2f}ms, {len(df_with_ntp)} NTP samples',
                fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()

output_dir = Path('results/ntp_correction_experiment/analysis')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'detailed_correction_comparison.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved detailed comparison: {output_path}")
plt.close()

# Create error evolution plot
fig, ax = plt.subplots(1, 1, figsize=(14, 6))
fig.suptitle('Prediction Error vs NTP Ground Truth Over Time', fontsize=14, fontweight='bold')

for method in ['none', 'linear', 'drift_aware', 'advanced']:
    if method not in data:
        continue

    df = data[method]
    df_with_ntp = df[df['has_ntp'] == True].copy()

    if len(df_with_ntp) > 0:
        ax.plot(df_with_ntp['elapsed_seconds'], df_with_ntp['chronotick_error_ms'],
               color=colors[method], linewidth=2, marker='o', markersize=8,
               label=f'{method.upper()} (MAE={df_with_ntp["chronotick_error_ms"].mean():.2f}ms)',
               alpha=0.8)

ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.set_xlabel('Time (seconds)', fontsize=11)
ax.set_ylabel('Error vs NTP (ms)', fontsize=11)
ax.set_title('Lower is better - shows how close ChronoTick predictions are to actual NTP',
            fontsize=10, style='italic')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3)

output_path = output_dir / 'error_evolution.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved error evolution: {output_path}")
plt.close()

# Print detailed statistics
print(f"\n{'='*80}")
print("DETAILED STATISTICS")
print(f"{'='*80}\n")

for method in ['none', 'linear', 'drift_aware', 'advanced']:
    if method not in data:
        continue

    df = data[method]
    df_with_ntp = df[df['has_ntp'] == True].copy()

    if len(df_with_ntp) == 0:
        continue

    print(f"{method.upper()}:")
    print(f"  NTP measurements collected: {len(df_with_ntp)}")
    print(f"  ChronoTick MAE: {df_with_ntp['chronotick_error_ms'].mean():.3f}ms")
    print(f"  ChronoTick StdDev: {df_with_ntp['chronotick_error_ms'].std():.3f}ms")
    print(f"  ChronoTick Max Error: {df_with_ntp['chronotick_error_ms'].max():.3f}ms")
    print(f"  Average ChronoTick offset: {df['chronotick_offset_ms'].mean():.2f}ms")
    print(f"  Average NTP ground truth: {df_with_ntp['ntp_ground_truth_offset_ms'].mean():.2f}ms")
    print(f"  Offset drift over time: {df['chronotick_offset_ms'].iloc[-1] - df['chronotick_offset_ms'].iloc[0]:.2f}ms")
    print()

print("="*80)
print("Analysis complete!")
