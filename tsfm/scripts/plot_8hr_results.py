#!/usr/bin/env python3
"""
Generate two key plots from 8-hour backtracking test:
1. System clock offset vs ChronoTick offset comparison
2. ChronoTick offset with TimesFM uncertainty bands
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load the CSVs
base_path = Path("results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED")
summary_csv = base_path / "summary_backtracking_20251014_010440.csv"
client_csv = base_path / "client_predictions_backtracking_20251014_010440.csv"

df_summary = pd.read_csv(summary_csv)
df_client = pd.read_csv(client_csv)

# Filter summary to only rows with NTP ground truth
df_with_ntp = df_summary[df_summary['has_ntp'] == True].copy()

# Merge client predictions with NTP ground truth data
df_merged = pd.merge(df_client, df_with_ntp[['timestamp', 'ntp_ground_truth_offset_ms', 'elapsed_seconds']],
                     on='timestamp', how='inner')

print(f"Total summary rows: {len(df_summary)}")
print(f"Rows with NTP ground truth: {len(df_with_ntp)}")
print(f"Merged rows (client + NTP): {len(df_merged)}")

# Calculate ChronoTick error for merged data
df_merged['chronotick_error_ms'] = abs(df_merged['offset_correction_ms'] - df_merged['ntp_ground_truth_offset_ms'])

# Calculate system clock error (system time has no correction, so error = abs(ntp_ground_truth))
df_merged['system_error_ms'] = abs(df_merged['ntp_ground_truth_offset_ms'])

# Convert elapsed seconds to hours
df_merged['elapsed_hours'] = df_merged['elapsed_seconds'] / 3600
df_with_ntp['elapsed_hours'] = df_with_ntp['elapsed_seconds'] / 3600

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# ============================================================================
# Plot 1: System Clock Offset vs ChronoTick Offset (same as before)
# ============================================================================

# Plot system clock error
ax1.plot(df_with_ntp['elapsed_hours'],
         df_with_ntp['system_error_ms'],
         'o-', color='red', alpha=0.6, linewidth=1.5, markersize=4,
         label='System Clock Error')

# Plot ChronoTick error
ax1.plot(df_with_ntp['elapsed_hours'],
         df_with_ntp['chronotick_error_ms'],
         's-', color='blue', alpha=0.6, linewidth=1.5, markersize=4,
         label='ChronoTick Error')

# Add zero reference line
ax1.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.3)

ax1.set_xlabel('Time (hours)', fontsize=12)
ax1.set_ylabel('Offset Error (ms)', fontsize=12)
ax1.set_title('System Clock vs ChronoTick Offset Error (8-hour test)', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=10)
ax1.grid(True, alpha=0.3)

# Add stats box
system_mean = df_with_ntp['system_error_ms'].mean()
system_std = df_with_ntp['system_error_ms'].std()
chronotick_mean = df_with_ntp['chronotick_error_ms'].mean()
chronotick_std = df_with_ntp['chronotick_error_ms'].std()

stats_text = f'System Clock: μ={system_mean:.2f}ms, σ={system_std:.2f}ms\n'
stats_text += f'ChronoTick: μ={chronotick_mean:.2f}ms, σ={chronotick_std:.2f}ms'
ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         fontsize=9)

# ============================================================================
# Plot 2: ChronoTick Offset with TimesFM Uncertainty Bands
# ============================================================================

# Plot system clock error (added per user request)
ax2.plot(df_merged['elapsed_hours'],
         df_merged['system_error_ms'],
         'o-', color='red', alpha=0.6, linewidth=1.5, markersize=4,
         label='System Clock Error', zorder=4)

# Plot ChronoTick error as line
ax2.plot(df_merged['elapsed_hours'],
         df_merged['chronotick_error_ms'],
         's-', color='blue', alpha=0.8, linewidth=2, markersize=4,
         label='ChronoTick Error', zorder=3)

# Add shaded uncertainty bands (±1σ from TimesFM quantiles) - darker color per user request
uncertainty = df_merged['offset_uncertainty_ms']
lower_bound = df_merged['chronotick_error_ms'] - uncertainty
upper_bound = df_merged['chronotick_error_ms'] + uncertainty

ax2.fill_between(df_merged['elapsed_hours'],
                  lower_bound,
                  upper_bound,
                  color='steelblue', alpha=0.5,
                  label='ChronoTick Uncertainty (±1σ from TimesFM quantiles)',
                  zorder=1)

# Add zero reference line
ax2.axhline(y=0, color='black', linestyle='--', linewidth=0.8, alpha=0.3, zorder=2)

ax2.set_xlabel('Time (hours)', fontsize=12)
ax2.set_ylabel('Offset Error (ms)', fontsize=12)
ax2.set_title('ChronoTick Offset with TimesFM Uncertainty Bands', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3, zorder=0)

# Add stats box
mean_uncertainty = df_merged['offset_uncertainty_ms'].mean()
min_uncertainty = df_merged['offset_uncertainty_ms'].min()
max_uncertainty = df_merged['offset_uncertainty_ms'].max()

stats_text = f'Mean ChronoTick Error: {df_merged["chronotick_error_ms"].mean():.2f}ms\n'
stats_text += f'Mean Uncertainty: {mean_uncertainty:.2f}ms\n'
stats_text += f'Uncertainty Range: [{min_uncertainty:.2f}, {max_uncertainty:.2f}]ms'
ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7),
         fontsize=9)

# Add note about limited corrections
note_text = f'Note: Only {len(correction_times)} NTP corrections due to network quality filtering'
ax2.text(0.02, 0.02, note_text, transform=ax2.transAxes,
         verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5),
         fontsize=9, style='italic')

plt.tight_layout()

# Save the plot
output_path = Path("results/ntp_correction_experiment/overnight_8hr_backtracking_ENHANCED_plots.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Plot saved to: {output_path}")

# Show summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"\nSystem Clock Error:")
print(f"  Mean: {system_mean:.3f} ms")
print(f"  Std:  {system_std:.3f} ms")
print(f"  Min:  {df_with_ntp['system_error_ms'].min():.3f} ms")
print(f"  Max:  {df_with_ntp['system_error_ms'].max():.3f} ms")

print(f"\nChronoTick Error:")
print(f"  Mean: {chronotick_mean:.3f} ms")
print(f"  Std:  {chronotick_std:.3f} ms")
print(f"  Min:  {df_with_ntp['chronotick_error_ms'].min():.3f} ms")
print(f"  Max:  {df_with_ntp['chronotick_error_ms'].max():.3f} ms")

print(f"\nTimesFM Uncertainty (from quantiles):")
print(f"  Mean: {mean_uncertainty:.3f} ms")
print(f"  Min:  {min_uncertainty:.3f} ms")
print(f"  Max:  {max_uncertainty:.3f} ms")

improvement = ((system_mean - chronotick_mean) / system_mean) * 100
print(f"\n📊 ChronoTick mean error improvement: {improvement:.1f}%")
print(f"📊 ChronoTick uncertainty quantification: ±{mean_uncertainty:.1f}ms average")

plt.show()
