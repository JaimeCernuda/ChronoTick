#!/usr/bin/env python3
"""
Plot ChronoTick vs System Clock accuracy from validation data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load data
csv_path = Path("results/fix_partial/homelab_validation.csv")
df = pd.read_csv(csv_path)

print("=" * 80)
print("CHRONOTICK FIX VALIDATION ANALYSIS")
print("=" * 80)
print(f"\nDataset: {csv_path}")
print(f"Total samples: {len(df)}")
print(f"Duration: {df['elapsed_seconds'].max() / 3600:.2f} hours")
print()

# Filter out error rows and rows without NTP
df_valid = df[df['chronotick_source'] != 'error'].copy()
df_ntp = df_valid[df_valid['has_ntp'] == True].copy()

print(f"Valid samples: {len(df_valid)}")
print(f"NTP ground truth samples: {len(df_ntp)}")
print()

# Calculate errors against NTP
# NTP offset is how much our system clock is off from true time
# ChronoTick offset is our correction to system clock
# Error = |ChronoTick correction - NTP offset|

df_ntp['chronotick_error'] = abs(df_ntp['chronotick_offset_ms'] - df_ntp['ntp_offset_ms'])
df_ntp['system_error'] = abs(df_ntp['ntp_offset_ms'])  # System clock error is just NTP offset

# Statistics
chronotick_mae = df_ntp['chronotick_error'].mean()
system_mae = df_ntp['system_error'].mean()
improvement_factor = system_mae / chronotick_mae if chronotick_mae > 0 else float('inf')

print("ACCURACY METRICS (vs NTP ground truth)")
print("-" * 80)
print(f"ChronoTick Mean Absolute Error: {chronotick_mae:.3f} ms")
print(f"System Clock Mean Absolute Error: {system_mae:.3f} ms")
print(f"Improvement Factor: {improvement_factor:.2f}x")
print()
print(f"ChronoTick Median Error: {df_ntp['chronotick_error'].median():.3f} ms")
print(f"System Clock Median Error: {df_ntp['system_error'].median():.3f} ms")
print()
print(f"ChronoTick 95th Percentile: {df_ntp['chronotick_error'].quantile(0.95):.3f} ms")
print(f"System Clock 95th Percentile: {df_ntp['system_error'].quantile(0.95):.3f} ms")
print()

# NTP acceptance analysis
ntp_count = len(df_ntp)
expected_ntp = int(df['elapsed_seconds'].max() / 120)  # 2-minute intervals
warmup_ntp = 27  # Expected during 60s warmup at 1Hz

print("NTP ACCEPTANCE ANALYSIS")
print("-" * 80)
print(f"NTP measurements received: {ntp_count}")
print(f"Expected (warmup + operational): ~{warmup_ntp + expected_ntp}")
print(f"Warmup NTP (60s @ 1Hz): {warmup_ntp} expected")
print(f"Operational NTP (2min intervals): {expected_ntp} expected")
post_warmup_ntp = max(0, ntp_count - warmup_ntp)
post_warmup_expected = expected_ntp
if post_warmup_expected > 0:
    acceptance_rate = (post_warmup_ntp / post_warmup_expected) * 100
    print(f"Post-warmup acceptance rate: {post_warmup_ntp}/{post_warmup_expected} ({acceptance_rate:.1f}%)")
print()

# Create visualizations
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('ChronoTick Fix Validation - Accuracy Analysis', fontsize=16, fontweight='bold')

# Plot 1: Time series of offsets
ax1 = axes[0, 0]
ax1.plot(df_valid['elapsed_seconds'] / 60, df_valid['chronotick_offset_ms'],
         'b-', alpha=0.3, linewidth=0.5, label='ChronoTick Offset')
if len(df_ntp) > 0:
    ax1.scatter(df_ntp['elapsed_seconds'] / 60, df_ntp['ntp_offset_ms'],
                c='red', s=50, alpha=0.7, marker='x', label='NTP Ground Truth', zorder=5)
ax1.set_xlabel('Time (minutes)', fontsize=12)
ax1.set_ylabel('Offset (ms)', fontsize=12)
ax1.set_title('Clock Offsets Over Time', fontsize=14, fontweight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

# Plot 2: Error comparison (box plot)
ax2 = axes[0, 1]
if len(df_ntp) > 0:
    box_data = [df_ntp['system_error'], df_ntp['chronotick_error']]
    bp = ax2.boxplot(box_data, labels=['System Clock', 'ChronoTick'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightblue')
    ax2.set_ylabel('Absolute Error (ms)', fontsize=12)
    ax2.set_title('Error Distribution (vs NTP)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add mean values as text
    ax2.text(1, system_mae, f'MAE: {system_mae:.2f}ms',
             ha='center', va='bottom', fontweight='bold', color='red')
    ax2.text(2, chronotick_mae, f'MAE: {chronotick_mae:.2f}ms',
             ha='center', va='bottom', fontweight='bold', color='blue')
else:
    ax2.text(0.5, 0.5, 'No NTP data available', ha='center', va='center',
             transform=ax2.transAxes, fontsize=14)
    ax2.set_title('Error Distribution (vs NTP)', fontsize=14, fontweight='bold')

# Plot 3: Error time series
ax3 = axes[1, 0]
if len(df_ntp) > 0:
    ax3.plot(df_ntp['elapsed_seconds'] / 60, df_ntp['system_error'],
             'r-', marker='o', markersize=4, label='System Clock Error', alpha=0.7)
    ax3.plot(df_ntp['elapsed_seconds'] / 60, df_ntp['chronotick_error'],
             'b-', marker='s', markersize=4, label='ChronoTick Error', alpha=0.7)
    ax3.set_xlabel('Time (minutes)', fontsize=12)
    ax3.set_ylabel('Absolute Error (ms)', fontsize=12)
    ax3.set_title('Prediction Accuracy Over Time', fontsize=14, fontweight='bold')
    ax3.legend(loc='best')
    ax3.grid(True, alpha=0.3)

    # Add improvement annotation
    ax3.text(0.02, 0.98, f'Improvement: {improvement_factor:.2f}x',
             transform=ax3.transAxes, fontsize=12, fontweight='bold',
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
else:
    ax3.text(0.5, 0.5, 'No NTP data available', ha='center', va='center',
             transform=ax3.transAxes, fontsize=14)
    ax3.set_title('Prediction Accuracy Over Time', fontsize=14, fontweight='bold')

# Plot 4: Prediction source breakdown
ax4 = axes[1, 1]
source_counts = df_valid['chronotick_source'].value_counts()
colors_map = {'cpu': 'skyblue', 'gpu': 'lightcoral', 'ntp_warm_up': 'lightgreen',
              'fusion': 'lightyellow', 'error': 'gray'}
colors = [colors_map.get(src, 'lightgray') for src in source_counts.index]
wedges, texts, autotexts = ax4.pie(source_counts.values, labels=source_counts.index,
                                     autopct='%1.1f%%', startangle=90, colors=colors)
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')
ax4.set_title('Prediction Source Distribution', fontsize=14, fontweight='bold')

plt.tight_layout()
plot_path = Path("results/fix_partial/accuracy_analysis.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"✓ Plot saved to: {plot_path}")
print()

# Additional detailed statistics
print("DETAILED STATISTICS")
print("-" * 80)
if len(df_ntp) > 0:
    print("\nChronoTick Error Distribution:")
    print(df_ntp['chronotick_error'].describe())
    print("\nSystem Clock Error Distribution:")
    print(df_ntp['system_error'].describe())
else:
    print("\nNo NTP ground truth data available for detailed statistics")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
