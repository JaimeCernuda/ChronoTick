#!/usr/bin/env python3
"""
Plot ChronoTick offset stability for ARES nodes (without NTP ground truth comparison)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

print("=" * 80)
print("ARES CHRONOTICK OFFSET STABILITY ANALYSIS")
print("=" * 80)
print()

# Load data for both ARES nodes
csv_path_11 = Path("results/fix_partial/ares_comp11_validation.csv")
csv_path_12 = Path("results/fix_partial/ares_comp12_validation.csv")

df11 = pd.read_csv(csv_path_11)
df12 = pd.read_csv(csv_path_12)

print(f"ARES-COMP-11:")
print(f"  Total samples: {len(df11)}")
print(f"  Duration: {df11['elapsed_seconds'].max() / 3600:.2f} hours")

print(f"\nARES-COMP-12:")
print(f"  Total samples: {len(df12)}")
print(f"  Duration: {df12['elapsed_seconds'].max() / 3600:.2f} hours")
print()

# Process both datasets
results = {}
for name, df in [("ares-comp-11", df11), ("ares-comp-12", df12)]:
    print(f"\n{'=' * 80}")
    print(f"ANALYSIS: {name.upper()}")
    print(f"{'=' * 80}\n")

    # Filter out error rows
    df_valid = df[df['chronotick_source'] != 'error'].copy()

    print(f"Valid samples: {len(df_valid)}")

    # Offset statistics
    offset_mean = df_valid['chronotick_offset_ms'].mean()
    offset_std = df_valid['chronotick_offset_ms'].std()
    offset_min = df_valid['chronotick_offset_ms'].min()
    offset_max = df_valid['chronotick_offset_ms'].max()
    offset_range = offset_max - offset_min

    print(f"\nOFFSET STATISTICS:")
    print("-" * 80)
    print(f"  Mean:   {offset_mean:.3f} ms")
    print(f"  Std Dev: {offset_std:.3f} ms")
    print(f"  Range:   {offset_min:.3f} to {offset_max:.3f} ms (span: {offset_range:.3f} ms)")
    print(f"  Median:  {df_valid['chronotick_offset_ms'].median():.3f} ms")

    # Uncertainty statistics
    uncertainty_mean = df_valid['chronotick_uncertainty_ms'].mean()
    uncertainty_std = df_valid['chronotick_uncertainty_ms'].std()

    print(f"\nUNCERTAINTY STATISTICS:")
    print("-" * 80)
    print(f"  Mean:   {uncertainty_mean:.3f} ms")
    print(f"  Std Dev: {uncertainty_std:.3f} ms")

    # Confidence statistics
    confidence_mean = df_valid['chronotick_confidence'].mean()

    print(f"\nCONFIDENCE:")
    print("-" * 80)
    print(f"  Mean:   {confidence_mean:.3f}")

    # Source distribution
    source_counts = df_valid['chronotick_source'].value_counts()
    print(f"\nPREDICTION SOURCES:")
    print("-" * 80)
    for source, count in source_counts.items():
        pct = (count / len(df_valid)) * 100
        print(f"  {source:20s}: {count:5d} ({pct:5.1f}%)")

    # Time-based drift analysis
    early = df_valid[df_valid['elapsed_seconds'] < 600]
    middle = df_valid[(df_valid['elapsed_seconds'] >= 3600) & (df_valid['elapsed_seconds'] < 7200)]
    late = df_valid[df_valid['elapsed_seconds'] >= 14400]

    print(f"\nDRIFT ANALYSIS (Offset Mean):")
    print("-" * 80)
    if len(early) > 0:
        print(f"  Early (<10min):  {early['chronotick_offset_ms'].mean():.3f} ms")
    if len(middle) > 0:
        print(f"  Middle (1-2hr):  {middle['chronotick_offset_ms'].mean():.3f} ms")
    if len(late) > 0:
        print(f"  Late (>4hr):     {late['chronotick_offset_ms'].mean():.3f} ms")

    # Estimate clock stability
    if len(df_valid) > 10:
        time_hours = df_valid['elapsed_seconds'].max() / 3600
        drift_estimate = offset_max - offset_min
        drift_rate = (drift_estimate / time_hours) if time_hours > 0 else 0
        print(f"\nESTIMATED CLOCK STABILITY:")
        print("-" * 80)
        print(f"  Total drift: {drift_estimate:.3f} ms over {time_hours:.2f} hours")
        print(f"  Drift rate:  {drift_rate:.3f} ms/hour")
        print(f"  Classification: {'STABLE' if drift_rate < 1.0 else 'MODERATE' if drift_rate < 5.0 else 'UNSTABLE'}")

    results[name] = {
        'df': df,
        'df_valid': df_valid,
        'offset_mean': offset_mean,
        'offset_std': offset_std,
        'offset_range': offset_range,
        'uncertainty_mean': uncertainty_mean,
        'confidence_mean': confidence_mean
    }

# Create visualization
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

fig.suptitle('ARES Nodes - ChronoTick Offset Stability (5 hours)', fontsize=16, fontweight='bold')

# ares-comp-11 plots
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])

# ares-comp-12 plots
ax4 = fig.add_subplot(gs[0, 1])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[2, 1])

# Plot ares-comp-11
r11 = results['ares-comp-11']
df11_valid = r11['df_valid']

# Offset time series
ax1.plot(df11_valid['elapsed_seconds'] / 60, df11_valid['chronotick_offset_ms'],
         'b-', alpha=0.6, linewidth=0.8)
ax1.set_xlabel('Time (minutes)', fontsize=11)
ax1.set_ylabel('Offset (ms)', fontsize=11)
ax1.set_title(f'ares-comp-11: ChronoTick Offset\n(Mean: {r11["offset_mean"]:.3f} ms, StdDev: {r11["offset_std"]:.3f} ms)',
              fontweight='bold', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.axhline(y=r11['offset_mean'], color='r', linestyle='--', linewidth=1, label=f'Mean: {r11["offset_mean"]:.3f} ms')
ax1.fill_between(df11_valid['elapsed_seconds'] / 60,
                  r11['offset_mean'] - r11['offset_std'],
                  r11['offset_mean'] + r11['offset_std'],
                  alpha=0.2, color='red', label='±1σ')
ax1.legend(loc='best', fontsize=9)

# Offset distribution
ax2.hist(df11_valid['chronotick_offset_ms'], bins=50, alpha=0.7, color='steelblue', edgecolor='black')
ax2.axvline(r11['offset_mean'], color='r', linestyle='--', linewidth=2, label=f'Mean: {r11["offset_mean"]:.3f} ms')
ax2.set_xlabel('Offset (ms)', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('ares-comp-11: Offset Distribution', fontweight='bold', fontsize=12)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

# Prediction sources
source_counts_11 = df11_valid['chronotick_source'].value_counts()
colors_map = {'cpu': 'skyblue', 'gpu': 'lightcoral', 'ntp_warm_up': 'lightgreen',
              'fusion': 'lightyellow'}
colors = [colors_map.get(src, 'lightgray') for src in source_counts_11.index]
wedges, texts, autotexts = ax3.pie(source_counts_11.values, labels=source_counts_11.index,
                                     autopct='%1.1f%%', startangle=90, colors=colors)
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')
ax3.set_title('ares-comp-11: Prediction Sources', fontweight='bold', fontsize=12)

# Plot ares-comp-12
r12 = results['ares-comp-12']
df12_valid = r12['df_valid']

# Offset time series
ax4.plot(df12_valid['elapsed_seconds'] / 60, df12_valid['chronotick_offset_ms'],
         'darkorange', alpha=0.6, linewidth=0.8)
ax4.set_xlabel('Time (minutes)', fontsize=11)
ax4.set_ylabel('Offset (ms)', fontsize=11)
ax4.set_title(f'ares-comp-12: ChronoTick Offset\n(Mean: {r12["offset_mean"]:.3f} ms, StdDev: {r12["offset_std"]:.3f} ms)',
              fontweight='bold', fontsize=12)
ax4.grid(True, alpha=0.3)
ax4.axhline(y=r12['offset_mean'], color='r', linestyle='--', linewidth=1, label=f'Mean: {r12["offset_mean"]:.3f} ms')
ax4.fill_between(df12_valid['elapsed_seconds'] / 60,
                  r12['offset_mean'] - r12['offset_std'],
                  r12['offset_mean'] + r12['offset_std'],
                  alpha=0.2, color='red', label='±1σ')
ax4.legend(loc='best', fontsize=9)

# Offset distribution
ax5.hist(df12_valid['chronotick_offset_ms'], bins=50, alpha=0.7, color='darkorange', edgecolor='black')
ax5.axvline(r12['offset_mean'], color='r', linestyle='--', linewidth=2, label=f'Mean: {r12["offset_mean"]:.3f} ms')
ax5.set_xlabel('Offset (ms)', fontsize=11)
ax5.set_ylabel('Frequency', fontsize=11)
ax5.set_title('ares-comp-12: Offset Distribution', fontweight='bold', fontsize=12)
ax5.legend(fontsize=9)
ax5.grid(True, alpha=0.3, axis='y')

# Prediction sources
source_counts_12 = df12_valid['chronotick_source'].value_counts()
colors = [colors_map.get(src, 'lightgray') for src in source_counts_12.index]
wedges, texts, autotexts = ax6.pie(source_counts_12.values, labels=source_counts_12.index,
                                     autopct='%1.1f%%', startangle=90, colors=colors)
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_fontweight('bold')
ax6.set_title('ares-comp-12: Prediction Sources', fontweight='bold', fontsize=12)

plot_path = Path("results/fix_partial/ares_stability_analysis.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print()
print(f"\n✓ Plot saved to: {plot_path}")

# Comparison summary
print()
print("=" * 80)
print("COMPARATIVE SUMMARY")
print("=" * 80)
print(f"\n{'Metric':<30s} {'ares-comp-11':>15s} {'ares-comp-12':>15s}")
print("-" * 80)
print(f"{'Mean Offset':<30s} {r11['offset_mean']:>14.3f}ms {r12['offset_mean']:>14.3f}ms")
print(f"{'Offset Std Dev':<30s} {r11['offset_std']:>14.3f}ms {r12['offset_std']:>14.3f}ms")
print(f"{'Offset Range':<30s} {r11['offset_range']:>14.3f}ms {r12['offset_range']:>14.3f}ms")
print(f"{'Mean Uncertainty':<30s} {r11['uncertainty_mean']:>14.3f}ms {r12['uncertainty_mean']:>14.3f}ms")
print(f"{'Mean Confidence':<30s} {r11['confidence_mean']:>15.3f} {r12['confidence_mean']:>15.3f}")

print()
print("=" * 80)
print("ARES ANALYSIS COMPLETE")
print("=" * 80)
print()
print("NOTE: No NTP ground truth available - client cannot reach external NTP servers.")
print("ChronoTick itself is using NTP proxy successfully (visible in logs).")
print("Rerun with --ntp-server 172.20.1.1:8123 to get ground truth comparison.")
