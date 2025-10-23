#!/usr/bin/env python3
"""
Create improved plots addressing the questions
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

data_dir = Path(__file__).parent

# Load data
homelab_csv = pd.read_csv(data_dir / 'homelab' / 'chronotick_client_validation_20251022_094657.csv')
ares11_csv = pd.read_csv(data_dir / 'ares-11' / 'chronotick_client_validation_20251022_105514.csv')
ares12_csv = pd.read_csv(data_dir / 'ares-12' / 'chronotick_client_validation_20251022_105732.csv')

# Filter ARES-12 outliers
ares12_ntp = ares12_csv[ares12_csv['has_ntp'] == True].copy()
ares12_ntp_filtered = ares12_ntp[np.abs(ares12_ntp['ntp_offset_ms']) < 20].copy()

colors = {'homelab': '#1f77b4', 'ares-11': '#ff7f0e', 'ares-12': '#2ca02c'}

# ============================================================================
# FIGURE 1: IMPROVED INDIVIDUAL PLOTS
# ============================================================================
fig, axes = plt.subplots(3, 3, figsize=(20, 14))
fig.suptitle('ChronoTick Validation - Improved Analysis', fontsize=16, fontweight='bold')

# Homelab - with spike region highlighted
ax = axes[0, 0]
ax.plot(homelab_csv['elapsed_seconds'] / 60, homelab_csv['chronotick_offset_ms'],
        label='ChronoTick', alpha=0.7, linewidth=0.8, color=colors['homelab'])
ntp_data = homelab_csv[homelab_csv['has_ntp'] == True]
ax.scatter(ntp_data['elapsed_seconds'] / 60, ntp_data['ntp_offset_ms'],
          label='NTP', color='red', alpha=0.6, s=20, marker='x')
# Highlight spike region (57-59 minutes)
ax.axvspan(57, 59, alpha=0.2, color='red', label='Spike region')
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Offset (ms)')
ax.set_title('HOMELAB: Offset Over Time (spike at 57-59 min)')
ax.legend()
ax.grid(True, alpha=0.3)

# ARES-11 - excellent performance
ax = axes[0, 1]
ax.plot(ares11_csv['elapsed_seconds'] / 60, ares11_csv['chronotick_offset_ms'],
        label='ChronoTick', alpha=0.7, linewidth=0.8, color=colors['ares-11'])
ntp_data = ares11_csv[ares11_csv['has_ntp'] == True]
ax.scatter(ntp_data['elapsed_seconds'] / 60, ntp_data['ntp_offset_ms'],
          label='NTP', color='red', alpha=0.6, s=20, marker='x')
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Offset (ms)')
ax.set_title('ARES-11: Excellent Stability ✓')
ax.legend()
ax.grid(True, alpha=0.3)

# ARES-12 - filtered outliers
ax = axes[0, 2]
ax.plot(ares12_csv['elapsed_seconds'] / 60, ares12_csv['chronotick_offset_ms'],
        label='ChronoTick', alpha=0.7, linewidth=0.8, color=colors['ares-12'])
# Plot filtered NTP
ax.scatter(ares12_ntp_filtered['elapsed_seconds'] / 60, ares12_ntp_filtered['ntp_offset_ms'],
          label='NTP (filtered)', color='red', alpha=0.6, s=20, marker='x')
# Mark outliers
outliers = ares12_ntp[np.abs(ares12_ntp['ntp_offset_ms']) >= 20]
if len(outliers) > 0:
    ax.scatter(outliers['elapsed_seconds'] / 60, [15]*len(outliers),
              label=f'Outliers (n={len(outliers)})', color='darkred', alpha=0.8, s=100, marker='v')
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('Offset (ms)')
ax.set_title('ARES-12: NTP Outliers Filtered (2 @ 581ms, 630ms)')
ax.legend()
ax.grid(True, alpha=0.3)

# Row 2: Fusion patterns over time
for idx, (name, df, color) in enumerate([('Homelab', homelab_csv, colors['homelab']),
                                           ('ARES-11', ares11_csv, colors['ares-11']),
                                           ('ARES-12', ares12_csv, colors['ares-12'])]):
    ax = axes[1, idx]

    # Create binary indicator: 1=fusion, 0=cpu
    fusion_indicator = (df['chronotick_source'] == 'fusion').astype(int)

    ax.scatter(df['elapsed_seconds'] / 60, fusion_indicator,
              c=df['chronotick_source'].map({'fusion': color, 'cpu': 'gray', 'error': 'red'}),
              alpha=0.5, s=5)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Prediction Source')
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['CPU', 'Fusion'])

    fusion_pct = 100 * (df['chronotick_source'] == 'fusion').sum() / len(df)
    ax.set_title(f'{name}: Fusion Pattern ({fusion_pct:.1f}% fusion)')
    ax.grid(True, alpha=0.3, axis='x')

# Row 3: Error comparison to system clock
for idx, (name, df, color) in enumerate([('Homelab', homelab_csv, colors['homelab']),
                                           ('ARES-11', ares11_csv, colors['ares-11']),
                                           ('ARES-12', ares12_csv, colors['ares-12'])]):
    ax = axes[2, idx]

    ntp_samples = df[df['has_ntp'] == True].copy()

    if len(ntp_samples) > 0:
        # For ARES-12, filter outliers
        if name == 'ARES-12':
            ntp_samples = ntp_samples[np.abs(ntp_samples['ntp_offset_ms']) < 20].copy()

        # System error = NTP offset
        # ChronoTick error = ChronoTick offset - NTP offset
        system_error = np.abs(ntp_samples['ntp_offset_ms'].values)
        chronotick_error = np.abs(ntp_samples['chronotick_offset_ms'].values - ntp_samples['ntp_offset_ms'].values)

        time_min = ntp_samples['elapsed_seconds'].values / 60

        ax.plot(time_min, system_error, label='System Clock Error', color='red', alpha=0.7, linewidth=2)
        ax.plot(time_min, chronotick_error, label='ChronoTick Error', color=color, alpha=0.7, linewidth=2)

        # Calculate MAE
        system_mae = system_error.mean()
        chronotick_mae = chronotick_error.mean()
        improvement = 100 * (system_mae - chronotick_mae) / system_mae

        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Absolute Error (ms)')
        ax.set_title(f'{name}: Error vs Ground Truth\n'
                     f'System MAE={system_mae:.2f}ms, ChronoTick MAE={chronotick_mae:.2f}ms\n'
                     f'Improvement: {improvement:.1f}%')
        ax.legend()
        ax.grid(True, alpha=0.3)

plt.tight_layout()
output_path = data_dir / 'improved_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Improved analysis saved to: {output_path}")

# ============================================================================
# FIGURE 2: HOMELAB SPIKE INVESTIGATION
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Homelab Spike Investigation (57-59 minutes)', fontsize=16, fontweight='bold')

# Full view with spike region
ax = axes[0, 0]
ax.plot(homelab_csv['elapsed_seconds'] / 60, homelab_csv['chronotick_offset_ms'],
        alpha=0.7, linewidth=0.8, color=colors['homelab'])
ax.axvspan(57, 59, alpha=0.2, color='red')
ax.set_xlabel('Time (minutes)')
ax.set_ylabel('ChronoTick Offset (ms)')
ax.set_title('Full Timeline (spike region highlighted)')
ax.grid(True, alpha=0.3)

# Zoomed view of spike region
ax = axes[0, 1]
spike_region = homelab_csv[(homelab_csv['elapsed_seconds'] >= 57*60) & (homelab_csv['elapsed_seconds'] <= 59*60)]
ax.plot(spike_region['elapsed_seconds'] / 60, spike_region['chronotick_offset_ms'],
        alpha=0.7, linewidth=1, color=colors['homelab'], marker='o', markersize=3)

# Color by source
for source, color in [('fusion', 'blue'), ('cpu', 'red')]:
    source_data = spike_region[spike_region['chronotick_source'] == source]
    ax.scatter(source_data['elapsed_seconds'] / 60, source_data['chronotick_offset_ms'],
              label=source, alpha=0.7, s=30, color=color)

ax.set_xlabel('Time (minutes)')
ax.set_ylabel('ChronoTick Offset (ms)')
ax.set_title('Zoomed: 57-59 minute spike')
ax.legend()
ax.grid(True, alpha=0.3)

# Histogram of offsets: spike region vs rest
ax = axes[1, 0]
spike_offsets = spike_region['chronotick_offset_ms']
normal_offsets = homelab_csv[~homelab_csv.index.isin(spike_region.index)]['chronotick_offset_ms']

ax.hist(normal_offsets, bins=50, alpha=0.5, label=f'Normal (μ={normal_offsets.mean():.2f}ms)', color='blue')
ax.hist(spike_offsets, bins=20, alpha=0.5, label=f'Spike region (μ={spike_offsets.mean():.2f}ms)', color='red')
ax.set_xlabel('Offset (ms)')
ax.set_ylabel('Frequency')
ax.set_title('Offset Distribution: Spike vs Normal')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Source breakdown in spike region
ax = axes[1, 1]
spike_sources = spike_region['chronotick_source'].value_counts()
ax.bar(spike_sources.index, spike_sources.values, alpha=0.7)
ax.set_xlabel('Source')
ax.set_ylabel('Count')
ax.set_title('Prediction Sources During Spike')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_path = data_dir / 'homelab_spike_investigation.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Homelab spike investigation saved to: {output_path}")

# ============================================================================
# FIGURE 3: COMPARISON SUMMARY
# ============================================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Cross-System Comparison Summary', fontsize=16, fontweight='bold')

# MAE comparison bar chart
ax = axes[0, 0]
systems = ['Homelab', 'ARES-11', 'ARES-12\n(filtered)']
system_mae = [2.818, 2.637, 1.601]  # ARES-12 filtered
chronotick_mae = [1.925, 2.122, 1.326]  # ARES-12 filtered
improvement = [31.7, 19.5, 17.2]  # ARES-12 filtered (estimated)

x = np.arange(len(systems))
width = 0.35

bars1 = ax.bar(x - width/2, system_mae, width, label='System Clock', alpha=0.7, color='red')
bars2 = ax.bar(x + width/2, chronotick_mae, width, label='ChronoTick', alpha=0.7, color='green')

# Add improvement percentages on top
for i, imp in enumerate(improvement):
    ax.text(i, max(system_mae[i], chronotick_mae[i]) + 0.2,
           f'{imp:.1f}%\nbetter', ha='center', fontsize=10, fontweight='bold')

ax.set_ylabel('Mean Absolute Error (ms)')
ax.set_title('MAE Comparison: System Clock vs ChronoTick')
ax.set_xticks(x)
ax.set_xticklabels(systems)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Fusion rate comparison
ax = axes[0, 1]
systems = ['Homelab', 'ARES-11', 'ARES-12']
fusion_pct = [80.3, 77.3, 80.8]
cpu_pct = [19.6, 22.6, 19.1]

x = np.arange(len(systems))
ax.bar(x, fusion_pct, label='Fusion', alpha=0.7, color='blue')
ax.bar(x, cpu_pct, bottom=fusion_pct, label='CPU-only', alpha=0.7, color='gray')

ax.set_ylabel('Percentage (%)')
ax.set_title('Prediction Source Distribution\n(~80% fusion, ~20% CPU-only)')
ax.set_xticks(x)
ax.set_xticklabels(systems)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 100])

# NTP rejection analysis
ax = axes[1, 0]
systems = ['Homelab', 'ARES-11']
accepted = [421, 400]
rejected = [34, 9]
rejection_rate = [7.5, 2.2]

x = np.arange(len(systems))
width = 0.35

ax.bar(x - width/2, accepted, width, label='Accepted', alpha=0.7, color='green')
ax.bar(x + width/2, rejected, width, label='Rejected', alpha=0.7, color='red')

# Add rejection rate on top
for i, rate in enumerate(rejection_rate):
    ax.text(i, max(accepted[i], rejected[i]) + 20,
           f'{rate:.1f}%\nrejected', ha='center', fontsize=10, fontweight='bold')

ax.set_ylabel('NTP Measurement Count')
ax.set_title('NTP Acceptance vs Rejection\n(Poor quality measurements rejected)')
ax.set_xticks(x)
ax.set_xticklabels(systems)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# Offset stability comparison
ax = axes[1, 1]
systems = ['Homelab', 'ARES-11', 'ARES-12']
mean_offset = [-2.311, 1.366, 1.284]
std_offset = [2.122, 0.479, 0.476]

x = np.arange(len(systems))
ax.bar(x, std_offset, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])

for i, (mean, std) in enumerate(zip(mean_offset, std_offset)):
    ax.text(i, std + 0.1, f'μ={mean:.2f}ms\nσ={std:.2f}ms',
           ha='center', fontsize=9)

ax.set_ylabel('Standard Deviation (ms)')
ax.set_title('ChronoTick Offset Stability\n(Lower is better)')
ax.set_xticks(x)
ax.set_xticklabels(systems)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
output_path = data_dir / 'comparison_summary.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Comparison summary saved to: {output_path}")

print("\n" + "="*80)
print("ALL PLOTS GENERATED")
print("="*80)
