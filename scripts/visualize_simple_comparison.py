#!/usr/bin/env python3
"""
Simple visualization: System Clock vs ChronoTick offset comparison
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Load data
results_dir = Path("/home/jcernuda/tick_project/ChronoTick/results/local-executions/backtracking_fix")
local_csv = results_dir / "chronotick_stability_20251017_014958.csv"
homelab_csv = results_dir / "homelab_test.csv"

print("Loading test results...")
df_local = pd.read_csv(local_csv)
df_homelab = pd.read_csv(homelab_csv)

# Filter for rows with NTP data
df_local_ntp = df_local[df_local['ntp_offset_ms'].notna()].copy()
df_homelab_ntp = df_homelab[df_homelab['ntp_offset_ms'].notna()].copy()

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# ============================================================================
# LOCAL TEST
# ============================================================================
if len(df_local_ntp) > 0:
    time_hours = df_local_ntp['elapsed_seconds'] / 3600
    chronotick_error = df_local_ntp['chronotick_error_vs_ntp_ms'].astype(float)
    system_error = df_local_ntp['system_error_vs_ntp_ms'].astype(float)

    # Plot with error bands
    ax1.plot(time_hours, chronotick_error, 'o-', color='#2E86AB',
             linewidth=2, markersize=6, alpha=0.7, label='ChronoTick Error vs NTP')
    ax1.plot(time_hours, system_error, 's-', color='#A23B72',
             linewidth=2, markersize=6, alpha=0.7, label='System Clock Error vs NTP')

    # Add running mean lines
    window = 5
    if len(chronotick_error) >= window:
        ct_mean = chronotick_error.rolling(window=window, center=True).mean()
        sys_mean = system_error.rolling(window=window, center=True).mean()
        ax1.plot(time_hours, ct_mean, '-', color='#2E86AB', linewidth=3, alpha=0.3)
        ax1.plot(time_hours, sys_mean, '-', color='#A23B72', linewidth=3, alpha=0.3)

    ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3, label='Perfect sync (0ms)')
    ax1.fill_between(time_hours, -50, 50, alpha=0.1, color='green', label='±50ms tolerance')

    # Statistics box
    ct_mean_val = chronotick_error.mean()
    ct_std_val = chronotick_error.std()
    sys_mean_val = system_error.mean()
    sys_std_val = system_error.std()

    stats_text = f'ChronoTick: {ct_mean_val:.1f} ± {ct_std_val:.1f} ms\n'
    stats_text += f'System Clock: {sys_mean_val:.1f} ± {sys_std_val:.1f} ms'

    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax1.set_xlabel('Time (hours)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Error vs NTP Ground Truth (ms)', fontsize=13, fontweight='bold')
    ax1.set_title('LOCAL TEST (WSL2)\nSystem Clock vs ChronoTick Accuracy',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper right', fontsize=10, framealpha=0.9)

# ============================================================================
# HOMELAB TEST
# ============================================================================
if len(df_homelab_ntp) > 0:
    time_hours = df_homelab_ntp['elapsed_seconds'] / 3600
    chronotick_error = df_homelab_ntp['chronotick_error_vs_ntp_ms'].astype(float)
    system_error = df_homelab_ntp['system_error_vs_ntp_ms'].astype(float)

    # Plot with error bands
    ax2.plot(time_hours, chronotick_error, 'o-', color='#2E86AB',
             linewidth=2, markersize=6, alpha=0.7, label='ChronoTick Error vs NTP')
    ax2.plot(time_hours, system_error, 's-', color='#A23B72',
             linewidth=2, markersize=6, alpha=0.7, label='System Clock Error vs NTP')

    # Add running mean lines
    window = 5
    if len(chronotick_error) >= window:
        ct_mean = chronotick_error.rolling(window=window, center=True).mean()
        sys_mean = system_error.rolling(window=window, center=True).mean()
        ax2.plot(time_hours, ct_mean, '-', color='#2E86AB', linewidth=3, alpha=0.3)
        ax2.plot(time_hours, sys_mean, '-', color='#A23B72', linewidth=3, alpha=0.3)

    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3, label='Perfect sync (0ms)')
    ax2.fill_between(time_hours, -50, 50, alpha=0.1, color='green', label='±50ms tolerance')

    # Statistics box
    ct_mean_val = chronotick_error.mean()
    ct_std_val = chronotick_error.std()
    sys_mean_val = system_error.mean()
    sys_std_val = system_error.std()

    stats_text = f'ChronoTick: {ct_mean_val:.1f} ± {ct_std_val:.1f} ms\n'
    stats_text += f'System Clock: {sys_mean_val:.1f} ± {sys_std_val:.1f} ms'

    ax2.text(0.02, 0.98, stats_text, transform=ax2.transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    ax2.set_xlabel('Time (hours)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Error vs NTP Ground Truth (ms)', fontsize=13, fontweight='bold')
    ax2.set_title('HOMELAB TEST\nSystem Clock vs ChronoTick Accuracy',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.legend(loc='upper right', fontsize=10, framealpha=0.9)

# Main title
fig.suptitle('ChronoTick Backtracking Fix: System Clock vs Corrected Time Accuracy\n' +
             'Comparing error vs NTP ground truth over extended test periods',
             fontsize=15, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save figure
output_path = results_dir / "simple_comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Visualization saved to: {output_path}")

# Also create a combined single plot for paper/presentation
fig2, ax = plt.subplots(1, 1, figsize=(12, 7))

# Plot both tests on same graph
if len(df_local_ntp) > 0:
    time_hours_local = df_local_ntp['elapsed_seconds'] / 3600
    ct_error_local = df_local_ntp['chronotick_error_vs_ntp_ms'].astype(float)
    sys_error_local = df_local_ntp['system_error_vs_ntp_ms'].astype(float)

    ax.plot(time_hours_local, ct_error_local, 'o-', color='#2E86AB',
            linewidth=1.5, markersize=4, alpha=0.6, label='ChronoTick (Local WSL2)')
    ax.plot(time_hours_local, sys_error_local, 's-', color='#A23B72',
            linewidth=1.5, markersize=4, alpha=0.6, label='System Clock (Local WSL2)')

if len(df_homelab_ntp) > 0:
    time_hours_homelab = df_homelab_ntp['elapsed_seconds'] / 3600
    ct_error_homelab = df_homelab_ntp['chronotick_error_vs_ntp_ms'].astype(float)
    sys_error_homelab = df_homelab_ntp['system_error_vs_ntp_ms'].astype(float)

    ax.plot(time_hours_homelab, ct_error_homelab, 'o-', color='#06A77D',
            linewidth=1.5, markersize=4, alpha=0.8, label='ChronoTick (Homelab)')
    ax.plot(time_hours_homelab, sys_error_homelab, 's-', color='#D62246',
            linewidth=1.5, markersize=4, alpha=0.8, label='System Clock (Homelab)')

ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label='Perfect sync (0ms)')
ax.fill_between([0, 12], -50, 50, alpha=0.1, color='green', label='±50ms tolerance')

ax.set_xlabel('Time (hours)', fontsize=14, fontweight='bold')
ax.set_ylabel('Error vs NTP Ground Truth (ms)', fontsize=14, fontweight='bold')
ax.set_title('System Clock vs ChronoTick: Accuracy Comparison\nError vs NTP Ground Truth (Lower is Better)',
             fontsize=15, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.legend(loc='best', fontsize=11, framealpha=0.9, ncol=2)

# Add summary statistics
if len(df_homelab_ntp) > 0:
    homelab_ct_mean = df_homelab_ntp['chronotick_error_vs_ntp_ms'].astype(float).mean()
    homelab_ct_std = df_homelab_ntp['chronotick_error_vs_ntp_ms'].astype(float).std()
    summary = f'Homelab ChronoTick: {homelab_ct_mean:.1f}ms ± {homelab_ct_std:.1f}ms'
    ax.text(0.98, 0.98, summary, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.9))

plt.tight_layout()

output_path2 = results_dir / "combined_comparison.png"
plt.savefig(output_path2, dpi=150, bbox_inches='tight')
print(f"✅ Combined plot saved to: {output_path2}")

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)
