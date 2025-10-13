#!/usr/bin/env python3
"""
Visualize Before/After Effects of Dataset Corrections

Shows:
1. SOLID LINE: Predictions returned to clients (what they received)
2. DASHED LINE: Dataset state after corrections (what ML model sees for future predictions)
3. RED X: NTP ground truth measurements
4. SHADED REGIONS: Correction events

This visualization demonstrates how NTP corrections modify the historical
dataset, which then affects future autoregressive predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

if len(sys.argv) < 2:
    print("Usage: python plot_correction_effects.py <method>")
    print("Example: python plot_correction_effects.py drift_aware")
    sys.exit(1)

method = sys.argv[1]

# Find most recent test results
results_dir = Path('results/ntp_correction_experiment/visualization_data')

# Find most recent files for this method
summary_files = sorted(results_dir.glob(f"summary_{method}_*.csv"))
client_files = sorted(results_dir.glob(f"client_predictions_{method}_*.csv"))
correction_files = sorted(results_dir.glob(f"dataset_corrections_{method}_*.csv"))

if not summary_files:
    print(f"No test results found for method: {method}")
    print(f"Looking in: {results_dir}")
    sys.exit(1)

# Prefer cleaned versions if they exist (for 'none' and 'linear' methods)
summary_file = summary_files[-1]
cleaned_version = Path(str(summary_file).replace('.csv', '_cleaned.csv'))
if cleaned_version.exists():
    print(f"Using cleaned version: {cleaned_version.name}")
    summary_file = cleaned_version

client_file = client_files[-1] if client_files else None
correction_file = correction_files[-1] if correction_files else None

print(f"Loading data for {method}...")
print(f"  Summary: {summary_file.name}")
if client_file:
    print(f"  Client predictions: {client_file.name}")
if correction_file:
    print(f"  Dataset corrections: {correction_file.name}")

# Load data
df_summary = pd.read_csv(summary_file)
df_client = pd.read_csv(client_file) if client_file else None
df_corrections = pd.read_csv(correction_file) if correction_file else None

# Create visualization
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1])
fig.suptitle(f'Correction Effects Visualization: {method.upper()}', fontsize=14, fontweight='bold')

# Plot 1: Main comparison
ax1.set_title('Client Predictions vs Dataset State (After Corrections)', fontsize=12)

# SOLID LINE: What clients received
ax1.plot(df_summary['elapsed_seconds'], df_summary['client_offset_ms'],
        color='blue', linewidth=2, label='Client predictions (continuous)', alpha=0.8)

# If we have dataset corrections, show the adjusted values
if df_corrections is not None and len(df_corrections) > 0:
    # Group by NTP event and plot dataset state after each correction
    ntp_events = df_corrections['ntp_event_timestamp'].unique()

    for i, ntp_time in enumerate(ntp_events[:5]):  # Show first 5 NTP events
        event_data = df_corrections[df_corrections['ntp_event_timestamp'] == ntp_time]

        if len(event_data) > 0:
            # Calculate time offsets
            time_offsets = event_data['time_since_interval_start_s'] + event_data['interval_start'].iloc[0] - df_summary['timestamp'].iloc[0]

            # Plot dataset state after this correction with markers
            ax1.plot(time_offsets,
                    event_data['offset_after_correction_ms'],
                    linestyle='--', linewidth=1.5, alpha=0.6, marker='x', markersize=3,
                    label=f'Corrected Dataset (sampling 1/sec) - NTP@{int(ntp_time-df_summary["timestamp"].iloc[0])}s' if i < 3 else None)

# Mark NTP evaluation measurements (ground truth comparisons)
df_with_ntp = df_summary[df_summary['has_ntp'] == True].copy()
if len(df_with_ntp) > 0:
    ax1.scatter(df_with_ntp['elapsed_seconds'], df_with_ntp['ntp_ground_truth_offset_ms'],
               color='red', s=100, marker='X', linewidth=2,
               label='NTP evaluation measurements', zorder=10)

# Highlight NTP correction events with vertical lines and labels
if df_corrections is not None and len(df_corrections) > 0:
    ntp_events = df_corrections['ntp_event_timestamp'].unique()
    for i, ntp_time in enumerate(ntp_events):
        elapsed = ntp_time - df_summary['timestamp'].iloc[0]
        ax1.axvline(x=elapsed, color='orange', linestyle=':', alpha=0.7, linewidth=2,
                   label='NTP correction event' if i == 0 else None)

ax1.set_xlabel('Time (seconds)', fontsize=11)
ax1.set_ylabel('Offset (ms)', fontsize=11)
ax1.legend(fontsize=9, loc='best')
ax1.grid(True, alpha=0.3)

# Plot 2: Correction deltas over time
ax2.set_title('Correction Deltas (How Much Dataset Was Adjusted)', fontsize=11)

if df_corrections is not None and len(df_corrections) > 0:
    # Show correction deltas for each NTP event
    ntp_events = df_corrections['ntp_event_timestamp'].unique()

    for i, ntp_time in enumerate(ntp_events):
        event_data = df_corrections[df_corrections['ntp_event_timestamp'] == ntp_time]

        if len(event_data) > 0:
            time_offsets = event_data['time_since_interval_start_s'] + event_data['interval_start'].iloc[0] - df_summary['timestamp'].iloc[0]

            ax2.plot(time_offsets, event_data['correction_delta_ms'],
                    linewidth=2, alpha=0.7,
                    label=f'NTP event @{int(ntp_time-df_summary["timestamp"].iloc[0])}s')

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.5)
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.set_ylabel('Correction Delta (ms)', fontsize=11)
    ax2.legend(fontsize=9, loc='best')
    ax2.grid(True, alpha=0.3)
else:
    ax2.text(0.5, 0.5, f'No corrections applied (method={method})',
            ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    ax2.set_xlabel('Time (seconds)', fontsize=11)
    ax2.set_ylabel('Correction Delta (ms)', fontsize=11)

plt.tight_layout()

output_path = results_dir / f'correction_effects_{method}.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\nSaved visualization: {output_path}")
plt.close()

# Print statistics
print(f"\n{'='*80}")
print(f"CORRECTION STATISTICS - {method.upper()}")
print(f"{'='*80}\n")

if df_corrections is not None and len(df_corrections) > 0:
    ntp_events = df_corrections['ntp_event_timestamp'].unique()
    print(f"Number of NTP correction events: {len(ntp_events)}")
    print(f"Total measurements corrected: {len(df_corrections)}")
    print(f"Average correction delta: {df_corrections['correction_delta_ms'].abs().mean():.3f}ms")
    print(f"Max correction delta: {df_corrections['correction_delta_ms'].abs().max():.3f}ms")
    print()

    for i, ntp_time in enumerate(ntp_events[:5]):
        event_data = df_corrections[df_corrections['ntp_event_timestamp'] == ntp_time]
        print(f"NTP Event {i+1}:")
        print(f"  Timestamp: {ntp_time:.0f}")
        print(f"  Measurements affected: {len(event_data)}")
        print(f"  Interval duration: {event_data['interval_duration_s'].iloc[0]:.0f}s")
        print(f"  Total error: {event_data['error_ms'].iloc[0]:.2f}ms")
        print(f"  Avg correction delta: {event_data['correction_delta_ms'].abs().mean():.3f}ms")
        print()
else:
    print("No corrections applied (method='none' or no NTP events)")

if len(df_with_ntp) > 0:
    chronotick_errors = df_with_ntp['chronotick_error_ms']
    print(f"ChronoTick Accuracy vs NTP Ground Truth:")
    print(f"  MAE: {chronotick_errors.mean():.3f}ms")
    print(f"  StdDev: {chronotick_errors.std():.3f}ms")
    print(f"  Max error: {chronotick_errors.max():.3f}ms")
    print(f"  Samples: {len(df_with_ntp)}")

print(f"\n{'='*80}")
