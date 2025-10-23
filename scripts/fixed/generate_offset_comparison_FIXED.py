#!/usr/bin/env python3
"""
FIXED VERSION - Generate Offset Comparison Figures

Generates alternative views of the showcase figures showing just offsets (not errors).
This version FIXES the sign inversion bug.

BUG FIX:
  Shows ntp_offset_ms directly (system clock offset from true time)
  No inversion applied

Output:
  - 3.1_offset_comparison.pdf (synchronized - both ChronoTick and system offsets)
  - 3.2_unsynchronized_only.pdf (unsynchronized - focus on drift)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set paper style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '-',
    'grid.linewidth': 0.5,
})

def generate_offset_comparison_synchronized(csv_path, output_dir):
    """
    Generate 3.1_offset_comparison.pdf - Shows both offsets for synchronized system.
    """
    print("\n" + "="*80)
    print("GENERATING: 3.1_offset_comparison.pdf")
    print("="*80)

    df = pd.read_csv(csv_path)
    ntp_df = df[df['has_ntp'] == True].copy()

    print(f"Total samples: {len(df)}")
    print(f"NTP samples: {len(ntp_df)}")

    ntp_df['elapsed_hours'] = ntp_df['elapsed_seconds'] / 3600

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot system clock offset (purple squares) - FIXED: no inversion
    ax.scatter(ntp_df['elapsed_hours'], ntp_df['ntp_offset_ms'],
               c='#CC79A7', marker='s', s=40, alpha=0.8,
               label='System Clock Offset', zorder=3)

    # Plot ChronoTick offset (blue circles)
    ax.scatter(ntp_df['elapsed_hours'], ntp_df['chronotick_offset_ms'],
               c='#5DA5DA', marker='o', s=35, alpha=0.7,
               label='ChronoTick Offset Prediction', zorder=4)

    # Perfect sync reference
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5,
               alpha=0.7, label='Perfect Sync (0 offset)', zorder=2)

    # Styling
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Offset from NTP Ground Truth (ms)', fontsize=12)
    ax.set_title('Synchronized System: ChronoTick vs System Clock Offsets',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, fancybox=False, shadow=False)
    ax.grid(True, alpha=0.3, zorder=1)

    max_hours = int(np.ceil(ntp_df['elapsed_hours'].max()))
    ax.set_xlim(0, max_hours)
    ax.set_xticks(range(0, max_hours + 1))

    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "3.1_offset_comparison.pdf"
    png_path = output_dir / "3.1_offset_comparison.png"

    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"\n✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")

    plt.close()

    # Stats
    system_mean = ntp_df['ntp_offset_ms'].mean()
    chronotick_mean = ntp_df['chronotick_offset_ms'].mean()

    print(f"\nOffset Statistics:")
    print(f"  System clock mean offset: {system_mean:.3f} ms")
    print(f"  ChronoTick mean offset: {chronotick_mean:.3f} ms")

    if system_mean < 0:
        print(f"  → System clock is {abs(system_mean):.3f} ms BEHIND")
    else:
        print(f"  → System clock is {system_mean:.3f} ms AHEAD")

def generate_unsynchronized_only(csv_path, output_dir):
    """
    Generate 3.2_unsynchronized_only.pdf - Focus on unsynchronized drift.
    """
    print("\n" + "="*80)
    print("GENERATING: 3.2_unsynchronized_only.pdf")
    print("="*80)

    df = pd.read_csv(csv_path)
    ntp_df = df[df['has_ntp'] == True].copy()

    print(f"Total samples: {len(df)}")
    print(f"NTP samples: {len(ntp_df)}")

    ntp_df['elapsed_hours'] = ntp_df['elapsed_seconds'] / 3600

    # Calculate drift
    hours = ntp_df['elapsed_hours'].values
    system_offsets = ntp_df['ntp_offset_ms'].values  # FIXED: no inversion
    drift_coef = np.polyfit(hours, system_offsets, 1)
    drift_rate = drift_coef[0]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot system clock offset with drift (purple squares) - FIXED
    ax.scatter(ntp_df['elapsed_hours'], ntp_df['ntp_offset_ms'],
               c='#CC79A7', marker='s', s=40, alpha=0.8,
               label=f'System Clock Offset (drift: {drift_rate:+.3f} ms/hr)', zorder=3)

    # Plot drift trend line
    drift_line = drift_coef[0] * hours + drift_coef[1]
    ax.plot(hours, drift_line, 'r--', linewidth=2.5, alpha=0.7,
            label=f'Linear Drift Trend', zorder=2)

    # Plot ChronoTick offset
    ax.scatter(ntp_df['elapsed_hours'], ntp_df['chronotick_offset_ms'],
               c='#5DA5DA', marker='o', s=35, alpha=0.7,
               label='ChronoTick Offset (compensated)', zorder=4)

    # Perfect sync reference
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5,
               alpha=0.5, label='Perfect Sync', zorder=1)

    # Styling
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Offset from NTP Ground Truth (ms)', fontsize=12)
    ax.set_title(f'Unsynchronized System: ChronoTick Compensates for {drift_rate:.3f} ms/hr Drift',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, fancybox=False, shadow=False, fontsize=9)
    ax.grid(True, alpha=0.3, zorder=0)

    max_hours = int(np.ceil(ntp_df['elapsed_hours'].max()))
    ax.set_xlim(0, max_hours)
    ax.set_xticks(range(0, max_hours + 1))

    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / "3.2_unsynchronized_only.pdf"
    png_path = output_dir / "3.2_unsynchronized_only.png"

    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"\n✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")

    plt.close()

    # Stats
    start_offset = drift_coef[1]
    end_offset = drift_coef[0] * hours[-1] + drift_coef[1]
    total_drift = end_offset - start_offset

    print(f"\nDrift Statistics:")
    print(f"  Drift rate: {drift_rate:+.3f} ms/hour")
    print(f"  Initial offset: {start_offset:.3f} ms")
    print(f"  Final offset: {end_offset:.3f} ms")
    print(f"  Total drift over {hours[-1]:.1f}h: {total_drift:.3f} ms")

    if drift_rate > 0:
        print(f"  → System clock drifting AHEAD")
    else:
        print(f"  → System clock drifting BEHIND")

def main():
    """Generate offset comparison figures."""
    print("="*80)
    print("OFFSET COMPARISON FIGURES GENERATOR - FIXED VERSION")
    print("="*80)

    output_dir = Path("results/figures_corrected/showcase")

    # 3.1 offset comparison (synchronized)
    sync_csv = Path("results/experiment-3/homelab/data.csv")
    if sync_csv.exists():
        generate_offset_comparison_synchronized(sync_csv, output_dir)
    else:
        print(f"\n⚠️  Synchronized dataset not found: {sync_csv}")

    # 3.2 unsynchronized only
    unsync_csv = Path("results/experiment-7/homelab/chronotick_client_validation_20251020_221631.csv")
    if unsync_csv.exists():
        generate_unsynchronized_only(unsync_csv, output_dir)
    else:
        print(f"\n⚠️  Unsynchronized dataset not found: {unsync_csv}")

    print("\n" + "="*80)
    print("OFFSET COMPARISON FIGURES COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir.absolute()}")

if __name__ == "__main__":
    main()
