#!/usr/bin/env python3
"""
FIXED VERSION - Generate Showcase Figures 3.1 and 3.2

Generates the synchronized and unsynchronized clock showcase figures.
This version FIXES the sign inversion bug in system_error calculation.

BUG FIX:
  OLD (WRONG): system_error = 0 - ntp_offset_ms  # ❌ INVERTED SIGN
  NEW (FIXED): system_error = ntp_offset_ms      # ✓ CORRECT

Output:
  - 3.1_synchronized_clock.pdf (Experiment-3 homelab - system NTP enabled)
  - 3.2_unsynchronized_clock.pdf (Experiment-7 homelab - system NTP disabled)
  - 3.1_offset_comparison.pdf (alternative view)
  - 3.2_unsynchronized_only.pdf (alternative view)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

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

def generate_synchronized_figure(csv_path, output_dir):
    """
    Generate synchronized clock showcase figure (3.1).

    Dataset: Experiment-3 homelab (system NTP enabled with chrony)
    Shows: ChronoTick vs system clock when system is synchronized
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 3.1: SYNCHRONIZED CLOCK")
    print("="*80)

    df = pd.read_csv(csv_path)
    ntp_df = df[df['has_ntp'] == True].copy()

    print(f"Total samples: {len(df)}")
    print(f"NTP samples: {len(ntp_df)}")

    # Convert to hours
    ntp_df['elapsed_hours'] = ntp_df['elapsed_seconds'] / 3600

    # FIXED: Correct error calculations
    # ChronoTick prediction error (this was always correct)
    ntp_df['chronotick_error'] = ntp_df['chronotick_offset_ms'] - ntp_df['ntp_offset_ms']

    # System clock offset (FIXED - no longer inverted!)
    ntp_df['system_error'] = ntp_df['ntp_offset_ms']  # ✓ FIXED: No negation!

    print(f"\n✓ FIXED ERROR CALCULATION:")
    print(f"  ChronoTick error range: [{ntp_df['chronotick_error'].min():.2f}, {ntp_df['chronotick_error'].max():.2f}] ms")
    print(f"  System offset range: [{ntp_df['system_error'].min():.2f}, {ntp_df['system_error'].max():.2f}] ms")
    print(f"  System offset mean: {ntp_df['system_error'].mean():.3f} ms")

    # Interpret the system offset
    if ntp_df['system_error'].mean() < 0:
        print(f"  → System clock is {abs(ntp_df['system_error'].mean()):.3f} ms BEHIND true time")
    else:
        print(f"  → System clock is {ntp_df['system_error'].mean():.3f} ms AHEAD of true time")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot ChronoTick ±1σ shaded region
    chronotick_lower = ntp_df['chronotick_error'] - ntp_df['chronotick_uncertainty_ms']
    chronotick_upper = ntp_df['chronotick_error'] + ntp_df['chronotick_uncertainty_ms']

    ax.fill_between(ntp_df['elapsed_hours'], chronotick_lower, chronotick_upper,
                     color='#5DA5DA', alpha=0.25, linewidth=0, label='ChronoTick ±1σ')

    # Plot System Clock offset (purple squares)
    ax.scatter(ntp_df['elapsed_hours'], ntp_df['system_error'],
               c='#CC79A7', marker='s', s=30, alpha=0.8,
               label='System Clock Offset', zorder=3)

    # Plot ChronoTick prediction error (blue circles)
    ax.scatter(ntp_df['elapsed_hours'], ntp_df['chronotick_error'],
               c='#5DA5DA', marker='o', s=30, alpha=0.8,
               label='ChronoTick Error', zorder=4)

    # Perfect sync reference line
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5,
               alpha=0.7, label='Perfect Sync', zorder=2)

    # Styling
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Offset from NTP Reference (ms)', fontsize=12)
    ax.set_title('Synchronized System Clock (chrony enabled)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, fancybox=False, shadow=False)
    ax.grid(True, alpha=0.3, zorder=1)

    # Set x-axis
    max_hours = int(np.ceil(ntp_df['elapsed_hours'].max()))
    ax.set_xlim(0, max_hours)
    ax.set_xticks(range(0, max_hours + 1))

    plt.tight_layout()

    # Save figures
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "3.1_synchronized_clock.pdf"
    png_path = output_dir / "3.1_synchronized_clock.png"

    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"\n✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")

    plt.close()

    # Calculate metrics
    chronotick_mae = ntp_df['chronotick_error'].abs().mean()
    system_mae = ntp_df['system_error'].abs().mean()

    print(f"\nMetrics:")
    print(f"  ChronoTick MAE: {chronotick_mae:.3f} ms")
    print(f"  System Clock MAE: {system_mae:.3f} ms")
    print(f"  Improvement: {(system_mae / chronotick_mae):.2f}× better")

def generate_unsynchronized_figure(csv_path, output_dir):
    """
    Generate unsynchronized clock showcase figure (3.2).

    Dataset: Experiment-7 homelab (system NTP disabled)
    Shows: ChronoTick vs system clock with unbounded drift
    """
    print("\n" + "="*80)
    print("GENERATING FIGURE 3.2: UNSYNCHRONIZED CLOCK")
    print("="*80)

    df = pd.read_csv(csv_path)
    ntp_df = df[df['has_ntp'] == True].copy()

    print(f"Total samples: {len(df)}")
    print(f"NTP samples: {len(ntp_df)}")

    # Convert to hours
    ntp_df['elapsed_hours'] = ntp_df['elapsed_seconds'] / 3600

    # FIXED: Correct error calculations
    ntp_df['chronotick_error'] = ntp_df['chronotick_offset_ms'] - ntp_df['ntp_offset_ms']
    ntp_df['system_error'] = ntp_df['ntp_offset_ms']  # ✓ FIXED: No negation!

    print(f"\n✓ FIXED ERROR CALCULATION:")
    print(f"  ChronoTick error range: [{ntp_df['chronotick_error'].min():.2f}, {ntp_df['chronotick_error'].max():.2f}] ms")
    print(f"  System offset range: [{ntp_df['system_error'].min():.2f}, {ntp_df['system_error'].max():.2f}] ms")

    # Calculate drift rate
    hours = ntp_df['elapsed_hours'].values
    system_offsets = ntp_df['system_error'].values
    drift_coef = np.polyfit(hours, system_offsets, 1)
    drift_rate = drift_coef[0]  # ms/hour

    print(f"  System drift rate: {drift_rate:.3f} ms/hour")
    if drift_rate > 0:
        print(f"  → System clock drifting AHEAD at {drift_rate:.3f} ms/hour")
    else:
        print(f"  → System clock drifting BEHIND at {abs(drift_rate):.3f} ms/hour")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot ChronoTick ±1σ shaded region
    chronotick_lower = ntp_df['chronotick_error'] - ntp_df['chronotick_uncertainty_ms']
    chronotick_upper = ntp_df['chronotick_error'] + ntp_df['chronotick_uncertainty_ms']

    ax.fill_between(ntp_df['elapsed_hours'], chronotick_lower, chronotick_upper,
                     color='#5DA5DA', alpha=0.25, linewidth=0, label='ChronoTick ±1σ')

    # Plot System Clock offset with drift (purple squares)
    ax.scatter(ntp_df['elapsed_hours'], ntp_df['system_error'],
               c='#CC79A7', marker='s', s=30, alpha=0.8,
               label=f'System Clock Offset ({drift_rate:.2f} ms/hr drift)', zorder=3)

    # Plot drift line
    drift_line = drift_coef[0] * hours + drift_coef[1]
    ax.plot(hours, drift_line, 'r--', linewidth=2, alpha=0.6, label='System Drift Trend')

    # Plot ChronoTick prediction error (blue circles)
    ax.scatter(ntp_df['elapsed_hours'], ntp_df['chronotick_error'],
               c='#5DA5DA', marker='o', s=30, alpha=0.8,
               label='ChronoTick Error', zorder=4)

    # Perfect sync reference line
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5,
               alpha=0.7, label='Perfect Sync', zorder=2)

    # Styling
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Offset from NTP Reference (ms)', fontsize=12)
    ax.set_title('Unsynchronized System Clock (NTP disabled)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, fancybox=False, shadow=False)
    ax.grid(True, alpha=0.3, zorder=1)

    # Set x-axis
    max_hours = int(np.ceil(ntp_df['elapsed_hours'].max()))
    ax.set_xlim(0, max_hours)
    ax.set_xticks(range(0, max_hours + 1))

    plt.tight_layout()

    # Save figures
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "3.2_unsynchronized_clock.pdf"
    png_path = output_dir / "3.2_unsynchronized_clock.png"

    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"\n✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")

    plt.close()

    # Calculate metrics
    chronotick_mae = ntp_df['chronotick_error'].abs().mean()
    system_mae = ntp_df['system_error'].abs().mean()

    print(f"\nMetrics:")
    print(f"  ChronoTick MAE: {chronotick_mae:.3f} ms")
    print(f"  System Clock MAE: {system_mae:.3f} ms")
    print(f"  Improvement: {(system_mae / chronotick_mae):.2f}× better")
    print(f"  System drift prevented: {drift_rate:.3f} ms/hour")

def main():
    """Generate all showcase figures."""
    print("="*80)
    print("SHOWCASE FIGURES GENERATOR - FIXED VERSION")
    print("="*80)
    print("\nBUG FIX APPLIED:")
    print("  OLD: system_error = 0 - ntp_offset_ms  ❌ (inverted sign)")
    print("  NEW: system_error = ntp_offset_ms      ✓ (correct)")

    output_dir = Path("results/figures_corrected/showcase")

    # Figure 3.1: Synchronized (Experiment-3 homelab)
    sync_csv = Path("results/experiment-3/homelab/data.csv")
    if sync_csv.exists():
        generate_synchronized_figure(sync_csv, output_dir)
    else:
        print(f"\n⚠️  Synchronized dataset not found: {sync_csv}")

    # Figure 3.2: Unsynchronized (Experiment-7 homelab)
    unsync_csv = Path("results/experiment-7/homelab/chronotick_client_validation_20251020_221631.csv")
    if unsync_csv.exists():
        generate_unsynchronized_figure(unsync_csv, output_dir)
    else:
        print(f"\n⚠️  Unsynchronized dataset not found: {unsync_csv}")

    print("\n" + "="*80)
    print("SHOWCASE FIGURES COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir.absolute()}")
    print("\nGenerated files:")
    print("  - 3.1_synchronized_clock.pdf/.png")
    print("  - 3.2_unsynchronized_clock.pdf/.png")

if __name__ == "__main__":
    main()
