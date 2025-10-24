#!/usr/bin/env python3
"""
ALTERNATIVE VERSION - Show BOTH as offsets (not mixing offset + error)

This makes the figure easier to interpret:
- System Clock: Actual offset from true time
- ChronoTick: Predicted offset from true time
- Both should be close together around -3ms

This is clearer than mixing offset (system) and error (ChronoTick).
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

def generate_both_offsets_figure(csv_path, output_dir):
    """
    Generate figure showing BOTH as offsets (clearer visualization).
    """
    print("\n" + "="*80)
    print("GENERATING: 3.1_synchronized_BOTH_OFFSETS.pdf")
    print("="*80)

    df = pd.read_csv(csv_path)
    ntp_df = df[df['has_ntp'] == True].copy()

    print(f"Total samples: {len(df)}")
    print(f"NTP samples: {len(ntp_df)}")

    ntp_df['elapsed_hours'] = ntp_df['elapsed_seconds'] / 3600

    # BOTH as offsets (no error calculation for ChronoTick)
    # System offset from true time
    system_offset = ntp_df['ntp_offset_ms']

    # ChronoTick's prediction of system offset
    chronotick_offset = ntp_df['chronotick_offset_ms']

    # Calculate prediction error for metrics only
    prediction_error = chronotick_offset - system_offset

    print(f"\n✓ BOTH SHOWN AS OFFSETS:")
    print(f"  System offset mean: {system_offset.mean():.3f} ms")
    print(f"  ChronoTick offset mean: {chronotick_offset.mean():.3f} ms")
    print(f"  Prediction error (for metrics): {prediction_error.abs().mean():.3f} ms MAE")

    if system_offset.mean() < 0:
        print(f"  → System clock is {abs(system_offset.mean()):.3f} ms BEHIND")
        print(f"  → ChronoTick predicts {abs(chronotick_offset.mean()):.3f} ms BEHIND")
    else:
        print(f"  → System clock is {system_offset.mean():.3f} ms AHEAD")
        print(f"  → ChronoTick predicts {chronotick_offset.mean():.3f} ms AHEAD")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot ChronoTick ±1σ uncertainty band (around ChronoTick offset, not error)
    chronotick_lower = ntp_df['chronotick_offset_ms'] - ntp_df['chronotick_uncertainty_ms']
    chronotick_upper = ntp_df['chronotick_offset_ms'] + ntp_df['chronotick_uncertainty_ms']

    ax.fill_between(ntp_df['elapsed_hours'], chronotick_lower, chronotick_upper,
                     color='#5DA5DA', alpha=0.2, linewidth=0,
                     label='ChronoTick ±1σ Uncertainty', zorder=2)

    # Plot System Clock offset (purple squares)
    ax.scatter(ntp_df['elapsed_hours'], ntp_df['ntp_offset_ms'],
               c='#CC79A7', marker='s', s=35, alpha=0.8,
               label='System Clock Offset (NTP ground truth)', zorder=4)

    # Plot ChronoTick offset prediction (blue circles)
    ax.scatter(ntp_df['elapsed_hours'], ntp_df['chronotick_offset_ms'],
               c='#5DA5DA', marker='o', s=30, alpha=0.7,
               label='ChronoTick Offset Prediction', zorder=3)

    # Perfect sync reference line
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5,
               alpha=0.7, label='Perfect Sync (0 offset)', zorder=1)

    # Styling
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Offset from True Time (ms)', fontsize=12)
    ax.set_title('Synchronized System: ChronoTick Tracks System Clock Offset',
                 fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, fancybox=False, shadow=False, fontsize=9)
    ax.grid(True, alpha=0.3, zorder=0)

    # Set x-axis
    max_hours = int(np.ceil(ntp_df['elapsed_hours'].max()))
    ax.set_xlim(0, max_hours)
    ax.set_xticks(range(0, max_hours + 1))

    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "3.1_synchronized_BOTH_OFFSETS.pdf"
    png_path = output_dir / "3.1_synchronized_BOTH_OFFSETS.png"

    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"\n✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")

    plt.close()

    # Calculate metrics
    mae = prediction_error.abs().mean()
    system_mae = system_offset.abs().mean()

    print(f"\nPerformance Metrics:")
    print(f"  ChronoTick prediction MAE: {mae:.3f} ms")
    print(f"  System Clock MAE from perfect sync: {system_mae:.3f} ms")
    print(f"  ChronoTick improvement: {(system_mae / mae):.2f}× better")

def main():
    """Generate clearer version showing both as offsets."""
    print("="*80)
    print("ALTERNATIVE FIGURE: BOTH AS OFFSETS (CLEARER)")
    print("="*80)
    print("\nThis version shows:")
    print("  • System Clock: Actual offset")
    print("  • ChronoTick: Predicted offset")
    print("  • Both on SAME scale (both offsets)")
    print("  • Should overlap/track each other")

    output_dir = Path("results/figures_corrected/showcase")

    # Synchronized (Experiment-3 homelab)
    sync_csv = Path("results/experiment-3/homelab/data.csv")
    if sync_csv.exists():
        generate_both_offsets_figure(sync_csv, output_dir)
    else:
        print(f"\n⚠️  Synchronized dataset not found: {sync_csv}")

    print("\n" + "="*80)
    print("COMPARISON OF APPROACHES")
    print("="*80)
    print("\n📊 3.1_synchronized_clock.pdf (original approach):")
    print("   • System: offset (-3ms)")
    print("   • ChronoTick: prediction error (+0.2ms)")
    print("   • Different scales - visually separated")
    print("   • Emphasizes accuracy (error near 0)")
    print()
    print("📊 3.1_synchronized_BOTH_OFFSETS.pdf (this new version):")
    print("   • System: offset (-3ms)")
    print("   • ChronoTick: offset (-3.2ms)")
    print("   • Same scale - visually together")
    print("   • Emphasizes tracking (both near -3ms)")
    print()
    print("📊 3.1_offset_comparison.pdf (similar to new version):")
    print("   • Also shows both as offsets")
    print("   • No uncertainty band")
    print("   • Cleaner but less information")

if __name__ == "__main__":
    main()
