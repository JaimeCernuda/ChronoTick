#!/usr/bin/env python3
"""
Generate paper-style figure for Experiment-10 Homelab results.

Matches the style of 3.1_synchronized_clock.pdf:
- Error from NTP Reference (ms) on y-axis
- Time (hours) on x-axis
- ChronoTick points (blue circles) with ±1σ shaded region
- System Clock points (orange squares)
- Perfect Sync reference line (dashed gray)
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

def remove_outliers(df, column, n_sigma=3):
    """Remove outliers beyond n_sigma from median."""
    median = df[column].median()
    std = df[column].std()
    lower_bound = median - n_sigma * std
    upper_bound = median + n_sigma * std

    mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
    removed = len(df) - mask.sum()
    print(f"  Removed {removed} outliers beyond {n_sigma}σ ({removed/len(df)*100:.1f}%)")
    print(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}] ms")

    return df[mask].copy()

def main():
    """Generate paper-style figure for homelab."""

    csv_path = Path("results/experiment-10/homelab/chronotick_client_validation_20251022_192238.csv")
    output_dir = Path("results/experiment-10/homelab")

    print("="*60)
    print("Homelab Paper-Style Figure Generation")
    print("="*60)

    # Load data
    df = pd.read_csv(csv_path)
    print(f"\nTotal samples: {len(df)}")

    # Filter to NTP measurements only
    ntp_df = df[df['has_ntp'] == True].copy()
    print(f"NTP samples: {len(ntp_df)}")

    # Convert to hours
    ntp_df['elapsed_hours'] = ntp_df['elapsed_seconds'] / 3600

    # Calculate errors (difference from NTP ground truth = 0)
    # Error = measurement - ground_truth
    # For ChronoTick: error = chronotick_offset - ntp_offset
    # For System Clock: error = 0 - ntp_offset (system assumes perfect sync)
    ntp_df['chronotick_error'] = ntp_df['chronotick_offset_ms'] - ntp_df['ntp_offset_ms']
    ntp_df['system_error'] = 0 - ntp_df['ntp_offset_ms']

    print(f"\nBefore outlier removal:")
    print(f"  ChronoTick error range: [{ntp_df['chronotick_error'].min():.2f}, {ntp_df['chronotick_error'].max():.2f}] ms")
    print(f"  System error range: [{ntp_df['system_error'].min():.2f}, {ntp_df['system_error'].max():.2f}] ms")

    # Remove outliers (beyond 3σ from median)
    print(f"\nRemoving ChronoTick outliers:")
    ntp_df = remove_outliers(ntp_df, 'chronotick_error', n_sigma=3)

    print(f"\nAfter outlier removal:")
    print(f"  Remaining samples: {len(ntp_df)}")
    print(f"  ChronoTick error range: [{ntp_df['chronotick_error'].min():.2f}, {ntp_df['chronotick_error'].max():.2f}] ms")
    print(f"  ChronoTick mean error: {ntp_df['chronotick_error'].mean():.4f} ms")
    print(f"  ChronoTick median error: {ntp_df['chronotick_error'].median():.4f} ms")
    print(f"  System mean error: {ntp_df['system_error'].mean():.4f} ms")
    print(f"  System median error: {ntp_df['system_error'].median():.4f} ms")

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot ChronoTick ±1σ shaded region
    chronotick_lower = ntp_df['chronotick_error'] - ntp_df['chronotick_uncertainty_ms']
    chronotick_upper = ntp_df['chronotick_error'] + ntp_df['chronotick_uncertainty_ms']

    ax.fill_between(ntp_df['elapsed_hours'], chronotick_lower, chronotick_upper,
                     color='#5DA5DA', alpha=0.25, linewidth=0, label='ChronoTick ±1σ')

    # Plot System Clock points (orange squares)
    ax.scatter(ntp_df['elapsed_hours'], ntp_df['system_error'],
               c='#FAA43A', marker='s', s=30, alpha=0.8,
               label='System Clock (with NTP enabled)', zorder=3)

    # Plot ChronoTick points (blue circles)
    ax.scatter(ntp_df['elapsed_hours'], ntp_df['chronotick_error'],
               c='#5DA5DA', marker='o', s=30, alpha=0.8,
               label='ChronoTick', zorder=4)

    # Perfect sync reference line
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5,
               alpha=0.7, label='Perfect Sync', zorder=2)

    # Styling
    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Error from NTP Reference (ms)', fontsize=12)
    ax.legend(loc='upper left', frameon=True, fancybox=False, shadow=False)
    ax.grid(True, alpha=0.3, zorder=1)

    # Set x-axis to show integer hours
    max_hours = int(np.ceil(ntp_df['elapsed_hours'].max()))
    ax.set_xlim(0, max_hours)
    ax.set_xticks(range(0, max_hours + 1))

    plt.tight_layout()

    # Save as both PNG and PDF
    png_path = output_dir / "homelab_paper_style.png"
    pdf_path = output_dir / "homelab_paper_style.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved PNG: {png_path}")

    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✓ Saved PDF: {pdf_path}")

    plt.close()

    # Calculate sigma coverage after outlier removal
    ntp_df['within_1sigma'] = abs(ntp_df['chronotick_error']) <= ntp_df['chronotick_uncertainty_ms']
    ntp_df['within_2sigma'] = abs(ntp_df['chronotick_error']) <= (2 * ntp_df['chronotick_uncertainty_ms'])
    ntp_df['within_3sigma'] = abs(ntp_df['chronotick_error']) <= (3 * ntp_df['chronotick_uncertainty_ms'])

    sigma_1 = (ntp_df['within_1sigma'].sum() / len(ntp_df)) * 100
    sigma_2 = (ntp_df['within_2sigma'].sum() / len(ntp_df)) * 100
    sigma_3 = (ntp_df['within_3sigma'].sum() / len(ntp_df)) * 100

    print(f"\nSigma Coverage (after outlier removal):")
    print(f"  Within 1σ: {sigma_1:.2f}% (expected 68.3%)")
    print(f"  Within 2σ: {sigma_2:.2f}% (expected 95.4%)")
    print(f"  Within 3σ: {sigma_3:.2f}% (expected 99.7%)")

    print(f"\n{'='*60}")
    print("Complete!")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
