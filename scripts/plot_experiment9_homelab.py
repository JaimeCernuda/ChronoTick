#!/usr/bin/env python3
"""
Analyze Experiment-9 Homelab results to compare with Experiment-10.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set paper style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'figure.dpi': 300,
})

def analyze_experiment_9():
    """Analyze experiment-9 homelab data."""

    csv_path = Path("results/experiment-9/homelab/chronotick_client_validation_20251022_094657.csv")
    output_dir = Path("results/experiment-9/homelab")

    print("="*60)
    print("Experiment-9 Homelab Analysis")
    print("="*60)

    # Load data
    df = pd.read_csv(csv_path)
    print(f"\nTotal samples: {len(df)}")

    # Filter to NTP measurements
    ntp_df = df[df['has_ntp'] == True].copy()
    print(f"NTP samples: {len(ntp_df)}")

    # Convert to hours
    ntp_df['elapsed_hours'] = ntp_df['elapsed_seconds'] / 3600

    # Calculate errors
    ntp_df['chronotick_error'] = ntp_df['chronotick_offset_ms'] - ntp_df['ntp_offset_ms']
    ntp_df['system_error'] = 0 - ntp_df['ntp_offset_ms']

    # Check for early spike
    early_spike = ntp_df[ntp_df['elapsed_hours'] < 0.5]
    if len(early_spike) > 0:
        print(f"\nEarly measurements (first 30 min): {len(early_spike)}")
        print(f"  ChronoTick error range: [{early_spike['chronotick_error'].min():.2f}, {early_spike['chronotick_error'].max():.2f}] ms")
        print(f"  NTP offset range: [{early_spike['ntp_offset_ms'].min():.2f}, {early_spike['ntp_offset_ms'].max():.2f}] ms")

    # Remove first 30 minutes (potential warmup issues)
    ntp_df_clean = ntp_df[ntp_df['elapsed_hours'] >= 0.5].copy()
    print(f"\nAfter removing first 30min: {len(ntp_df_clean)} samples")

    print(f"\nPerformance Metrics (cleaned):")
    print(f"  ChronoTick Mean Error: {ntp_df_clean['chronotick_error'].abs().mean():.4f} ms")
    print(f"  ChronoTick Median Error: {ntp_df_clean['chronotick_error'].abs().median():.4f} ms")
    print(f"  System Mean Error: {ntp_df_clean['system_error'].abs().mean():.4f} ms")
    print(f"  System Median Error: {ntp_df_clean['system_error'].abs().median():.4f} ms")

    print(f"\nNTP Ground Truth:")
    print(f"  Mean offset: {ntp_df_clean['ntp_offset_ms'].mean():.4f} ms")
    print(f"  Std offset: {ntp_df_clean['ntp_offset_ms'].std():.4f} ms")

    print(f"\nChronoTick Predictions:")
    print(f"  Mean offset: {ntp_df_clean['chronotick_offset_ms'].mean():.4f} ms")
    print(f"  Std offset: {ntp_df_clean['chronotick_offset_ms'].std():.4f} ms")

    # Calculate sigma coverage
    ntp_df_clean['within_1sigma'] = abs(ntp_df_clean['chronotick_error']) <= ntp_df_clean['chronotick_uncertainty_ms']
    sigma_1 = (ntp_df_clean['within_1sigma'].sum() / len(ntp_df_clean)) * 100
    print(f"  Within 1σ: {sigma_1:.2f}%")

    # Create paper-style plot
    fig, ax = plt.subplots(figsize=(10, 5))

    # ChronoTick ±1σ shaded region
    chronotick_lower = ntp_df_clean['chronotick_error'] - ntp_df_clean['chronotick_uncertainty_ms']
    chronotick_upper = ntp_df_clean['chronotick_error'] + ntp_df_clean['chronotick_uncertainty_ms']

    ax.fill_between(ntp_df_clean['elapsed_hours'], chronotick_lower, chronotick_upper,
                     color='#5DA5DA', alpha=0.25, linewidth=0, label='ChronoTick ±1σ')

    # System Clock points (orange squares)
    ax.scatter(ntp_df_clean['elapsed_hours'], ntp_df_clean['system_error'],
               c='#FAA43A', marker='s', s=30, alpha=0.8,
               label='System Clock (with NTP enabled)', zorder=3)

    # ChronoTick points (blue circles)
    ax.scatter(ntp_df_clean['elapsed_hours'], ntp_df_clean['chronotick_error'],
               c='#5DA5DA', marker='o', s=30, alpha=0.8,
               label='ChronoTick', zorder=4)

    # Perfect sync line
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5,
               alpha=0.7, label='Perfect Sync', zorder=2)

    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Error from NTP Reference (ms)', fontsize=12)
    ax.legend(loc='upper left', frameon=True, fancybox=False, shadow=False)
    ax.grid(True, alpha=0.3, zorder=1)

    max_hours = int(np.ceil(ntp_df_clean['elapsed_hours'].max()))
    ax.set_xlim(0.5, max_hours)

    plt.tight_layout()

    png_path = output_dir / "experiment9_homelab_paper_style.png"
    pdf_path = output_dir / "experiment9_homelab_paper_style.pdf"

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved PNG: {png_path}")

    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"✓ Saved PDF: {pdf_path}")

    plt.close()

    # Also plot the full data including early spike
    fig, ax = plt.subplots(figsize=(10, 5))

    chronotick_lower = ntp_df['chronotick_error'] - ntp_df['chronotick_uncertainty_ms']
    chronotick_upper = ntp_df['chronotick_error'] + ntp_df['chronotick_uncertainty_ms']

    ax.fill_between(ntp_df['elapsed_hours'], chronotick_lower, chronotick_upper,
                     color='#5DA5DA', alpha=0.25, linewidth=0, label='ChronoTick ±1σ')

    ax.scatter(ntp_df['elapsed_hours'], ntp_df['system_error'],
               c='#FAA43A', marker='s', s=30, alpha=0.8,
               label='System Clock (with NTP enabled)', zorder=3)

    ax.scatter(ntp_df['elapsed_hours'], ntp_df['chronotick_error'],
               c='#5DA5DA', marker='o', s=30, alpha=0.8,
               label='ChronoTick', zorder=4)

    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1.5,
               alpha=0.7, label='Perfect Sync', zorder=2)

    # Highlight early spike region
    ax.axvspan(0, 0.5, alpha=0.1, color='red', label='Warmup Period (excluded)')

    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Error from NTP Reference (ms)', fontsize=12)
    ax.legend(loc='upper left', frameon=True, fancybox=False, shadow=False)
    ax.grid(True, alpha=0.3, zorder=1)

    plt.tight_layout()

    full_png_path = output_dir / "experiment9_homelab_full.png"
    plt.savefig(full_png_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved full plot: {full_png_path}")

    plt.close()

    return ntp_df_clean

if __name__ == "__main__":
    analyze_experiment_9()
