#!/usr/bin/env python3
"""
Investigate the bimodal NTP pattern on ARES comp-11.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'figure.dpi': 150,
})

def analyze_bimodal_pattern(platform, csv_path):
    """Analyze NTP bimodal pattern."""

    df = pd.read_csv(csv_path)
    ntp_df = df[df['has_ntp'] == True].copy()

    print(f"\n{'='*60}")
    print(f"{platform} NTP Analysis")
    print(f"{'='*60}")
    print(f"Total NTP measurements: {len(ntp_df)}")

    # Check bimodal pattern
    positive = (ntp_df['ntp_offset_ms'] > 0).sum()
    negative = (ntp_df['ntp_offset_ms'] < 0).sum()

    print(f"\nNTP Offset Distribution:")
    print(f"  Positive offsets: {positive} ({positive/len(ntp_df)*100:.1f}%)")
    print(f"  Negative offsets: {negative} ({negative/len(ntp_df)*100:.1f}%)")
    print(f"  Mean: {ntp_df['ntp_offset_ms'].mean():.4f} ms")
    print(f"  Median: {ntp_df['ntp_offset_ms'].median():.4f} ms")
    print(f"  Std: {ntp_df['ntp_offset_ms'].std():.4f} ms")

    # Check ChronoTick vs System alignment
    ntp_df['chronotick_error'] = ntp_df['chronotick_offset_ms'] - ntp_df['ntp_offset_ms']
    ntp_df['system_error'] = 0 - ntp_df['ntp_offset_ms']

    print(f"\nChronoTick Error from NTP:")
    print(f"  Mean: {ntp_df['chronotick_error'].mean():.4f} ms")
    print(f"  Median: {ntp_df['chronotick_error'].median():.4f} ms")
    print(f"  Std: {ntp_df['chronotick_error'].std():.4f} ms")

    print(f"\nSystem Clock Error from NTP:")
    print(f"  Mean: {ntp_df['system_error'].mean():.4f} ms")
    print(f"  Median: {ntp_df['system_error'].median():.4f} ms")
    print(f"  Std: {ntp_df['system_error'].std():.4f} ms")

    # Check correlation between ChronoTick and System
    correlation = ntp_df[['chronotick_offset_ms', 'ntp_offset_ms']].corr().iloc[0, 1]
    print(f"\nCorrelation between ChronoTick and NTP offset: {correlation:.4f}")

    # Plot histogram
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # NTP offset histogram
    axes[0].hist(ntp_df['ntp_offset_ms'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Sync')
    axes[0].axvline(ntp_df['ntp_offset_ms'].mean(), color='green', linestyle='-', linewidth=2, label=f'Mean: {ntp_df["ntp_offset_ms"].mean():.2f}ms')
    axes[0].set_xlabel('NTP Offset (ms)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'{platform}: NTP Offset Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Error comparison
    errors = pd.DataFrame({
        'ChronoTick Error': ntp_df['chronotick_error'],
        'System Clock Error': ntp_df['system_error']
    })

    axes[1].hist([errors['ChronoTick Error'], errors['System Clock Error']],
                 bins=30, label=['ChronoTick Error', 'System Clock Error'],
                 color=['steelblue', 'orange'], alpha=0.6, edgecolor='black')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect Sync')
    axes[1].set_xlabel('Error from NTP Reference (ms)')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'{platform}: Error Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = csv_path.parent / f"{platform.lower().replace(' ', '_')}_bimodal_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    plt.close()

    # Time series showing the flickering
    ntp_df['elapsed_hours'] = ntp_df['elapsed_seconds'] / 3600

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.scatter(ntp_df['elapsed_hours'], ntp_df['ntp_offset_ms'],
               c=['red' if x > 0 else 'blue' for x in ntp_df['ntp_offset_ms']],
               s=30, alpha=0.6, label='NTP Measurements')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Perfect Sync')
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('NTP Offset (ms)')
    ax.set_title(f'{platform}: NTP Offset Over Time (Red=Positive, Blue=Negative)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = csv_path.parent / f"{platform.lower().replace(' ', '_')}_ntp_flickering.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

def main():
    """Analyze all platforms."""

    platforms = [
        ("ARES comp-11", Path("results/experiment-10/ares-11/chronotick_client_validation_20251022_192420.csv")),
        ("ARES comp-12", Path("results/experiment-10/ares-12/chronotick_client_validation_20251022_192443.csv")),
        ("Homelab", Path("results/experiment-10/homelab/chronotick_client_validation_20251022_192238.csv")),
    ]

    for platform, csv_path in platforms:
        if csv_path.exists():
            analyze_bimodal_pattern(platform, csv_path)

if __name__ == "__main__":
    main()
