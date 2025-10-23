#!/usr/bin/env python3
"""
Generate Figure 1.5: Synchronization Frequency Impact on MAE

Shows MAE for different ChronoTick NTP sync intervals:
- 2-minute (baseline)
- 10-minute (sparse)
- 1-hour (very sparse) - PLACEHOLDER
- Adaptable (future work) - PLACEHOLDER
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def calculate_mae(csv_path):
    """Calculate MAE for a dataset."""
    df = pd.read_csv(csv_path)
    df_ntp = df[df['has_ntp'] == True].copy()
    df_ntp['error_ms'] = df_ntp['chronotick_offset_ms'] - df_ntp['ntp_offset_ms']
    return df_ntp['error_ms'].abs().mean()

def main():
    """Generate synchronization frequency comparison figure."""

    # Dataset paths
    path_2min = Path('results/experiment-7/homelab/chronotick_client_validation_20251020_221631.csv')
    path_10min = Path('results/experiment-7/ares-comp-11/chronotick_client_validation_20251020_220343.csv')

    # Calculate real MAEs
    mae_2min = calculate_mae(path_2min)
    mae_10min = calculate_mae(path_10min)

    # PLACEHOLDERS for missing data
    mae_1hour = 15.0  # RED PLACEHOLDER - needs real data
    mae_adaptable = 8.0  # RED PLACEHOLDER - future work

    # Data
    categories = ['2-minute\nbaseline', '10-minute\nsparse', '1-hour\nvery sparse', 'Adaptable\n(future)']
    maes = [mae_2min, mae_10min, mae_1hour, mae_adaptable]
    colors = ['#0072B2', '#009E73', '#FF0000', '#FF0000']  # Blue, Green, RED placeholders

    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    x = np.arange(len(categories))
    bars = ax.bar(x, maes, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for i, (bar, val, color) in enumerate(zip(bars, maes, colors)):
        height = bar.get_height()
        label_text = f'{val:.3f}' if i < 2 else f'{val:.1f}\n[PLACEHOLDER]'
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                label_text, ha='center', va='bottom',
                fontsize=10, fontweight='bold',
                color='red' if color == '#FF0000' else 'black')

    ax.set_ylabel('Mean Absolute Error (ms)', fontsize=12)
    ax.set_xlabel('ChronoTick Internal NTP Synchronization Frequency', fontsize=12)
    ax.set_title('Impact of Synchronization Frequency on Prediction Accuracy', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim([0, max(maes) * 1.2])
    ax.grid(axis='y', alpha=0.3)

    # Add note about placeholders
    ax.text(0.98, 0.98, 'Red bars = PLACEHOLDER DATA\nNeeds real experiment',
            transform=ax.transAxes, fontsize=9, color='red',
            ha='right', va='top',
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='red', alpha=0.8))

    # Save figure
    output_dir = Path('results/figures/5_frequency')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / '5.1_ntp_frequency_comparison.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")

    output_path_png = output_dir / '5.1_ntp_frequency_comparison.png'
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"PNG saved to: {output_path_png}")

    plt.close()

    # Print statistics
    print("\n" + "="*80)
    print("SYNCHRONIZATION FREQUENCY RESULTS")
    print("="*80)
    print(f"2-minute baseline:   {mae_2min:.3f} ms (Experiment-7 Homelab)")
    print(f"10-minute sparse:    {mae_10min:.3f} ms (Experiment-7 ARES-11)")
    print(f"1-hour very sparse:  {mae_1hour:.1f} ms [PLACEHOLDER - need data]")
    print(f"Adaptable:           {mae_adaptable:.1f} ms [PLACEHOLDER - future work]")

    print("\nDatasets used:")
    print(f"  2-min:  {path_2min}")
    print(f"  10-min: {path_10min}")

if __name__ == '__main__':
    main()
