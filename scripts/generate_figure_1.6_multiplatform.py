#!/usr/bin/env python3
"""
Generate Figure 1.6: Multi-Platform Consistency

Shows ChronoTick performance across three diverse platforms:
- Cloud Media Server (Homelab - synchronized)
- HPC Cluster (ARES comp-11)
- Workstation (WSL2 - this machine)

Layout: 3 offset plots (top row) + 1 MAE comparison (bottom)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_and_process(csv_path, platform_name):
    """Load dataset and calculate metrics."""
    df = pd.read_csv(csv_path)
    df_ntp = df[df['has_ntp'] == True].copy()

    # Calculate errors
    df_ntp['chronotick_error_ms'] = df_ntp['chronotick_offset_ms'] - df_ntp['ntp_offset_ms']
    df_ntp['elapsed_hours'] = df_ntp['elapsed_seconds'] / 3600

    # Metrics
    mae = df_ntp['chronotick_error_ms'].abs().mean()
    rmse = np.sqrt((df_ntp['chronotick_error_ms'] ** 2).mean())
    max_error = df_ntp['chronotick_error_ms'].abs().max()

    return df_ntp, mae, rmse, max_error

def main():
    """Generate multi-platform consistency figure."""

    # Three diverse platform datasets (all 8-hour runs)
    datasets = {
        'Cloud Media Server': Path('results/experiment-3/homelab/data.csv'),  # Synchronized, 1.678 ms MAE
        'HPC Cluster': Path('results/experiment-7/ares-comp-11/chronotick_client_validation_20251020_220343.csv'),  # 4.941 ms MAE
        'Workstation (WSL2)': Path('results/experiment-1/wsl2/chronotick_client_validation_20251018_020105.csv')  # 127.440 ms MAE (virtualized)
    }

    # Process all datasets
    data = {}
    maes = {}
    for platform, path in datasets.items():
        df_ntp, mae, rmse, max_err = load_and_process(path, platform)
        data[platform] = df_ntp
        maes[platform] = mae

    # Create figure with 2x2 grid
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Top row: 3 offset plots
    platform_colors = {
        'Cloud Media Server': '#E69F00',
        'HPC Cluster': '#56B4E9',
        'Workstation (WSL2)': '#009E73'
    }

    for idx, (platform, df_ntp) in enumerate(data.items()):
        ax = fig.add_subplot(gs[0, idx])

        # Plot system clock offset and ChronoTick offset
        ax.scatter(df_ntp['elapsed_hours'], df_ntp['ntp_offset_ms'],
                  label='System Clock Offset', color='#CC79A7', s=20, alpha=0.6, marker='s')

        ax.scatter(df_ntp['elapsed_hours'], df_ntp['chronotick_offset_ms'],
                  label='ChronoTick Offset', color=platform_colors[platform], s=20, alpha=0.7, marker='o')

        # Zero reference line
        ax.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5, label='Perfect Sync')

        ax.set_xlabel('Time (hours)', fontsize=10)
        ax.set_ylabel('Offset from NTP (ms)', fontsize=10)
        ax.set_title(platform, fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8, framealpha=0.9)
        ax.grid(alpha=0.3)

    # Bottom: MAE comparison bar chart (spans all 3 columns)
    ax_mae = fig.add_subplot(gs[1, :])

    platforms = list(maes.keys())
    mae_values = list(maes.values())
    colors = [platform_colors[p] for p in platforms]

    x = np.arange(len(platforms))
    bars = ax_mae.bar(x, mae_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, val in zip(bars, mae_values):
        height = bar.get_height()
        ax_mae.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{val:.3f} ms', ha='center', va='bottom',
                   fontsize=11, fontweight='bold')

    ax_mae.set_ylabel('Mean Absolute Error (ms)', fontsize=12)
    ax_mae.set_xlabel('Platform', fontsize=12)
    ax_mae.set_title('Platform-Independent Performance (MAE Comparison)', fontsize=13, fontweight='bold')
    ax_mae.set_xticks(x)
    ax_mae.set_xticklabels(platforms, fontsize=11)
    ax_mae.set_ylim([0, max(mae_values) * 1.3])
    ax_mae.grid(axis='y', alpha=0.3)

    # Save figure
    output_dir = Path('results/figures/6_multiplatform')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / '6_multiplatform_consistency.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")

    output_path_png = output_dir / '6_multiplatform_consistency.png'
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"PNG saved to: {output_path_png}")

    plt.close()

    # Print statistics
    print("\n" + "="*80)
    print("MULTI-PLATFORM CONSISTENCY RESULTS (Experiment-7)")
    print("="*80)
    for platform, mae in maes.items():
        print(f"{platform:25} MAE: {mae:7.3f} ms")

    avg_mae = np.mean(list(maes.values()))
    std_mae = np.std(list(maes.values()))
    print(f"\nAverage MAE: {avg_mae:.3f} ms")
    print(f"Std Dev:     {std_mae:.3f} ms")
    print(f"Variation:   {(std_mae/avg_mae)*100:.1f}%")

if __name__ == '__main__':
    main()
