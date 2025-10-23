#!/usr/bin/env python3
"""
Generate Figure 1.7: Multi-Node Agreement and Uncertainty Quantification

Shows that ChronoTick predictions and uncertainties are consistent across
different HPC nodes without per-node calibration (zero-shot deployment).

Demonstrates:
1. Node-to-node prediction agreement
2. Consistent uncertainty quantification
3. Validates contribution #2 (uncertainty enables distributed algorithms)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_and_process(csv_path, node_name):
    """Load dataset and calculate metrics."""
    df = pd.read_csv(csv_path)
    df_ntp = df[df['has_ntp'] == True].copy()

    # Calculate errors and hours
    df_ntp['chronotick_error_ms'] = df_ntp['chronotick_offset_ms'] - df_ntp['ntp_offset_ms']
    df_ntp['elapsed_hours'] = df_ntp['elapsed_seconds'] / 3600

    # Metrics
    mae = df_ntp['chronotick_error_ms'].abs().mean()

    # Coverage statistics
    within_1sigma = (df_ntp['chronotick_error_ms'].abs() <= df_ntp['chronotick_uncertainty_ms']).mean() * 100
    within_2sigma = (df_ntp['chronotick_error_ms'].abs() <= 2 * df_ntp['chronotick_uncertainty_ms']).mean() * 100
    within_3sigma = (df_ntp['chronotick_error_ms'].abs() <= 3 * df_ntp['chronotick_uncertainty_ms']).mean() * 100

    return df_ntp, mae, within_1sigma, within_2sigma, within_3sigma

def main():
    """Generate multi-node agreement figure."""

    # Experiment-7 ARES nodes (both 8-hour runs, 10-min internal sync)
    datasets = {
        'HPC Node 1 (comp-11)': Path('results/experiment-7/ares-comp-11/chronotick_client_validation_20251020_220343.csv'),
        'HPC Node 2 (comp-12)': Path('results/experiment-7/ares-comp-12/chronotick_client_validation_20251020_220540.csv')
    }

    # Process datasets
    data = {}
    metrics = {}
    for node, path in datasets.items():
        df_ntp, mae, cov_1s, cov_2s, cov_3s = load_and_process(path, node)
        data[node] = df_ntp
        metrics[node] = {
            'mae': mae,
            'coverage_1sigma': cov_1s,
            'coverage_2sigma': cov_2s,
            'coverage_3sigma': cov_3s
        }

    # Create figure with 3 subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.subplots_adjust(hspace=0.35)

    node_colors = {
        'HPC Node 1 (comp-11)': '#0072B2',
        'HPC Node 2 (comp-12)': '#D55E00'
    }

    # Plot 1: Both nodes' prediction errors with uncertainty bands
    ax1 = axes[0]
    for node, df_ntp in data.items():
        color = node_colors[node]

        # Plot error
        ax1.scatter(df_ntp['elapsed_hours'], df_ntp['chronotick_error_ms'],
                   label=f'{node} Error', color=color, s=15, alpha=0.6, marker='o')

        # Plot uncertainty bands
        ax1.fill_between(df_ntp['elapsed_hours'],
                        df_ntp['chronotick_error_ms'] - df_ntp['chronotick_uncertainty_ms'],
                        df_ntp['chronotick_error_ms'] + df_ntp['chronotick_uncertainty_ms'],
                        alpha=0.15, color=color, label=f'{node} ±1σ')

    ax1.axhline(0, color='black', linewidth=1, linestyle='--', alpha=0.5, label='Perfect Prediction')
    ax1.set_xlabel('Time (hours)', fontsize=11)
    ax1.set_ylabel('Prediction Error from NTP (ms)', fontsize=11)
    ax1.set_title('Multi-Node Prediction Agreement with Uncertainty Quantification', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9, ncol=2, framealpha=0.9)
    ax1.grid(alpha=0.3)

    # Plot 2: Uncertainty magnitude comparison
    ax2 = axes[1]
    for node, df_ntp in data.items():
        color = node_colors[node]
        ax2.scatter(df_ntp['elapsed_hours'], df_ntp['chronotick_uncertainty_ms'],
                   label=f'{node} Uncertainty', color=color, s=15, alpha=0.6)

    ax2.set_xlabel('Time (hours)', fontsize=11)
    ax2.set_ylabel('Prediction Uncertainty ±1σ (ms)', fontsize=11)
    ax2.set_title('Consistent Uncertainty Quantification Across Nodes', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9, framealpha=0.9)
    ax2.grid(alpha=0.3)

    # Plot 3: MAE and Coverage comparison (bar charts side-by-side)
    ax3 = axes[2]

    nodes = list(metrics.keys())
    maes = [metrics[n]['mae'] for n in nodes]
    cov_2sigma = [metrics[n]['coverage_2sigma'] for n in nodes]

    x = np.arange(len(nodes))
    width = 0.35

    # MAE bars
    bars1 = ax3.bar(x - width/2, maes, width, label='MAE (ms)', color='#56B4E9', alpha=0.8, edgecolor='black')

    # Create twin axis for coverage percentage
    ax3_twin = ax3.twinx()
    bars2 = ax3_twin.bar(x + width/2, cov_2sigma, width, label='2σ Coverage (%)', color='#009E73', alpha=0.8, edgecolor='black')

    # Labels for MAE bars
    for bar, val in zip(bars1, maes):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Labels for coverage bars
    for bar, val in zip(bars2, cov_2sigma):
        height = bar.get_height()
        ax3_twin.text(bar.get_x() + bar.get_width()/2., height + 1,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax3.set_ylabel('Mean Absolute Error (ms)', fontsize=11)
    ax3_twin.set_ylabel('Uncertainty Coverage (%)', fontsize=11)
    ax3.set_xlabel('HPC Cluster Node', fontsize=11)
    ax3.set_title('Zero-Shot Deployment: Consistent Performance Without Per-Node Calibration', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([n.split('(')[1].replace(')', '') for n in nodes], fontsize=11)
    ax3.set_ylim([0, max(maes) * 1.4])
    ax3_twin.set_ylim([0, 100])

    # Combine legends
    lines1, labels1 = ax3.get_legend_handles_labels()
    lines2, labels2 = ax3_twin.get_legend_handles_labels()
    ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10, framealpha=0.9)

    ax3.grid(axis='y', alpha=0.3)

    # Save figure
    output_dir = Path('results/figures/7_multinode')
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / '7_multinode_agreement.pdf'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to: {output_path}")

    output_path_png = output_dir / '7_multinode_agreement.png'
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight')
    print(f"PNG saved to: {output_path_png}")

    plt.close()

    # Print statistics
    print("\n" + "="*80)
    print("MULTI-NODE AGREEMENT RESULTS (Experiment-7)")
    print("="*80)
    for node, m in metrics.items():
        print(f"\n{node}:")
        print(f"  MAE:             {m['mae']:.3f} ms")
        print(f"  1σ Coverage:     {m['coverage_1sigma']:.1f}%")
        print(f"  2σ Coverage:     {m['coverage_2sigma']:.1f}%")
        print(f"  3σ Coverage:     {m['coverage_3sigma']:.1f}%")

    # Calculate agreement metrics
    mae_diff = abs(metrics[nodes[0]]['mae'] - metrics[nodes[1]]['mae'])
    cov_diff = abs(metrics[nodes[0]]['coverage_2sigma'] - metrics[nodes[1]]['coverage_2sigma'])

    print(f"\nNode Agreement Metrics:")
    print(f"  MAE difference:      {mae_diff:.3f} ms ({(mae_diff/metrics[nodes[0]]['mae'])*100:.1f}%)")
    print(f"  Coverage difference: {cov_diff:.1f} percentage points")

if __name__ == '__main__':
    main()
