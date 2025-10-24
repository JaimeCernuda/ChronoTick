#!/usr/bin/env python3
"""
Multi-Node Temporal Alignment Figure

Shows two HPC nodes running ChronoTick independently on a shared timeline.
Demonstrates temporal agreement between nodes when they measure at similar times.

Key features:
- Shared absolute timeline (Node 2 starts later, appears later)
- ChronoTick prediction error for both nodes
- ±3σ uncertainty bands
- Agreement rate annotation
- Paper color scheme
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import timedelta

def generate_multinode_alignment_figure(node1_csv, node2_csv, output_dir):
    """Generate temporal alignment figure for two nodes."""

    print("="*80)
    print("MULTI-NODE TEMPORAL ALIGNMENT FIGURE")
    print("="*80)

    # Load data
    df1 = pd.read_csv(node1_csv)
    df2 = pd.read_csv(node2_csv)

    df1_ntp = df1[df1['has_ntp'] == True].copy()
    df2_ntp = df2[df2['has_ntp'] == True].copy()

    print(f"\nNode 1: {len(df1_ntp)} NTP samples")
    print(f"Node 2: {len(df2_ntp)} NTP samples")

    # Parse timestamps
    df1_ntp['timestamp'] = pd.to_datetime(df1_ntp['datetime'])
    df2_ntp['timestamp'] = pd.to_datetime(df2_ntp['datetime'])

    # Find start times
    start1 = df1_ntp['timestamp'].iloc[0]
    start2 = df2_ntp['timestamp'].iloc[0]

    # Use earliest start as reference (t=0)
    reference_time = min(start1, start2)

    # Convert to hours from reference
    df1_ntp['hours_from_start'] = (df1_ntp['timestamp'] - reference_time).dt.total_seconds() / 3600
    df2_ntp['hours_from_start'] = (df2_ntp['timestamp'] - reference_time).dt.total_seconds() / 3600

    start_delay = (start2 - start1).total_seconds() / 60  # minutes
    print(f"\nNode 2 started {abs(start_delay):.1f} minutes {'after' if start_delay > 0 else 'before'} Node 1")

    # Calculate ChronoTick prediction errors (offset from NTP truth)
    df1_ntp['error'] = df1_ntp['chronotick_offset_ms'] - df1_ntp['ntp_offset_ms']
    df2_ntp['error'] = df2_ntp['chronotick_offset_ms'] - df2_ntp['ntp_offset_ms']

    # Calculate ±3σ bounds
    df1_ntp['error_lower'] = df1_ntp['error'] - 3 * df1_ntp['chronotick_uncertainty_ms']
    df1_ntp['error_upper'] = df1_ntp['error'] + 3 * df1_ntp['chronotick_uncertainty_ms']
    df2_ntp['error_lower'] = df2_ntp['error'] - 3 * df2_ntp['chronotick_uncertainty_ms']
    df2_ntp['error_upper'] = df2_ntp['error'] + 3 * df2_ntp['chronotick_uncertainty_ms']

    print(f"\nNode 1 MAE: {df1_ntp['error'].abs().mean():.3f} ms")
    print(f"Node 2 MAE: {df2_ntp['error'].abs().mean():.3f} ms")

    # Calculate agreement rate
    agreements = []
    for idx1, row1 in df1_ntp.iterrows():
        t1 = row1['timestamp']
        time_diff = (df2_ntp['timestamp'] - t1).abs()
        close_measurements = df2_ntp[time_diff <= pd.Timedelta(seconds=60)]

        if len(close_measurements) > 0:
            closest_idx = time_diff[time_diff <= pd.Timedelta(seconds=60)].idxmin()
            row2 = df2_ntp.loc[closest_idx]

            diff = abs(row1['error'] - row2['error'])
            combined_3sigma = 3 * (row1['chronotick_uncertainty_ms'] + row2['chronotick_uncertainty_ms'])

            agreements.append(diff <= combined_3sigma)

    agreement_rate = sum(agreements) / len(agreements) * 100 if agreements else 0
    print(f"\nTemporal Agreement: {agreement_rate:.1f}% (within combined ±3σ)")
    print(f"Co-occurring measurements: {len(agreements)}")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))

    # Node 1 (blue)
    ax.fill_between(df1_ntp['hours_from_start'],
                     df1_ntp['error_lower'],
                     df1_ntp['error_upper'],
                     alpha=0.2, color='#0072B2', label='HPC Node 1 ±3σ', zorder=2)

    ax.scatter(df1_ntp['hours_from_start'], df1_ntp['error'],
               color='#0072B2', marker='o', s=15, alpha=0.7,
               label='HPC Node 1', zorder=4)

    # Node 2 (orange)
    ax.fill_between(df2_ntp['hours_from_start'],
                     df2_ntp['error_lower'],
                     df2_ntp['error_upper'],
                     alpha=0.2, color='#D55E00', label='HPC Node 2 ±3σ', zorder=2)

    ax.scatter(df2_ntp['hours_from_start'], df2_ntp['error'],
               color='#D55E00', marker='s', s=15, alpha=0.7,
               label='HPC Node 2', zorder=4)

    # Perfect sync reference
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3, zorder=1)

    # Mark start of Node 2 if delayed
    if abs(start_delay) > 0.5:  # More than 30 seconds
        node2_start_hours = df2_ntp['hours_from_start'].iloc[0]
        ax.axvline(node2_start_hours, color='#D55E00', linewidth=1,
                   linestyle=':', alpha=0.5, zorder=1)
        ax.text(node2_start_hours + 0.1, ax.get_ylim()[1] * 0.9,
                f'Node 2\nstarts\n({abs(start_delay):.1f} min later)',
                fontsize=8, color='#D55E00', alpha=0.7)

    # Add agreement rate annotation
    ax.text(0.02, 0.98,
            f'Temporal Agreement: {agreement_rate:.1f}%\n(within combined ±3σ)',
            transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Styling
    ax.set_xlabel('Time from Experiment Start (hours)', fontsize=10)
    ax.set_ylabel('ChronoTick Prediction Error (ms)', fontsize=10)
    ax.legend(loc='upper right', fontsize=9, ncol=2)
    ax.grid(alpha=0.3)

    # Set x-axis
    max_hours = max(df1_ntp['hours_from_start'].max(), df2_ntp['hours_from_start'].max())
    ax.set_xlim(0, np.ceil(max_hours))

    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "5.12_multinode_temporal_alignment.pdf"
    png_path = output_dir / "5.12_multinode_temporal_alignment.png"

    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"\n✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")

    plt.close()

    # Print summary stats
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print('='*80)
    print(f"\nNode 1 (HPC comp-11):")
    print(f"  Start time: {start1}")
    print(f"  Duration: {df1_ntp['hours_from_start'].max():.2f} hours")
    print(f"  MAE: {df1_ntp['error'].abs().mean():.3f} ms")
    print(f"  Mean 3σ: {3 * df1_ntp['chronotick_uncertainty_ms'].mean():.3f} ms")

    print(f"\nNode 2 (HPC comp-12):")
    print(f"  Start time: {start2}")
    print(f"  Duration: {df2_ntp['hours_from_start'].max():.2f} hours")
    print(f"  MAE: {df2_ntp['error'].abs().mean():.3f} ms")
    print(f"  Mean 3σ: {3 * df2_ntp['chronotick_uncertainty_ms'].mean():.3f} ms")

    print(f"\nTemporal Agreement:")
    print(f"  When both nodes measure at similar times,")
    print(f"  they agree {agreement_rate:.1f}% of the time")
    print(f"  (within their combined ±3σ uncertainty bounds)")
    print(f"  Co-occurring measurements: {len(agreements)}")

def main():
    """Generate multi-node temporal alignment figure."""

    print("="*80)
    print("MULTI-NODE TEMPORAL ALIGNMENT - Experiment-7 HPC Nodes")
    print("="*80)
    print("\nThis figure shows:")
    print("  • Two HPC nodes running ChronoTick independently")
    print("  • Shared timeline (Node 2 starts ~2 minutes later)")
    print("  • ChronoTick prediction errors with ±3σ bounds")
    print("  • Temporal agreement rate when both measure simultaneously")

    # Experiment-7 HPC nodes
    node1_csv = Path("results/experiment-7/ares-comp-11/chronotick_client_validation_20251020_220343.csv")
    node2_csv = Path("results/experiment-7/ares-comp-12/chronotick_client_validation_20251020_220540.csv")
    output_dir = Path("results/figures/5/experiment-7")

    if node1_csv.exists() and node2_csv.exists():
        generate_multinode_alignment_figure(node1_csv, node2_csv, output_dir)
    else:
        print("\n⚠️  Dataset files not found!")
        if not node1_csv.exists():
            print(f"  Missing: {node1_csv}")
        if not node2_csv.exists():
            print(f"  Missing: {node2_csv}")
        return

    print("\n" + "="*80)
    print("FIGURE COMPLETE")
    print("="*80)
    print(f"\nOutput: results/figures/5/experiment-7/")
    print("  • 5.12_multinode_temporal_alignment.pdf")
    print("  • 5.12_multinode_temporal_alignment.png")

if __name__ == "__main__":
    main()
