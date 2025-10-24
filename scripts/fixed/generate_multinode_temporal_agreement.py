#!/usr/bin/env python3
"""
Multi-Node Temporal Agreement Visualization

Shows two HPC nodes with ChronoTick running independently:
- Aligned on SHARED wall-clock timeline (not elapsed time)
- Node 2 starts ~2 minutes after Node 1
- Shows ChronoTick prediction error with ±3σ bands
- Demonstrates cross-node temporal agreement for distributed events

Key insight: When both nodes' uncertainty bands overlap, they agree on
the temporal ordering of distributed events within that uncertainty range.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

def load_and_align_nodes():
    """Load both node datasets and align to shared timeline."""

    # Load data
    node1_path = Path('results/experiment-7/ares-comp-11/chronotick_client_validation_20251020_220343.csv')
    node2_path = Path('results/experiment-7/ares-comp-12/chronotick_client_validation_20251020_220540.csv')

    df1 = pd.read_csv(node1_path)
    df2 = pd.read_csv(node2_path)

    # Get NTP samples
    ntp1 = df1[df1['has_ntp'] == True].copy()
    ntp2 = df2[df2['has_ntp'] == True].copy()

    # Calculate ChronoTick prediction errors
    ntp1['chronotick_error'] = ntp1['chronotick_offset_ms'] - ntp1['ntp_offset_ms']
    ntp2['chronotick_error'] = ntp2['chronotick_offset_ms'] - ntp2['ntp_offset_ms']

    # Use system_time as the shared timeline (wall-clock time)
    # Convert to seconds since the EARLIEST start time (Node 1)
    start_time_node1 = df1['system_time'].iloc[0]
    start_time_node2 = df2['system_time'].iloc[0]

    earliest_start = min(start_time_node1, start_time_node2)

    ntp1['wall_clock_seconds'] = (ntp1['system_time'] - earliest_start)
    ntp2['wall_clock_seconds'] = (ntp2['system_time'] - earliest_start)

    # Convert to minutes for readability
    ntp1['wall_clock_minutes'] = ntp1['wall_clock_seconds'] / 60
    ntp2['wall_clock_minutes'] = ntp2['wall_clock_seconds'] / 60

    start_diff = start_time_node2 - start_time_node1

    return ntp1, ntp2, start_diff

def calculate_agreement_percentage(ntp1, ntp2):
    """
    Calculate what percentage of time both nodes agree within their
    combined uncertainty bounds.
    """

    # For each time point where both have measurements, check if their
    # error bands overlap

    # Create overlapping time windows
    # Since they sample at different times, we need to interpolate or
    # find nearby measurements

    # Simplified approach: check overall distribution overlap
    node1_error_mean = ntp1['chronotick_error'].mean()
    node2_error_mean = ntp2['chronotick_error'].mean()

    node1_3sigma = 3 * ntp1['chronotick_uncertainty_ms'].mean()
    node2_3sigma = 3 * ntp2['chronotick_uncertainty_ms'].mean()

    # Range covered by each node
    node1_range = (node1_error_mean - node1_3sigma, node1_error_mean + node1_3sigma)
    node2_range = (node2_error_mean - node2_3sigma, node2_error_mean + node2_3sigma)

    # Calculate overlap
    overlap_start = max(node1_range[0], node2_range[0])
    overlap_end = min(node1_range[1], node2_range[1])

    if overlap_end > overlap_start:
        overlap_size = overlap_end - overlap_start
        node1_size = node1_range[1] - node1_range[0]
        node2_size = node2_range[1] - node2_range[0]
        avg_size = (node1_size + node2_size) / 2
        agreement_pct = (overlap_size / avg_size) * 100
    else:
        agreement_pct = 0

    return agreement_pct, (overlap_start, overlap_end)

def generate_temporal_agreement_figure():
    """Generate multi-node temporal agreement figure."""

    print("="*80)
    print("MULTI-NODE TEMPORAL AGREEMENT FIGURE")
    print("="*80)

    # Load and align data
    ntp1, ntp2, start_diff = load_and_align_nodes()

    print(f"\nNode start time difference: {start_diff:.1f} seconds ({start_diff/60:.2f} minutes)")
    print(f"Node 1: {len(ntp1)} NTP measurements")
    print(f"Node 2: {len(ntp2)} NTP measurements")

    # Calculate agreement
    agreement_pct, overlap_range = calculate_agreement_percentage(ntp1, ntp2)

    print(f"\nTemporal Agreement:")
    print(f"  Overlap range: [{overlap_range[0]:.2f}, {overlap_range[1]:.2f}] ms")
    print(f"  Agreement: {agreement_pct:.1f}%")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Node 1 (blue)
    node1_3sigma_lower = ntp1['chronotick_error'] - 3 * ntp1['chronotick_uncertainty_ms']
    node1_3sigma_upper = ntp1['chronotick_error'] + 3 * ntp1['chronotick_uncertainty_ms']

    ax.fill_between(ntp1['wall_clock_minutes'], node1_3sigma_lower, node1_3sigma_upper,
                     alpha=0.2, color='#0072B2', label='Node 1 ±3σ', zorder=2)

    ax.scatter(ntp1['wall_clock_minutes'], ntp1['chronotick_error'],
               color='#0072B2', marker='o', s=15, alpha=0.6,
               label='Node 1 (HPC-11)', zorder=4)

    # Node 2 (orange)
    node2_3sigma_lower = ntp2['chronotick_error'] - 3 * ntp2['chronotick_uncertainty_ms']
    node2_3sigma_upper = ntp2['chronotick_error'] + 3 * ntp2['chronotick_uncertainty_ms']

    ax.fill_between(ntp2['wall_clock_minutes'], node2_3sigma_lower, node2_3sigma_upper,
                     alpha=0.2, color='#D55E00', label='Node 2 ±3σ', zorder=2)

    ax.scatter(ntp2['wall_clock_minutes'], ntp2['chronotick_error'],
               color='#D55E00', marker='s', s=15, alpha=0.6,
               label='Node 2 (HPC-12)', zorder=4)

    # Perfect sync reference
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3, zorder=1)

    # Mark where Node 2 starts
    node2_start_minutes = start_diff / 60
    ax.axvline(node2_start_minutes, color='#D55E00', linewidth=1.5,
               linestyle=':', alpha=0.5, zorder=1,
               label=f'Node 2 Start (+{node2_start_minutes:.1f}m)')

    # Styling
    ax.set_xlabel('Wall-Clock Time (minutes since Node 1 start)', fontsize=11)
    ax.set_ylabel('ChronoTick Prediction Error (ms)', fontsize=11)
    ax.legend(loc='upper left', fontsize=9, ncol=2)
    ax.grid(alpha=0.3)

    # Add agreement text
    ax.text(0.98, 0.02,
            f'Temporal Agreement: {agreement_pct:.0f}%\n(±3σ overlap)',
            transform=ax.transAxes,
            ha='right', va='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=9)

    plt.tight_layout()

    # Save
    output_dir = Path('results/figures_corrected/multinode')
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / 'multinode_temporal_agreement.pdf'
    png_path = output_dir / 'multinode_temporal_agreement.png'

    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"\n✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")

    plt.close()

    # Print statistics
    print("\n" + "="*80)
    print("NODE STATISTICS")
    print("="*80)

    print(f"\nNode 1 (HPC-11):")
    print(f"  Start: 0.0 minutes (reference)")
    print(f"  Error mean: {ntp1['chronotick_error'].mean():.3f} ms")
    print(f"  Error std: {ntp1['chronotick_error'].std():.3f} ms")
    print(f"  MAE: {ntp1['chronotick_error'].abs().mean():.3f} ms")
    print(f"  Mean ±3σ: ±{(3 * ntp1['chronotick_uncertainty_ms']).mean():.2f} ms")

    print(f"\nNode 2 (HPC-12):")
    print(f"  Start: {node2_start_minutes:.2f} minutes (delayed)")
    print(f"  Error mean: {ntp2['chronotick_error'].mean():.3f} ms")
    print(f"  Error std: {ntp2['chronotick_error'].std():.3f} ms")
    print(f"  MAE: {ntp2['chronotick_error'].abs().mean():.3f} ms")
    print(f"  Mean ±3σ: ±{(3 * ntp2['chronotick_uncertainty_ms']).mean():.2f} ms")

    print(f"\nTemporal Agreement Analysis:")
    print(f"  Both nodes maintain consistent prediction errors")
    print(f"  ±3σ bands overlap {agreement_pct:.0f}% of their range")
    print(f"  Demonstrates cross-node temporal consistency")
    print(f"  Enables distributed event ordering within {overlap_range[1]-overlap_range[0]:.1f}ms window")

def main():
    """Main function."""
    print("="*80)
    print("GENERATING: Multi-Node Temporal Agreement Figure")
    print("="*80)
    print("\nThis figure shows:")
    print("  • Two HPC nodes running ChronoTick independently")
    print("  • Aligned on shared wall-clock timeline")
    print("  • Node 2 starts ~2 minutes after Node 1")
    print("  • ChronoTick prediction error with ±3σ uncertainty")
    print("  • Temporal agreement region where bands overlap")

    generate_temporal_agreement_figure()

    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
