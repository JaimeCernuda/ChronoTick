#!/usr/bin/env python3
"""
Multi-Node Temporal Alignment - APPROACH 2

Approach 1: Compare prediction errors when both measure NTP at similar wall-clock times
Approach 2: Compare timestamps when both system clocks show the same time T

This implements APPROACH 2:
- When Node 1's system clock reads T and has NTP ground truth
- Find when Node 2's system clock also reads T (different wall-clock moment)
- Do they agree on what true time it is?
- Uses ALL Node 2 samples (not just NTP), giving much denser coverage

This directly tests: "If both nodes generate a distributed event at system time T,
do they agree on the true timestamp?"
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import timedelta

def generate_approach2_alignment_figure(node1_csv, node2_csv, output_dir):
    """Generate temporal alignment using Approach 2: system time matching."""

    print("="*80)
    print("APPROACH 2: Distributed Event Timestamp Agreement")
    print("="*80)
    print("\nThis approach tests: When both nodes' system clocks show time T,")
    print("do they agree on what true time it is?")
    print("\nKey difference:")
    print("  • Uses ALL Node 2 samples (not just NTP measurements)")
    print("  • Compares at same LOGICAL time (system clock time)")
    print("  • Tests distributed event timestamp consistency")

    # Load data
    df1 = pd.read_csv(node1_csv)
    df2 = pd.read_csv(node2_csv)

    # Node 1: Only NTP samples (for ground truth)
    df1_ntp = df1[df1['has_ntp'] == True].copy()

    # Node 2: ALL samples (NTP and ChronoTick-only)
    df2_all = df2.copy()

    print(f"\n{'='*80}")
    print("DATA COVERAGE")
    print('='*80)
    print(f"Node 1 NTP samples: {len(df1_ntp)} (ground truth)")
    print(f"Node 2 ALL samples: {len(df2_all)} (including {len(df2[df2['has_ntp']==True])} NTP)")
    print(f"  → {len(df2_all)/len(df1_ntp):.1f}x denser sampling on Node 2!")

    # Parse timestamps
    df1_ntp['timestamp'] = pd.to_datetime(df1_ntp['datetime'])
    df2_all['timestamp'] = pd.to_datetime(df2_all['datetime'])

    # Calculate elapsed time from experiment start
    start1 = df1_ntp['timestamp'].iloc[0]
    start2 = df2_all['timestamp'].iloc[0]
    reference_time = min(start1, start2)

    df1_ntp['hours_from_start'] = (df1_ntp['timestamp'] - reference_time).dt.total_seconds() / 3600
    df2_all['hours_from_start'] = (df2_all['timestamp'] - reference_time).dt.total_seconds() / 3600

    start_delay = (start2 - start1).total_seconds() / 60
    print(f"\nNode 2 started {abs(start_delay):.1f} minutes {'after' if start_delay > 0 else 'before'} Node 1")

    # APPROACH 2: Match by system_time
    print(f"\n{'='*80}")
    print("APPROACH 2: SYSTEM TIME MATCHING")
    print('='*80)

    agreements = []
    matched_pairs = []

    for idx1, row1 in df1_ntp.iterrows():
        # Node 1's system time when it measured NTP
        system_time_1 = row1['system_time']
        ntp_offset_1 = row1['ntp_offset_ms']  # Ground truth
        chronotick_offset_1 = row1['chronotick_offset_ms']
        uncertainty_1 = row1['chronotick_uncertainty_ms']

        # Find Node 2 samples where system_time is close to system_time_1
        # Allow 1-second tolerance for matching
        time_diff = np.abs(df2_all['system_time'] - system_time_1)
        close_samples = df2_all[time_diff <= 1.0]

        if len(close_samples) > 0:
            # Get closest match
            closest_idx = time_diff[time_diff <= 1.0].idxmin()
            row2 = df2_all.loc[closest_idx]

            chronotick_offset_2 = row2['chronotick_offset_ms']
            uncertainty_2 = row2['chronotick_uncertainty_ms']

            # Agreement check:
            # Node 1 ground truth: true_time = system_time + ntp_offset
            # Node 2 prediction: true_time = system_time + chronotick_offset
            # Difference: |ntp_offset_1 - chronotick_offset_2|
            diff = abs(ntp_offset_1 - chronotick_offset_2)
            combined_3sigma = 3 * (uncertainty_1 + uncertainty_2)

            agrees = diff <= combined_3sigma
            agreements.append(agrees)

            # Store for visualization
            matched_pairs.append({
                'hours': row1['hours_from_start'],
                'system_time': system_time_1,
                'node1_ntp_offset': ntp_offset_1,
                'node2_chronotick_offset': chronotick_offset_2,
                'difference': diff,
                'combined_3sigma': combined_3sigma,
                'agrees': agrees,
                'node2_has_ntp': row2['has_ntp']
            })

    agreement_rate = sum(agreements) / len(agreements) * 100 if agreements else 0

    print(f"\nTemporal Agreement (Approach 2): {agreement_rate:.1f}%")
    print(f"Matched pairs: {len(agreements)}")
    print(f"  Node 2 had NTP: {sum([p['node2_has_ntp'] for p in matched_pairs])}")
    print(f"  Node 2 ChronoTick-only: {sum([not p['node2_has_ntp'] for p in matched_pairs])}")

    # Calculate statistics
    differences = [p['difference'] for p in matched_pairs]
    print(f"\nDifference statistics:")
    print(f"  Mean: {np.mean(differences):.3f} ms")
    print(f"  Median: {np.median(differences):.3f} ms")
    print(f"  Std: {np.std(differences):.3f} ms")
    print(f"  Max: {np.max(differences):.3f} ms")

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), height_ratios=[2, 1])

    # Top plot: Timeline view
    # Node 1 ChronoTick error (for reference)
    df1_ntp['error'] = df1_ntp['chronotick_offset_ms'] - df1_ntp['ntp_offset_ms']
    df1_ntp['error_lower'] = df1_ntp['error'] - 3 * df1_ntp['chronotick_uncertainty_ms']
    df1_ntp['error_upper'] = df1_ntp['error'] + 3 * df1_ntp['chronotick_uncertainty_ms']

    ax1.fill_between(df1_ntp['hours_from_start'],
                     df1_ntp['error_lower'],
                     df1_ntp['error_upper'],
                     alpha=0.2, color='#0072B2', label='Node 1 ±3σ', zorder=2)

    ax1.scatter(df1_ntp['hours_from_start'], df1_ntp['error'],
               color='#0072B2', marker='o', s=15, alpha=0.7,
               label='Node 1 Error', zorder=4)

    # Node 2: Show ALL samples, color-code by whether they matched
    matched_hours = [p['hours'] for p in matched_pairs if p['agrees']]
    unmatched_hours = [p['hours'] for p in matched_pairs if not p['agrees']]

    if matched_hours:
        ax1.scatter(matched_hours, [0]*len(matched_hours),
                   color='#009E73', marker='|', s=100, linewidths=2,
                   label=f'Node 2 Agrees ({len(matched_hours)})', zorder=5, alpha=0.7)

    if unmatched_hours:
        ax1.scatter(unmatched_hours, [0]*len(unmatched_hours),
                   color='#E69F00', marker='|', s=100, linewidths=2,
                   label=f'Node 2 Disagrees ({len(unmatched_hours)})', zorder=5, alpha=0.7)

    ax1.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3, zorder=1)

    # Annotation
    ax1.text(0.02, 0.98,
            f'Approach 2 Agreement: {agreement_rate:.1f}%\n'
            f'(when system clocks show same time T)',
            transform=ax1.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    ax1.set_xlabel('Time from Experiment Start (hours)', fontsize=10)
    ax1.set_ylabel('Node 1 ChronoTick Error (ms)', fontsize=10)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_title('Distributed Event Timestamp Agreement (Approach 2)', fontsize=11, fontweight='bold')

    # Bottom plot: Difference distribution
    ax2.hist(differences, bins=30, color='#0072B2', alpha=0.7, edgecolor='black')
    ax2.axvline(np.mean([p['combined_3sigma'] for p in matched_pairs]),
               color='#D55E00', linewidth=2, linestyle='--',
               label=f'Mean Combined ±3σ: {np.mean([p["combined_3sigma"] for p in matched_pairs]):.1f} ms')
    ax2.set_xlabel('|NTP(Node1) - ChronoTick(Node2)| at system_time=T (ms)', fontsize=10)
    ax2.set_ylabel('Count', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_title('Distribution of Timestamp Differences', fontsize=10)

    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "5.12_multinode_APPROACH2.pdf"
    png_path = output_dir / "5.12_multinode_APPROACH2.png"

    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"\n✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")

    plt.close()

    # Comparison summary
    print(f"\n{'='*80}")
    print("APPROACH 2 SUMMARY")
    print('='*80)
    print(f"\nSemantic meaning:")
    print(f"  When both nodes' system clocks show time T,")
    print(f"  they agree {agreement_rate:.1f}% of the time on true timestamp")
    print(f"  (within combined ±3σ uncertainty)")
    print(f"\nThis directly tests distributed event timestamp consistency:")
    print(f"  • Node 1: 'At system_time=T, true_time is T+{np.mean([p['node1_ntp_offset'] for p in matched_pairs]):.2f}ms'")
    print(f"  • Node 2: 'At system_time=T, true_time is T+{np.mean([p['node2_chronotick_offset'] for p in matched_pairs]):.2f}ms'")
    print(f"  • Mean difference: {np.mean(differences):.3f} ms")

def main():
    """Generate Approach 2 multi-node temporal alignment figure."""

    print("="*80)
    print("MULTI-NODE TEMPORAL ALIGNMENT - APPROACH 2")
    print("="*80)
    print("\nAPPROACH 1 (previous): Wall-clock time matching")
    print("  • Compare when both measure NTP at similar wall-clock times")
    print("  • Only uses NTP measurement moments")
    print("  • Sparse sampling")
    print("\nAPPROACH 2 (this script): System clock time matching")
    print("  • Compare when both system clocks show same time T")
    print("  • Uses ALL Node 2 samples (not just NTP)")
    print("  • Dense sampling - tests distributed event timestamps")
    print("  • Asks: 'Do both nodes agree on true time when generating events?'")

    # Experiment-7 HPC nodes
    node1_csv = Path("results/experiment-7/ares-comp-11/chronotick_client_validation_20251020_220343.csv")
    node2_csv = Path("results/experiment-7/ares-comp-12/chronotick_client_validation_20251020_220540.csv")
    output_dir = Path("results/figures/5/experiment-7")

    if node1_csv.exists() and node2_csv.exists():
        generate_approach2_alignment_figure(node1_csv, node2_csv, output_dir)
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
    print("  • 5.12_multinode_APPROACH2.pdf")
    print("  • 5.12_multinode_APPROACH2.png")

if __name__ == "__main__":
    main()
