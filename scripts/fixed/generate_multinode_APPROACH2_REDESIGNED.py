#!/usr/bin/env python3
"""
Multi-Node Temporal Alignment - APPROACH 2 REDESIGNED

This redesign shows BOTH nodes across the entire timeline with clear visualization
of when system_time matches and whether they agree.

Key visualization elements:
1. Both nodes plotted on same timeline
2. Show ALL data for both nodes
3. Highlight the specific moments where system_time matches
4. Color-code agreement vs disagreement
5. Make it crystal clear what's being compared
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def generate_approach2_redesigned(node1_csv, node2_csv, output_dir):
    """Generate redesigned Approach 2 figure showing both nodes clearly."""

    print("="*80)
    print("APPROACH 2 REDESIGNED: Clear Visualization of Both Nodes")
    print("="*80)

    # Load data
    df1 = pd.read_csv(node1_csv)
    df2 = pd.read_csv(node2_csv)

    df1_ntp = df1[df1['has_ntp'] == True].copy()
    df2_all = df2.copy()

    print(f"\nNode 1 NTP samples: {len(df1_ntp)}")
    print(f"Node 2 ALL samples: {len(df2_all)} (including {len(df2[df2['has_ntp']==True])} NTP)")

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

    # APPROACH 2: System time matching
    print(f"\n{'='*80}")
    print("FINDING SYSTEM TIME MATCHES")
    print('='*80)

    matched_pairs = []

    for idx1, row1 in df1_ntp.iterrows():
        system_time_1 = row1['system_time']
        ntp_offset_1 = row1['ntp_offset_ms']
        uncertainty_1 = row1['chronotick_uncertainty_ms']
        hours_1 = row1['hours_from_start']

        # Find Node 2 samples where system_time matches (within 1 second)
        time_diff = np.abs(df2_all['system_time'] - system_time_1)
        close_samples = df2_all[time_diff <= 1.0]

        if len(close_samples) > 0:
            closest_idx = time_diff[time_diff <= 1.0].idxmin()
            row2 = df2_all.loc[closest_idx]

            chronotick_offset_2 = row2['chronotick_offset_ms']
            uncertainty_2 = row2['chronotick_uncertainty_ms']
            hours_2 = row2['hours_from_start']

            # Agreement calculation
            diff = abs(ntp_offset_1 - chronotick_offset_2)
            combined_3sigma = 3 * (uncertainty_1 + uncertainty_2)
            agrees = diff <= combined_3sigma

            matched_pairs.append({
                'system_time': system_time_1,
                'node1_hours': hours_1,
                'node2_hours': hours_2,
                'node1_offset': ntp_offset_1,  # Ground truth
                'node2_offset': chronotick_offset_2,  # Prediction
                'node1_uncertainty': uncertainty_1,
                'node2_uncertainty': uncertainty_2,
                'difference': diff,
                'combined_3sigma': combined_3sigma,
                'agrees': agrees,
                'node2_has_ntp': row2['has_ntp']
            })

    agreement_rate = sum([p['agrees'] for p in matched_pairs]) / len(matched_pairs) * 100 if matched_pairs else 0

    print(f"\nMatched pairs: {len(matched_pairs)}")
    print(f"Agreement rate: {agreement_rate:.1f}%")
    print(f"  Agree: {sum([p['agrees'] for p in matched_pairs])}")
    print(f"  Disagree: {sum([not p['agrees'] for p in matched_pairs])}")

    # Create comprehensive visualization
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[2, 2, 1], hspace=0.3)

    # Top plot: Node 1 timeline with matches
    ax1 = fig.add_subplot(gs[0])

    # Plot Node 1 NTP offset (ground truth)
    ax1.scatter(df1_ntp['hours_from_start'], df1_ntp['ntp_offset_ms'],
               color='#0072B2', marker='o', s=30, alpha=0.6,
               label='Node 1 NTP Offset (ground truth)', zorder=3)

    # Highlight matched moments
    for pair in matched_pairs:
        color = '#009E73' if pair['agrees'] else '#E69F00'
        marker = 'o' if pair['agrees'] else 'x'
        ax1.scatter(pair['node1_hours'], pair['node1_offset'],
                   color=color, marker=marker, s=100, linewidths=2,
                   zorder=5, edgecolors='black')

    ax1.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3, zorder=1)
    ax1.set_ylabel('Node 1 NTP Offset (ms)', fontsize=11, fontweight='bold')
    ax1.set_title('Node 1: NTP Ground Truth (HPC comp-11)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, np.ceil(max(df1_ntp['hours_from_start'].max(), df2_all['hours_from_start'].max())))

    # Middle plot: Node 2 timeline with matches
    ax2 = fig.add_subplot(gs[1])

    # Plot ALL Node 2 ChronoTick predictions
    ax2.scatter(df2_all['hours_from_start'], df2_all['chronotick_offset_ms'],
               color='#D55E00', marker='s', s=10, alpha=0.3,
               label='Node 2 ChronoTick Predictions (all)', zorder=2)

    # Highlight matched moments
    for pair in matched_pairs:
        color = '#009E73' if pair['agrees'] else '#E69F00'
        marker = 's' if pair['agrees'] else 'x'
        ax2.scatter(pair['node2_hours'], pair['node2_offset'],
                   color=color, marker=marker, s=100, linewidths=2,
                   zorder=5, edgecolors='black')

    ax2.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3, zorder=1)
    ax2.set_ylabel('Node 2 ChronoTick Offset (ms)', fontsize=11, fontweight='bold')
    ax2.set_title('Node 2: ChronoTick Predictions (HPC comp-12)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, np.ceil(max(df1_ntp['hours_from_start'].max(), df2_all['hours_from_start'].max())))

    # Add legend for matched points
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#009E73', edgecolor='black', label=f'Matched & Agree ({sum([p["agrees"] for p in matched_pairs])})'),
        Patch(facecolor='#E69F00', edgecolor='black', label=f'Matched & Disagree ({sum([not p["agrees"] for p in matched_pairs])})')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=9, title='System Time Matches')

    # Bottom plot: Direct comparison of matched pairs
    ax3 = fig.add_subplot(gs[2])

    for i, pair in enumerate(matched_pairs):
        color = '#009E73' if pair['agrees'] else '#E69F00'
        alpha = 0.7 if pair['agrees'] else 1.0
        linewidth = 1 if pair['agrees'] else 2

        # Draw line connecting the two predictions
        ax3.plot([pair['node1_offset'], pair['node2_offset']],
                [1, 2], color=color, alpha=alpha, linewidth=linewidth, zorder=2)

        # Draw points
        ax3.scatter(pair['node1_offset'], 1, color='#0072B2', marker='o', s=50, zorder=3, edgecolors='black')
        ax3.scatter(pair['node2_offset'], 2, color='#D55E00', marker='s', s=50, zorder=3, edgecolors='black')

    ax3.set_ylim(0.5, 2.5)
    ax3.set_yticks([1, 2])
    ax3.set_yticklabels(['Node 1\n(NTP truth)', 'Node 2\n(ChronoTick)'])
    ax3.set_xlabel('Offset from True Time (ms)', fontsize=11, fontweight='bold')
    ax3.set_title(f'Direct Comparison at Matched system_time Values ({len(matched_pairs)} pairs)', fontsize=11, fontweight='bold')
    ax3.grid(alpha=0.3, axis='x')
    ax3.axvline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3, zorder=1)

    # Add annotation
    textstr = f'Approach 2 Agreement: {agreement_rate:.1f}%\n' \
              f'When system_time matches\n' \
              f'Mean difference: {np.mean([p["difference"] for p in matched_pairs]):.2f} ms'
    ax3.text(0.98, 0.97, textstr,
            transform=ax3.transAxes,
            fontsize=10,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    plt.suptitle('Multi-Node Temporal Alignment: Approach 2 (System Time Matching)',
                 fontsize=14, fontweight='bold', y=0.995)

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "5.12_multinode_APPROACH2_REDESIGNED.pdf"
    png_path = output_dir / "5.12_multinode_APPROACH2_REDESIGNED.png"

    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"\n✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")

    plt.close()

    # Print detailed match information
    print(f"\n{'='*80}")
    print("MATCHED PAIRS DETAIL")
    print('='*80)
    print(f"\n{'Pair':<5} {'Node1 Time':<12} {'Node2 Time':<12} {'N1 Offset':<12} {'N2 Offset':<12} {'Diff':<10} {'Agree':<8}")
    print('-'*80)
    for i, pair in enumerate(matched_pairs[:10]):  # Show first 10
        agrees_str = '✓ YES' if pair['agrees'] else '✗ NO'
        print(f"{i+1:<5} {pair['node1_hours']:>11.2f}h {pair['node2_hours']:>11.2f}h "
              f"{pair['node1_offset']:>11.2f}ms {pair['node2_offset']:>11.2f}ms "
              f"{pair['difference']:>9.2f}ms {agrees_str:<8}")
    if len(matched_pairs) > 10:
        print(f"... and {len(matched_pairs) - 10} more pairs")

    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    print(f"\nApproach 2 measures: When both nodes' system clocks show time T")
    print(f"(at DIFFERENT wall-clock moments), do they agree on true time?")
    print(f"\nResult: {agreement_rate:.1f}% agreement across {len(matched_pairs)} matched pairs")
    print(f"  • Node 1 provides ground truth (NTP offset)")
    print(f"  • Node 2 provides ChronoTick prediction")
    print(f"  • Mean difference: {np.mean([p['difference'] for p in matched_pairs]):.3f} ms")
    print(f"  • This tests distributed event timestamp consistency!")

def main():
    """Generate redesigned Approach 2 figure."""

    print("="*80)
    print("APPROACH 2 REDESIGNED: Clear Visualization Across Full Timeline")
    print("="*80)
    print("\nThis redesign shows:")
    print("  • BOTH nodes' full timelines")
    print("  • All data for both nodes (not just matched moments)")
    print("  • Highlighted matches (green = agree, orange = disagree)")
    print("  • Direct comparison panel showing offset differences")
    print("  • Crystal clear what's being compared")

    node1_csv = Path("results/experiment-7/ares-comp-11/chronotick_client_validation_20251020_220343.csv")
    node2_csv = Path("results/experiment-7/ares-comp-12/chronotick_client_validation_20251020_220540.csv")
    output_dir = Path("results/figures/5/experiment-7")

    if node1_csv.exists() and node2_csv.exists():
        generate_approach2_redesigned(node1_csv, node2_csv, output_dir)
    else:
        print("\n⚠️  Dataset files not found!")
        return

    print("\n" + "="*80)
    print("REDESIGNED FIGURE COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
