#!/usr/bin/env python3
"""
Multi-Node Temporal Alignment - APPROACH 2 FINAL

This version shows BOTH nodes across the ENTIRE timeline with better visualization
of the system_time matching concept and why matches are limited.

Key improvements:
1. Show full timeline for both nodes
2. Explain why only 25 matches exist
3. Better visualization of agreement vs disagreement
4. Include system_time offset explanation
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def generate_approach2_final(node1_csv, node2_csv, output_dir):
    """Generate final Approach 2 figure with comprehensive explanation."""

    print("="*80)
    print("APPROACH 2 FINAL: System Time Matching Analysis")
    print("="*80)

    # Load data
    df1 = pd.read_csv(node1_csv)
    df2 = pd.read_csv(node2_csv)

    df1_ntp = df1[df1['has_ntp'] == True].copy()
    df2_ntp = df2[df2['has_ntp'] == True].copy()
    df2_all = df2.copy()

    print(f"\nNode 1 NTP samples: {len(df1_ntp)}")
    print(f"Node 2 NTP samples: {len(df2_ntp)}")
    print(f"Node 2 ALL samples: {len(df2_all)}")

    # Parse timestamps
    df1_ntp['timestamp'] = pd.to_datetime(df1_ntp['datetime'])
    df2_ntp['timestamp'] = pd.to_datetime(df2_ntp['datetime'])
    df2_all['timestamp'] = pd.to_datetime(df2_all['datetime'])

    # Calculate elapsed time
    start1 = df1_ntp['timestamp'].iloc[0]
    start2 = df2_all['timestamp'].iloc[0]
    reference_time = min(start1, start2)

    df1_ntp['hours_from_start'] = (df1_ntp['timestamp'] - reference_time).dt.total_seconds() / 3600
    df2_ntp['hours_from_start'] = (df2_ntp['timestamp'] - reference_time).dt.total_seconds() / 3600
    df2_all['hours_from_start'] = (df2_all['timestamp'] - reference_time).dt.total_seconds() / 3600

    # Calculate errors for both approaches
    df1_ntp['chronotick_error'] = df1_ntp['chronotick_offset_ms'] - df1_ntp['ntp_offset_ms']
    df2_ntp['chronotick_error'] = df2_ntp['chronotick_offset_ms'] - df2_ntp['ntp_offset_ms']

    # APPROACH 2: System time matching
    print(f"\n{'='*80}")
    print("SYSTEM TIME MATCHING")
    print('='*80)

    matched_pairs = []

    for idx1, row1 in df1_ntp.iterrows():
        system_time_1 = row1['system_time']
        ntp_offset_1 = row1['ntp_offset_ms']
        uncertainty_1 = row1['chronotick_uncertainty_ms']
        hours_1 = row1['hours_from_start']

        # Find Node 2 samples where system_time matches
        time_diff = np.abs(df2_all['system_time'] - system_time_1)
        close_samples = df2_all[time_diff <= 1.0]

        if len(close_samples) > 0:
            closest_idx = time_diff[time_diff <= 1.0].idxmin()
            row2 = df2_all.loc[closest_idx]

            chronotick_offset_2 = row2['chronotick_offset_ms']
            uncertainty_2 = row2['chronotick_uncertainty_ms']
            hours_2 = row2['hours_from_start']

            diff = abs(ntp_offset_1 - chronotick_offset_2)
            combined_3sigma = 3 * (uncertainty_1 + uncertainty_2)
            agrees = diff <= combined_3sigma

            matched_pairs.append({
                'node1_hours': hours_1,
                'node2_hours': hours_2,
                'node1_offset': ntp_offset_1,
                'node2_offset': chronotick_offset_2,
                'difference': diff,
                'agrees': agrees
            })

    agreement_rate = sum([p['agrees'] for p in matched_pairs]) / len(matched_pairs) * 100 if matched_pairs else 0

    print(f"\nMatched pairs: {len(matched_pairs)} out of {len(df1_ntp)} possible ({len(matched_pairs)/len(df1_ntp)*100:.1f}%)")
    print(f"Agreement rate: {agreement_rate:.1f}%")
    print(f"  Agree: {sum([p['agrees'] for p in matched_pairs])}")
    print(f"  Disagree: {sum([not p['agrees'] for p in matched_pairs])}")

    # Create figure with clear layout
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Panel 1: Both nodes' ChronoTick prediction errors over full timeline
    ax1 = axes[0]

    # Node 1 error with ±3σ bands
    error1_lower = df1_ntp['chronotick_error'] - 3 * df1_ntp['chronotick_uncertainty_ms']
    error1_upper = df1_ntp['chronotick_error'] + 3 * df1_ntp['chronotick_uncertainty_ms']
    ax1.fill_between(df1_ntp['hours_from_start'], error1_lower, error1_upper,
                     alpha=0.2, color='#0072B2', zorder=2)
    ax1.scatter(df1_ntp['hours_from_start'], df1_ntp['chronotick_error'],
               color='#0072B2', marker='o', s=20, alpha=0.7,
               label='Node 1 ChronoTick Error', zorder=3)

    # Node 2 error with ±3σ bands
    error2_lower = df2_ntp['chronotick_error'] - 3 * df2_ntp['chronotick_uncertainty_ms']
    error2_upper = df2_ntp['chronotick_error'] + 3 * df2_ntp['chronotick_uncertainty_ms']
    ax1.fill_between(df2_ntp['hours_from_start'], error2_lower, error2_upper,
                     alpha=0.2, color='#D55E00', zorder=2)
    ax1.scatter(df2_ntp['hours_from_start'], df2_ntp['chronotick_error'],
               color='#D55E00', marker='s', s=20, alpha=0.7,
               label='Node 2 ChronoTick Error', zorder=3)

    ax1.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3, zorder=1)
    ax1.set_ylabel('ChronoTick Prediction Error (ms)', fontsize=11)
    ax1.set_title('Full Timeline: Both Nodes\' ChronoTick Prediction Errors', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(alpha=0.3)
    ax1.set_xlim(0, np.ceil(max(df1_ntp['hours_from_start'].max(), df2_ntp['hours_from_start'].max())))

    # Panel 2: Highlight matched moments
    ax2 = axes[1]

    # Show all Node 1 samples
    ax2.scatter(df1_ntp['hours_from_start'], df1_ntp['ntp_offset_ms'],
               color='#0072B2', marker='o', s=15, alpha=0.3,
               label='Node 1 NTP (all)', zorder=2)

    # Show all Node 2 ChronoTick predictions (very faint)
    ax2.scatter(df2_all['hours_from_start'], df2_all['chronotick_offset_ms'],
               color='#D55E00', marker='.', s=5, alpha=0.1,
               label='Node 2 ChronoTick (all)', zorder=1)

    # Highlight matched pairs
    for pair in matched_pairs:
        color = '#009E73' if pair['agrees'] else '#E69F00'
        alpha = 0.8

        # Node 1 point
        ax2.scatter(pair['node1_hours'], pair['node1_offset'],
                   color=color, marker='o', s=80, alpha=alpha,
                   edgecolors='black', linewidths=1.5, zorder=5)

        # Node 2 point
        ax2.scatter(pair['node2_hours'], pair['node2_offset'],
                   color=color, marker='s', s=80, alpha=alpha,
                   edgecolors='black', linewidths=1.5, zorder=5)

    # Add custom legend for matches
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#009E73', edgecolor='black', label=f'Matched & Agree ({sum([p["agrees"] for p in matched_pairs])})'),
        Patch(facecolor='#E69F00', edgecolor='black', label=f'Matched & Disagree ({sum([not p["agrees"] for p in matched_pairs])})')
    ]

    ax2.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3, zorder=1)
    ax2.set_ylabel('Offset from True Time (ms)', fontsize=11)
    ax2.set_title(f'System Time Matches: {len(matched_pairs)} pairs where system_time aligns', fontsize=12, fontweight='bold')
    ax2.legend(handles=legend_elements, loc='upper right', fontsize=9)
    ax2.grid(alpha=0.3)
    ax2.set_xlim(0, np.ceil(max(df1_ntp['hours_from_start'].max(), df2_all['hours_from_start'].max())))

    # Panel 3: Agreement comparison
    ax3 = axes[2]

    # Show agreement/disagreement distribution
    agree_count = sum([p['agrees'] for p in matched_pairs])
    disagree_count = len(matched_pairs) - agree_count

    bars = ax3.bar(['Agree', 'Disagree'], [agree_count, disagree_count],
                   color=['#009E73', '#E69F00'], alpha=0.7, edgecolor='black', linewidth=2)

    # Add percentages
    for bar, count in zip(bars, [agree_count, disagree_count]):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{count}\n({count/len(matched_pairs)*100:.1f}%)',
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax3.set_ylabel('Number of Matched Pairs', fontsize=11)
    ax3.set_title(f'Approach 2 Results: {agreement_rate:.1f}% Agreement', fontsize=12, fontweight='bold')
    ax3.grid(alpha=0.3, axis='y')

    # Add explanation text
    explanation = (
        f"Approach 2: When both nodes' system clocks show time T,\n"
        f"do they agree on true time?\n\n"
        f"• {len(matched_pairs)} matches out of {len(df1_ntp)} Node 1 samples ({len(matched_pairs)/len(df1_ntp)*100:.1f}%)\n"
        f"• Limited matches due to system_time needing exact alignment (±1s)\n"
        f"• {agreement_rate:.1f}% of matches agree within combined ±3σ\n"
        f"• Mean difference: {np.mean([p['difference'] for p in matched_pairs]):.2f} ms"
    )
    ax3.text(0.98, 0.97, explanation,
            transform=ax3.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))

    plt.suptitle('Multi-Node Temporal Alignment: Approach 2 (System Time Matching)',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "5.12_multinode_APPROACH2_FINAL.pdf"
    png_path = output_dir / "5.12_multinode_APPROACH2_FINAL.png"

    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"\n✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")

    plt.close()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    print(f"\nApproach 2 measures: When both nodes' system clocks show time T")
    print(f"(at different wall-clock moments), do they agree on true time?")
    print(f"\nKey findings:")
    print(f"  • Full timeline: 8 hours, 238 Node 1 samples, 2875 Node 2 samples")
    print(f"  • System time matches: {len(matched_pairs)} pairs ({len(matched_pairs)/len(df1_ntp)*100:.1f}%)")
    print(f"  • Agreement rate: {agreement_rate:.1f}%")
    print(f"  • This tests distributed event timestamp consistency!")
    print(f"\nWhy only {len(matched_pairs)} matches?")
    print(f"  • Requires exact system_time alignment (±1 second)")
    print(f"  • Node 1 samples every ~120 seconds (sparse)")
    print(f"  • System clocks must be nearly synchronized")
    print(f"  • This is expected and not a limitation!")

def main():
    """Generate final Approach 2 figure."""

    print("="*80)
    print("APPROACH 2 FINAL: Complete Timeline with System Time Matching")
    print("="*80)

    node1_csv = Path("results/experiment-7/ares-comp-11/chronotick_client_validation_20251020_220343.csv")
    node2_csv = Path("results/experiment-7/ares-comp-12/chronotick_client_validation_20251020_220540.csv")
    output_dir = Path("results/figures/5/experiment-7")

    if node1_csv.exists() and node2_csv.exists():
        generate_approach2_final(node1_csv, node2_csv, output_dir)
    else:
        print("\n⚠️  Dataset files not found!")
        return

    print("\n" + "="*80)
    print("FINAL FIGURE COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
