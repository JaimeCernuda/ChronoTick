#!/usr/bin/env python3
"""
Multi-Node Temporal Alignment - ALIGNED TRUTH TEST

CORRECT APPROACH per user clarification:

Both nodes have same configuration:
- NTP every ~120 seconds
- ChronoTick every ~10 seconds

Question: When Node 1's NTP says "true time is T", does Node 2's ChronoTick
prediction at the aligned moment agree (T within ChronoTick₂ ± 3σ)?

And vice versa for Node 2.

This is different from Approach 1:
- Approach 1: Compares prediction errors between nodes
- This approach: Checks if ground truth from one node falls within
  ChronoTick prediction band of the other node

This directly tests distributed coordination:
"Does my ChronoTick prediction match your NTP ground truth?"
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import timedelta

def find_nearest_sample(target_time, df, time_column='timestamp', tolerance_seconds=60):
    """Find nearest sample to target time within tolerance."""
    time_diffs = (df[time_column] - target_time).abs()
    min_diff = time_diffs.min()

    if min_diff <= pd.Timedelta(seconds=tolerance_seconds):
        return df.loc[time_diffs.idxmin()]
    return None

def generate_aligned_truth_test(node1_csv, node2_csv, output_dir):
    """Generate figure testing if ground truth from one node falls within other's prediction."""

    print("="*80)
    print("ALIGNED TRUTH TEST")
    print("="*80)
    print("\nQuestion: When Node 1 knows true time T (from NTP),")
    print("does Node 2's ChronoTick prediction at that moment include T?")
    print("(And vice versa)")

    # Load data
    df1 = pd.read_csv(node1_csv)
    df2 = pd.read_csv(node2_csv)

    # Both nodes have NTP and ChronoTick samples
    df1_ntp = df1[df1['has_ntp'] == True].copy()
    df2_ntp = df2[df2['has_ntp'] == True].copy()
    df1_all = df1.copy()
    df2_all = df2.copy()

    print(f"\n{'='*80}")
    print("DATA STRUCTURE")
    print('='*80)
    print(f"Node 1: {len(df1_ntp)} NTP samples, {len(df1_all)} total samples")
    print(f"Node 2: {len(df2_ntp)} NTP samples, {len(df2_all)} total samples")
    print(f"✓ Both nodes have same configuration (NTP every ~120s, ChronoTick every ~10s)")

    # Parse timestamps
    df1_ntp['timestamp'] = pd.to_datetime(df1_ntp['datetime'])
    df2_ntp['timestamp'] = pd.to_datetime(df2_ntp['datetime'])
    df1_all['timestamp'] = pd.to_datetime(df1_all['datetime'])
    df2_all['timestamp'] = pd.to_datetime(df2_all['datetime'])

    # Calculate start time offset
    start1 = df1_ntp['timestamp'].iloc[0]
    start2 = df2_ntp['timestamp'].iloc[0]
    start_offset = (start2 - start1).total_seconds()

    print(f"\n{'='*80}")
    print("TIMELINE ALIGNMENT")
    print('='*80)
    print(f"Node 1 start: {start1}")
    print(f"Node 2 start: {start2}")
    print(f"Start offset: {start_offset:.1f} seconds ({start_offset/60:.2f} minutes)")
    print(f"Node 2 started {abs(start_offset):.1f}s {'after' if start_offset > 0 else 'before'} Node 1")

    # Calculate elapsed time from earliest start
    reference_time = min(start1, start2)
    df1_ntp['hours_from_start'] = (df1_ntp['timestamp'] - reference_time).dt.total_seconds() / 3600
    df2_ntp['hours_from_start'] = (df2_ntp['timestamp'] - reference_time).dt.total_seconds() / 3600
    df1_all['hours_from_start'] = (df1_all['timestamp'] - reference_time).dt.total_seconds() / 3600
    df2_all['hours_from_start'] = (df2_all['timestamp'] - reference_time).dt.total_seconds() / 3600

    # Test 1: When Node 1 NTP says true time is T, does Node 2 ChronoTick agree?
    print(f"\n{'='*80}")
    print("TEST 1: Node 1 NTP Truth → Node 2 ChronoTick Prediction")
    print('='*80)

    test1_results = []

    for idx1, row1 in df1_ntp.iterrows():
        ntp_time1 = row1['timestamp']
        ntp_truth1 = row1['ntp_offset_ms']  # This is the GROUND TRUTH

        # Find nearest Node 2 ChronoTick sample at aligned time
        # Use ALL Node 2 samples (not just NTP) for ChronoTick prediction
        nearest2 = find_nearest_sample(ntp_time1, df2_all, tolerance_seconds=5)

        if nearest2 is not None:
            chronotick2 = nearest2['chronotick_offset_ms']
            uncertainty2 = nearest2['chronotick_uncertainty_ms']

            # Does Node 2's ChronoTick prediction contain Node 1's NTP truth?
            lower_bound = chronotick2 - 3 * uncertainty2
            upper_bound = chronotick2 + 3 * uncertainty2

            agrees = (ntp_truth1 >= lower_bound) and (ntp_truth1 <= upper_bound)

            test1_results.append({
                'hours': row1['hours_from_start'],
                'ntp_truth': ntp_truth1,
                'chronotick_pred': chronotick2,
                'lower': lower_bound,
                'upper': upper_bound,
                'agrees': agrees,
                'difference': abs(ntp_truth1 - chronotick2)
            })

    agreement1 = sum([r['agrees'] for r in test1_results]) / len(test1_results) * 100 if test1_results else 0

    print(f"Comparisons: {len(test1_results)} / {len(df1_ntp)} ({len(test1_results)/len(df1_ntp)*100:.1f}%)")
    print(f"Agreement: {agreement1:.1f}%")
    print(f"  Node 1 truth within Node 2 ChronoTick ±3σ: {sum([r['agrees'] for r in test1_results])}")
    print(f"  Outside bounds: {sum([not r['agrees'] for r in test1_results])}")

    # Test 2: When Node 2 NTP says true time is T, does Node 1 ChronoTick agree?
    print(f"\n{'='*80}")
    print("TEST 2: Node 2 NTP Truth → Node 1 ChronoTick Prediction")
    print('='*80)

    test2_results = []

    for idx2, row2 in df2_ntp.iterrows():
        ntp_time2 = row2['timestamp']
        ntp_truth2 = row2['ntp_offset_ms']  # This is the GROUND TRUTH

        # Find nearest Node 1 ChronoTick sample at aligned time
        nearest1 = find_nearest_sample(ntp_time2, df1_all, tolerance_seconds=5)

        if nearest1 is not None:
            chronotick1 = nearest1['chronotick_offset_ms']
            uncertainty1 = nearest1['chronotick_uncertainty_ms']

            # Does Node 1's ChronoTick prediction contain Node 2's NTP truth?
            lower_bound = chronotick1 - 3 * uncertainty1
            upper_bound = chronotick1 + 3 * uncertainty1

            agrees = (ntp_truth2 >= lower_bound) and (ntp_truth2 <= upper_bound)

            test2_results.append({
                'hours': row2['hours_from_start'],
                'ntp_truth': ntp_truth2,
                'chronotick_pred': chronotick1,
                'lower': lower_bound,
                'upper': upper_bound,
                'agrees': agrees,
                'difference': abs(ntp_truth2 - chronotick1)
            })

    agreement2 = sum([r['agrees'] for r in test2_results]) / len(test2_results) * 100 if test2_results else 0

    print(f"Comparisons: {len(test2_results)} / {len(df2_ntp)} ({len(test2_results)/len(df2_ntp)*100:.1f}%)")
    print(f"Agreement: {agreement2:.1f}%")
    print(f"  Node 2 truth within Node 1 ChronoTick ±3σ: {sum([r['agrees'] for r in test2_results])}")
    print(f"  Outside bounds: {sum([not r['agrees'] for r in test2_results])}")

    # Overall agreement
    total_comparisons = len(test1_results) + len(test2_results)
    total_agreements = sum([r['agrees'] for r in test1_results]) + sum([r['agrees'] for r in test2_results])
    overall_agreement = total_agreements / total_comparisons * 100 if total_comparisons > 0 else 0

    print(f"\n{'='*80}")
    print("OVERALL AGREEMENT")
    print('='*80)
    print(f"Total comparisons: {total_comparisons}")
    print(f"Total agreements: {total_agreements}")
    print(f"Overall agreement: {overall_agreement:.1f}%")

    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Panel 1: Node 1 NTP truth vs Node 2 ChronoTick prediction
    ax1 = axes[0]

    for result in test1_results:
        color = '#009E73' if result['agrees'] else '#E69F00'
        alpha = 0.6 if result['agrees'] else 0.9

        # Plot truth point
        ax1.scatter(result['hours'], result['ntp_truth'],
                   color='#0072B2', marker='o', s=50, zorder=4, edgecolors='black', linewidths=1)

        # Plot prediction with uncertainty band
        ax1.plot([result['hours'], result['hours']],
                [result['lower'], result['upper']],
                color='#D55E00', linewidth=6, alpha=0.3, zorder=2)

        ax1.scatter(result['hours'], result['chronotick_pred'],
                   color='#D55E00', marker='s', s=50, zorder=3, edgecolors='black', linewidths=1)

    ax1.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3, zorder=1)
    ax1.set_ylabel('Offset (ms)', fontsize=11)
    ax1.set_title(f'Test 1: Node 1 NTP Truth vs Node 2 ChronoTick Prediction ({agreement1:.1f}% agreement)',
                 fontsize=12, fontweight='bold')
    ax1.legend(['Node 1 NTP Truth', 'Node 2 ChronoTick ±3σ', 'Node 2 ChronoTick'], loc='upper right', fontsize=9)
    ax1.grid(alpha=0.3)

    # Panel 2: Node 2 NTP truth vs Node 1 ChronoTick prediction
    ax2 = axes[1]

    for result in test2_results:
        color = '#009E73' if result['agrees'] else '#E69F00'
        alpha = 0.6 if result['agrees'] else 0.9

        # Plot truth point
        ax2.scatter(result['hours'], result['ntp_truth'],
                   color='#D55E00', marker='s', s=50, zorder=4, edgecolors='black', linewidths=1)

        # Plot prediction with uncertainty band
        ax2.plot([result['hours'], result['hours']],
                [result['lower'], result['upper']],
                color='#0072B2', linewidth=6, alpha=0.3, zorder=2)

        ax2.scatter(result['hours'], result['chronotick_pred'],
                   color='#0072B2', marker='o', s=50, zorder=3, edgecolors='black', linewidths=1)

    ax2.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3, zorder=1)
    ax2.set_ylabel('Offset (ms)', fontsize=11)
    ax2.set_title(f'Test 2: Node 2 NTP Truth vs Node 1 ChronoTick Prediction ({agreement2:.1f}% agreement)',
                 fontsize=12, fontweight='bold')
    ax2.legend(['Node 2 NTP Truth', 'Node 1 ChronoTick ±3σ', 'Node 1 ChronoTick'], loc='upper right', fontsize=9)
    ax2.grid(alpha=0.3)

    # Panel 3: Overall agreement statistics
    ax3 = axes[2]

    categories = ['Test 1\n(N1→N2)', 'Test 2\n(N2→N1)', 'Overall']
    agree_counts = [
        sum([r['agrees'] for r in test1_results]),
        sum([r['agrees'] for r in test2_results]),
        total_agreements
    ]
    total_counts = [
        len(test1_results),
        len(test2_results),
        total_comparisons
    ]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax3.bar(x - width/2, agree_counts, width, label='Agree', color='#009E73', alpha=0.7, edgecolor='black')
    bars2 = ax3.bar(x + width/2, [t - a for t, a in zip(total_counts, agree_counts)],
                   width, label='Disagree', color='#E69F00', alpha=0.7, edgecolor='black')

    ax3.set_ylabel('Number of Comparisons', fontsize=11)
    ax3.set_title(f'Cross-Node Agreement: {overall_agreement:.1f}%', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3, axis='y')

    # Add percentages on bars
    for i, (agree, total) in enumerate(zip(agree_counts, total_counts)):
        pct = agree / total * 100 if total > 0 else 0
        ax3.text(i, total + 5, f'{pct:.1f}%', ha='center', fontsize=10, fontweight='bold')

    plt.suptitle('Multi-Node Temporal Alignment: Aligned Truth Test',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "5.12_multinode_ALIGNED_TRUTH_TEST.pdf"
    png_path = output_dir / "5.12_multinode_ALIGNED_TRUTH_TEST.png"

    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"\n✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")

    plt.close()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    print(f"\nThis approach tests: When one node knows true time T (from NTP),")
    print(f"does the other node's ChronoTick prediction agree?")
    print(f"\nResults:")
    print(f"  • Test 1 (N1 NTP → N2 ChronoTick): {agreement1:.1f}% agreement")
    print(f"  • Test 2 (N2 NTP → N1 ChronoTick): {agreement2:.1f}% agreement")
    print(f"  • Overall: {overall_agreement:.1f}% agreement")
    print(f"  • Total comparisons: {total_comparisons}")
    print(f"\nThis directly tests distributed coordination:")
    print(f'"Does my ChronoTick prediction match your NTP ground truth?"')

def main():
    """Generate aligned truth test figure."""

    print("="*80)
    print("MULTI-NODE TEMPORAL ALIGNMENT - ALIGNED TRUTH TEST")
    print("="*80)
    print("\nCorrected understanding:")
    print("  • Both nodes have SAME configuration")
    print("  • NTP every ~120s, ChronoTick every ~10s")
    print("  • Test: Does ground truth from one node fall within")
    print("    ChronoTick ±3σ prediction of the other node?")
    print("  • This is DIFFERENT from comparing prediction errors")

    node1_csv = Path("results/experiment-7/ares-comp-11/chronotick_client_validation_20251020_220343.csv")
    node2_csv = Path("results/experiment-7/ares-comp-12/chronotick_client_validation_20251020_220540.csv")
    output_dir = Path("results/figures/5/experiment-7")

    if node1_csv.exists() and node2_csv.exists():
        generate_aligned_truth_test(node1_csv, node2_csv, output_dir)
    else:
        print("\n⚠️  Dataset files not found!")
        return

    print("\n" + "="*80)
    print("ALIGNED TRUTH TEST COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
