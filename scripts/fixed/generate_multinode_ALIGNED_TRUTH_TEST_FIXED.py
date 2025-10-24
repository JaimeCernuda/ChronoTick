#!/usr/bin/env python3
"""
Multi-Node Temporal Alignment - ALIGNED TRUTH TEST (FIXED)

CORRECTED: Properly align timelines by elapsed time, not wall-clock timestamps.

When Node 1 measures NTP at elapsed_time=T (seconds since its start):
- Find Node 2's ChronoTick sample at elapsed_time≈T (seconds since its start)
- This accounts for the 102-second start offset properly
- Tests: Does Node 2's prediction at the same ELAPSED time agree with Node 1's ground truth?
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def find_by_elapsed_time(target_elapsed, df, elapsed_column='elapsed_seconds', tolerance_seconds=5):
    """Find nearest sample by elapsed time (not wall-clock)."""
    time_diffs = np.abs(df[elapsed_column] - target_elapsed)
    min_diff = time_diffs.min()

    if min_diff <= tolerance_seconds:
        return df.loc[time_diffs.idxmin()]
    return None

def generate_aligned_truth_test_fixed(node1_csv, node2_csv, output_dir):
    """Generate aligned truth test with proper elapsed time matching."""

    print("="*80)
    print("ALIGNED TRUTH TEST (FIXED)")
    print("="*80)
    print("\nCORRECTED: Align by elapsed time, not wall-clock timestamps")
    print("\nWhen Node 1 at elapsed_time=T has NTP ground truth,")
    print("compare with Node 2 ChronoTick at elapsed_time=T")
    print("(This properly accounts for 102s start offset)")

    # Load data
    df1 = pd.read_csv(node1_csv)
    df2 = pd.read_csv(node2_csv)

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

    # Verify elapsed_seconds column exists
    if 'elapsed_seconds' not in df1_all.columns or 'elapsed_seconds' not in df2_all.columns:
        print("\n⚠️  'elapsed_seconds' column not found! Computing from timestamps...")
        df1_all['elapsed_seconds'] = (df1_all['timestamp'] - df1_all['timestamp'].iloc[0]).dt.total_seconds()
        df2_all['elapsed_seconds'] = (df2_all['timestamp'] - df2_all['timestamp'].iloc[0]).dt.total_seconds()
        df1_ntp['elapsed_seconds'] = (df1_ntp['timestamp'] - df1_ntp['timestamp'].iloc[0]).dt.total_seconds()
        df2_ntp['elapsed_seconds'] = (df2_ntp['timestamp'] - df2_ntp['timestamp'].iloc[0]).dt.total_seconds()

    # Calculate hours for visualization
    reference_time = min(start1, start2)
    df1_ntp['hours_from_start'] = (df1_ntp['timestamp'] - reference_time).dt.total_seconds() / 3600
    df2_ntp['hours_from_start'] = (df2_ntp['timestamp'] - reference_time).dt.total_seconds() / 3600
    df1_all['hours_from_start'] = (df1_all['timestamp'] - reference_time).dt.total_seconds() / 3600
    df2_all['hours_from_start'] = (df2_all['timestamp'] - reference_time).dt.total_seconds() / 3600

    # TEST 1: Node 1 NTP truth → Node 2 ChronoTick prediction
    # Match by ELAPSED TIME (not wall-clock)
    print(f"\n{'='*80}")
    print("TEST 1: Node 1 NTP Truth → Node 2 ChronoTick Prediction")
    print('='*80)
    print("Matching by elapsed_time (accounts for start offset)")

    test1_results = []

    for idx1, row1 in df1_ntp.iterrows():
        elapsed1 = row1['elapsed_seconds']  # Elapsed time since Node 1 started
        ntp_truth1 = row1['ntp_offset_ms']  # Ground truth

        # Find Node 2 sample at same ELAPSED time
        nearest2 = find_by_elapsed_time(elapsed1, df2_all, tolerance_seconds=5)

        if nearest2 is not None:
            chronotick2 = nearest2['chronotick_offset_ms']
            uncertainty2 = nearest2['chronotick_uncertainty_ms']

            # Does Node 2's ChronoTick prediction contain Node 1's NTP truth?
            lower_bound = chronotick2 - 3 * uncertainty2
            upper_bound = chronotick2 + 3 * uncertainty2

            agrees = (ntp_truth1 >= lower_bound) and (ntp_truth1 <= upper_bound)

            test1_results.append({
                'elapsed': elapsed1,
                'hours': row1['hours_from_start'],
                'ntp_truth': ntp_truth1,
                'chronotick_pred': chronotick2,
                'lower': lower_bound,
                'upper': upper_bound,
                'agrees': agrees,
                'difference': abs(ntp_truth1 - chronotick2)
            })

    agreement1 = sum([r['agrees'] for r in test1_results]) / len(test1_results) * 100 if test1_results else 0

    print(f"\nComparisons: {len(test1_results)} / {len(df1_ntp)} ({len(test1_results)/len(df1_ntp)*100:.1f}%)")
    print(f"Agreement: {agreement1:.1f}%")
    print(f"  Node 1 truth within Node 2 ChronoTick ±3σ: {sum([r['agrees'] for r in test1_results])}")
    print(f"  Outside bounds: {sum([not r['agrees'] for r in test1_results])}")

    # Show some examples
    print(f"\nExample matches (first 5):")
    print(f"{'Elapsed(s)':<12} {'N1 Truth':<12} {'N2 Pred':<12} {'N2 Lower':<12} {'N2 Upper':<12} {'Agree':<8}")
    print('-'*80)
    for i, r in enumerate(test1_results[:5]):
        agrees_str = '✓' if r['agrees'] else '✗'
        print(f"{r['elapsed']:>11.1f}s {r['ntp_truth']:>11.2f}ms {r['chronotick_pred']:>11.2f}ms "
              f"{r['lower']:>11.2f}ms {r['upper']:>11.2f}ms {agrees_str:<8}")

    # TEST 2: Node 2 NTP truth → Node 1 ChronoTick prediction
    print(f"\n{'='*80}")
    print("TEST 2: Node 2 NTP Truth → Node 1 ChronoTick Prediction")
    print('='*80)
    print("Matching by elapsed_time (accounts for start offset)")

    test2_results = []

    for idx2, row2 in df2_ntp.iterrows():
        elapsed2 = row2['elapsed_seconds']  # Elapsed time since Node 2 started
        ntp_truth2 = row2['ntp_offset_ms']  # Ground truth

        # Find Node 1 sample at same ELAPSED time
        nearest1 = find_by_elapsed_time(elapsed2, df1_all, tolerance_seconds=5)

        if nearest1 is not None:
            chronotick1 = nearest1['chronotick_offset_ms']
            uncertainty1 = nearest1['chronotick_uncertainty_ms']

            # Does Node 1's ChronoTick prediction contain Node 2's NTP truth?
            lower_bound = chronotick1 - 3 * uncertainty1
            upper_bound = chronotick1 + 3 * uncertainty1

            agrees = (ntp_truth2 >= lower_bound) and (ntp_truth2 <= upper_bound)

            test2_results.append({
                'elapsed': elapsed2,
                'hours': row2['hours_from_start'],
                'ntp_truth': ntp_truth2,
                'chronotick_pred': chronotick1,
                'lower': lower_bound,
                'upper': upper_bound,
                'agrees': agrees,
                'difference': abs(ntp_truth2 - chronotick1)
            })

    agreement2 = sum([r['agrees'] for r in test2_results]) / len(test2_results) * 100 if test2_results else 0

    print(f"\nComparisons: {len(test2_results)} / {len(df2_ntp)} ({len(test2_results)/len(df2_ntp)*100:.1f}%)")
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

    # Panel 1: Test 1 results
    ax1 = axes[0]

    for result in test1_results:
        color = '#009E73' if result['agrees'] else '#E69F00'

        # Plot uncertainty band
        ax1.plot([result['hours'], result['hours']],
                [result['lower'], result['upper']],
                color='#D55E00', linewidth=6, alpha=0.3, zorder=2)

        # Plot prediction
        ax1.scatter(result['hours'], result['chronotick_pred'],
                   color='#D55E00', marker='s', s=50, zorder=3, edgecolors='black', linewidths=1)

        # Plot truth
        ax1.scatter(result['hours'], result['ntp_truth'],
                   color='#0072B2', marker='o', s=50, zorder=4,
                   edgecolors=color, linewidths=2)

    ax1.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3, zorder=1)
    ax1.set_ylabel('Offset (ms)', fontsize=11)
    ax1.set_title(f'Test 1: Node 1 NTP Truth vs Node 2 ChronoTick Prediction (aligned by elapsed time) - {agreement1:.1f}% agreement',
                 fontsize=11, fontweight='bold')
    ax1.legend(['Node 2 ChronoTick ±3σ', 'Node 2 ChronoTick', 'Node 1 NTP Truth'], loc='upper right', fontsize=9)
    ax1.grid(alpha=0.3)

    # Panel 2: Test 2 results
    ax2 = axes[1]

    for result in test2_results:
        color = '#009E73' if result['agrees'] else '#E69F00'

        # Plot uncertainty band
        ax2.plot([result['hours'], result['hours']],
                [result['lower'], result['upper']],
                color='#0072B2', linewidth=6, alpha=0.3, zorder=2)

        # Plot prediction
        ax2.scatter(result['hours'], result['chronotick_pred'],
                   color='#0072B2', marker='o', s=50, zorder=3, edgecolors='black', linewidths=1)

        # Plot truth
        ax2.scatter(result['hours'], result['ntp_truth'],
                   color='#D55E00', marker='s', s=50, zorder=4,
                   edgecolors=color, linewidths=2)

    ax2.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3, zorder=1)
    ax2.set_ylabel('Offset (ms)', fontsize=11)
    ax2.set_title(f'Test 2: Node 2 NTP Truth vs Node 1 ChronoTick Prediction (aligned by elapsed time) - {agreement2:.1f}% agreement',
                 fontsize=11, fontweight='bold')
    ax2.legend(['Node 1 ChronoTick ±3σ', 'Node 1 ChronoTick', 'Node 2 NTP Truth'], loc='upper right', fontsize=9)
    ax2.grid(alpha=0.3)

    # Panel 3: Overall statistics
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
    ax3.set_title(f'Cross-Node Agreement (Aligned by Elapsed Time): {overall_agreement:.1f}%', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3, axis='y')

    # Add percentages
    for i, (agree, total) in enumerate(zip(agree_counts, total_counts)):
        pct = agree / total * 100 if total > 0 else 0
        ax3.text(i, total + 5, f'{pct:.1f}%', ha='center', fontsize=10, fontweight='bold')

    plt.suptitle('Multi-Node Temporal Alignment: Aligned Truth Test (Fixed)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "5.12_multinode_ALIGNED_TRUTH_TEST_FIXED.pdf"
    png_path = output_dir / "5.12_multinode_ALIGNED_TRUTH_TEST_FIXED.png"

    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"\n✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")

    plt.close()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    print(f"\nThis FIXED approach properly aligns by elapsed time:")
    print(f"  • At elapsed_time=T on Node 1, we check Node 2 at elapsed_time=T")
    print(f"  • This accounts for 102s start offset between nodes")
    print(f"  • Tests: Does ground truth from one node fall within")
    print(f"    ChronoTick ±3σ prediction of the other node?")
    print(f"\nResults:")
    print(f"  • Test 1 (N1 NTP → N2 ChronoTick): {agreement1:.1f}% agreement")
    print(f"  • Test 2 (N2 NTP → N1 ChronoTick): {agreement2:.1f}% agreement")
    print(f"  • Overall: {overall_agreement:.1f}% agreement")
    print(f"  • Total comparisons: {total_comparisons}")

def main():
    """Generate aligned truth test figure (fixed version)."""

    print("="*80)
    print("MULTI-NODE TEMPORAL ALIGNMENT - ALIGNED TRUTH TEST (FIXED)")
    print("="*80)
    print("\nFIXED: Now properly aligns by elapsed_time instead of wall-clock")
    print("  • Accounts for 102-second start offset between nodes")
    print("  • When Node 1 at elapsed_time=T has NTP ground truth,")
    print("    find Node 2's ChronoTick at elapsed_time=T (not wall-clock match)")

    node1_csv = Path("results/experiment-7/ares-comp-11/chronotick_client_validation_20251020_220343.csv")
    node2_csv = Path("results/experiment-7/ares-comp-12/chronotick_client_validation_20251020_220540.csv")
    output_dir = Path("results/figures/5/experiment-7")

    if node1_csv.exists() and node2_csv.exists():
        generate_aligned_truth_test_fixed(node1_csv, node2_csv, output_dir)
    else:
        print("\n⚠️  Dataset files not found!")
        return

    print("\n" + "="*80)
    print("ALIGNED TRUTH TEST (FIXED) COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
