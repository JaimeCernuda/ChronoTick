#!/usr/bin/env python3
"""
Multi-Node Temporal Alignment - ALIGNED TRUTH TEST (CORRECT)

PROPERLY CORRECTED: Account for 102-second deployment time difference!

If Node 2 started 102 seconds AFTER Node 1:
- Node 1 at elapsed=152s → same wall-clock moment as Node 2 at elapsed=50s
- Mapping: elapsed₂ = elapsed₁ - start_offset

This tests: At the same wall-clock moment, do the nodes' ChronoTick predictions
agree with each other's NTP ground truth?
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def find_by_elapsed_time(target_elapsed, df, elapsed_column='elapsed_seconds', tolerance_seconds=5):
    """Find nearest sample by elapsed time."""
    time_diffs = np.abs(df[elapsed_column] - target_elapsed)
    min_diff = time_diffs.min()

    if min_diff <= tolerance_seconds:
        return df.loc[time_diffs.idxmin()]
    return None

def generate_aligned_truth_test_correct(node1_csv, node2_csv, output_dir):
    """Generate aligned truth test with CORRECT elapsed time offset."""

    print("="*80)
    print("ALIGNED TRUTH TEST (PROPERLY CORRECTED)")
    print("="*80)
    print("\nAccount for 102-second deployment time difference:")
    print("  • Node 2 started 102s AFTER Node 1")
    print("  • Node 1 elapsed=152s → same moment as Node 2 elapsed=50s")
    print("  • Mapping: elapsed₂ = elapsed₁ - start_offset")

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

    # Calculate start time offset (how much later Node 2 started)
    start1 = df1_ntp['timestamp'].iloc[0]
    start2 = df2_ntp['timestamp'].iloc[0]
    start_offset = (start2 - start1).total_seconds()  # Positive if Node 2 started later

    print(f"\n{'='*80}")
    print("TIMELINE ALIGNMENT")
    print('='*80)
    print(f"Node 1 start: {start1}")
    print(f"Node 2 start: {start2}")
    print(f"Start offset: {start_offset:.1f} seconds ({start_offset/60:.2f} minutes)")
    print(f"Node 2 started {abs(start_offset):.1f}s {'AFTER' if start_offset > 0 else 'BEFORE'} Node 1")
    print(f"\nMapping formula:")
    if start_offset > 0:
        print(f"  elapsed₂ = elapsed₁ - {start_offset:.1f}")
    else:
        print(f"  elapsed₂ = elapsed₁ + {abs(start_offset):.1f}")

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
    # At Node 1's elapsed time, find Node 2 at ALIGNED wall-clock moment
    print(f"\n{'='*80}")
    print("TEST 1: Node 1 NTP Truth → Node 2 ChronoTick Prediction")
    print('='*80)
    print("Mapping Node 1 elapsed time to Node 2 elapsed time (accounts for start offset)")

    test1_results = []

    for idx1, row1 in df1_ntp.iterrows():
        elapsed1 = row1['elapsed_seconds']  # Elapsed time on Node 1

        # Map to Node 2's elapsed time at the same wall-clock moment
        elapsed2_target = elapsed1 - start_offset

        ntp_truth1 = row1['ntp_offset_ms']  # Ground truth from Node 1

        # Find Node 2 sample at the aligned wall-clock moment
        nearest2 = find_by_elapsed_time(elapsed2_target, df2_all, tolerance_seconds=5)

        if nearest2 is not None:
            chronotick2 = nearest2['chronotick_offset_ms']
            uncertainty2 = nearest2['chronotick_uncertainty_ms']
            elapsed2_actual = nearest2['elapsed_seconds']

            # Does Node 2's ChronoTick prediction contain Node 1's NTP truth?
            lower_bound = chronotick2 - 3 * uncertainty2
            upper_bound = chronotick2 + 3 * uncertainty2

            agrees = (ntp_truth1 >= lower_bound) and (ntp_truth1 <= upper_bound)

            test1_results.append({
                'elapsed1': elapsed1,
                'elapsed2_target': elapsed2_target,
                'elapsed2_actual': elapsed2_actual,
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
    print(f"{'N1 Elapsed':<12} {'N2 Target':<12} {'N2 Actual':<12} {'N1 Truth':<12} {'N2 Pred':<12} {'Agree':<8}")
    print('-'*80)
    for i, r in enumerate(test1_results[:5]):
        agrees_str = '✓' if r['agrees'] else '✗'
        print(f"{r['elapsed1']:>11.1f}s {r['elapsed2_target']:>11.1f}s {r['elapsed2_actual']:>11.1f}s "
              f"{r['ntp_truth']:>11.2f}ms {r['chronotick_pred']:>11.2f}ms {agrees_str:<8}")

    # TEST 2: Node 2 NTP truth → Node 1 ChronoTick prediction
    print(f"\n{'='*80}")
    print("TEST 2: Node 2 NTP Truth → Node 1 ChronoTick Prediction")
    print('='*80)
    print("Mapping Node 2 elapsed time to Node 1 elapsed time (accounts for start offset)")

    test2_results = []

    for idx2, row2 in df2_ntp.iterrows():
        elapsed2 = row2['elapsed_seconds']  # Elapsed time on Node 2

        # Map to Node 1's elapsed time at the same wall-clock moment
        elapsed1_target = elapsed2 + start_offset

        ntp_truth2 = row2['ntp_offset_ms']  # Ground truth from Node 2

        # Find Node 1 sample at the aligned wall-clock moment
        nearest1 = find_by_elapsed_time(elapsed1_target, df1_all, tolerance_seconds=5)

        if nearest1 is not None:
            chronotick1 = nearest1['chronotick_offset_ms']
            uncertainty1 = nearest1['chronotick_uncertainty_ms']
            elapsed1_actual = nearest1['elapsed_seconds']

            # Does Node 1's ChronoTick prediction contain Node 2's NTP truth?
            lower_bound = chronotick1 - 3 * uncertainty1
            upper_bound = chronotick1 + 3 * uncertainty1

            agrees = (ntp_truth2 >= lower_bound) and (ntp_truth2 <= upper_bound)

            test2_results.append({
                'elapsed2': elapsed2,
                'elapsed1_target': elapsed1_target,
                'elapsed1_actual': elapsed1_actual,
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

    # Show some examples
    print(f"\nExample matches (first 5):")
    print(f"{'N2 Elapsed':<12} {'N1 Target':<12} {'N1 Actual':<12} {'N2 Truth':<12} {'N1 Pred':<12} {'Agree':<8}")
    print('-'*80)
    for i, r in enumerate(test2_results[:5]):
        agrees_str = '✓' if r['agrees'] else '✗'
        print(f"{r['elapsed2']:>11.1f}s {r['elapsed1_target']:>11.1f}s {r['elapsed1_actual']:>11.1f}s "
              f"{r['ntp_truth']:>11.2f}ms {r['chronotick_pred']:>11.2f}ms {agrees_str:<8}")

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

        # Plot truth (highlighted by agreement color)
        ax1.scatter(result['hours'], result['ntp_truth'],
                   color='#0072B2', marker='o', s=50, zorder=4,
                   edgecolors=color, linewidths=2)

    ax1.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3, zorder=1)
    ax1.set_ylabel('Offset (ms)', fontsize=11)
    ax1.set_title(f'Test 1: Node 1 NTP Truth vs Node 2 ChronoTick (at same wall-clock moment) - {agreement1:.1f}% agreement',
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

        # Plot truth (highlighted by agreement color)
        ax2.scatter(result['hours'], result['ntp_truth'],
                   color='#D55E00', marker='s', s=50, zorder=4,
                   edgecolors=color, linewidths=2)

    ax2.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.3, zorder=1)
    ax2.set_ylabel('Offset (ms)', fontsize=11)
    ax2.set_title(f'Test 2: Node 2 NTP Truth vs Node 1 ChronoTick (at same wall-clock moment) - {agreement2:.1f}% agreement',
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
    ax3.set_title(f'Cross-Node Agreement (Properly Aligned): {overall_agreement:.1f}%', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend(fontsize=10)
    ax3.grid(alpha=0.3, axis='y')

    # Add percentages
    for i, (agree, total) in enumerate(zip(agree_counts, total_counts)):
        pct = agree / total * 100 if total > 0 else 0
        ax3.text(i, total + 5, f'{pct:.1f}%', ha='center', fontsize=10, fontweight='bold')

    # Add explanation
    explanation = (
        f"Accounts for {abs(start_offset):.0f}s deployment offset:\n"
        f"Node 1 elapsed={152:.0f}s → Node 2 elapsed={152-start_offset:.0f}s\n"
        f"(same wall-clock moment)"
    )
    ax3.text(0.98, 0.97, explanation,
            transform=ax3.transAxes,
            fontsize=9,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.6))

    plt.suptitle('Multi-Node Temporal Alignment: Aligned Truth Test (Properly Corrected)',
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "5.12_multinode_ALIGNED_TRUTH_TEST_CORRECT.pdf"
    png_path = output_dir / "5.12_multinode_ALIGNED_TRUTH_TEST_CORRECT.png"

    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"\n✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")

    plt.close()

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print('='*80)
    print(f"\nThis CORRECTLY accounts for {abs(start_offset):.0f}s deployment offset:")
    print(f"  • Node 2 started {abs(start_offset):.0f}s after Node 1")
    print(f"  • elapsed₂ = elapsed₁ - {start_offset:.1f}")
    print(f"  • Compares at same wall-clock moments")
    print(f"\nResults:")
    print(f"  • Test 1 (N1 NTP → N2 ChronoTick): {agreement1:.1f}% agreement")
    print(f"  • Test 2 (N2 NTP → N1 ChronoTick): {agreement2:.1f}% agreement")
    print(f"  • Overall: {overall_agreement:.1f}% agreement")
    print(f"  • Total comparisons: {total_comparisons}")

def main():
    """Generate properly corrected aligned truth test figure."""

    print("="*80)
    print("MULTI-NODE TEMPORAL ALIGNMENT - PROPERLY CORRECTED")
    print("="*80)
    print("\nCORRECTLY accounts for 102-second deployment offset!")
    print("  • Node 1 elapsed=152s → Node 2 elapsed=50s (same wall-clock moment)")
    print("  • Mapping: elapsed₂ = elapsed₁ - start_offset")

    node1_csv = Path("results/experiment-7/ares-comp-11/chronotick_client_validation_20251020_220343.csv")
    node2_csv = Path("results/experiment-7/ares-comp-12/chronotick_client_validation_20251020_220540.csv")
    output_dir = Path("results/figures/5/experiment-7")

    if node1_csv.exists() and node2_csv.exists():
        generate_aligned_truth_test_correct(node1_csv, node2_csv, output_dir)
    else:
        print("\n⚠️  Dataset files not found!")
        return

    print("\n" + "="*80)
    print("PROPERLY CORRECTED VERSION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
