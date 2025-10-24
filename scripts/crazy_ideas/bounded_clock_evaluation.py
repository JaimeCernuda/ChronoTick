#!/usr/bin/env python3
"""
BOUNDED CLOCK EVALUATION - The Google Spanner TrueTime Narrative

Comparison:
1. System Clock (chrony-synchronized): Single-point estimate, no uncertainty
2. ChronoTick (bounded clock): Interval estimate with uncertainty bounds

The VALUE of ChronoTick is NOT just accuracy, but KNOWING when uncertain!

Like Google Spanner's TrueTime:
- Traditional: "The time is X" (single point, might be wrong)
- TrueTime/ChronoTick: "The time is between [X-ε, X+ε]" (bounded, quantified uncertainty)

This enables:
- Confident assignment when uncertainty is low
- Conservative buffering when near boundaries
- Identification of ambiguous cases
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

def load_data(node1_csv, node2_csv):
    """Load data."""
    df1 = pd.read_csv(node1_csv)
    df2 = pd.read_csv(node2_csv)

    df1_all = df1.copy()
    df2_all = df2.copy()

    df1_all['timestamp'] = pd.to_datetime(df1_all['datetime'])
    df2_all['timestamp'] = pd.to_datetime(df2_all['datetime'])

    df1_ntp = df1[df1['has_ntp'] == True].copy()
    df2_ntp = df2[df2['has_ntp'] == True].copy()
    df1_ntp['timestamp'] = pd.to_datetime(df1_ntp['datetime'])
    df2_ntp['timestamp'] = pd.to_datetime(df2_ntp['datetime'])

    start1 = df1_ntp['timestamp'].iloc[0]
    start2 = df2_ntp['timestamp'].iloc[0]
    start_offset = (start2 - start1).total_seconds()

    df1_all['elapsed_seconds'] = (df1_all['timestamp'] - start1).dt.total_seconds()
    df2_all['elapsed_seconds'] = (df2_all['timestamp'] - start2).dt.total_seconds()
    df1_ntp['elapsed_seconds'] = (df1_ntp['timestamp'] - start1).dt.total_seconds()
    df2_ntp['elapsed_seconds'] = (df2_ntp['timestamp'] - start2).dt.total_seconds()

    return {
        'node1_all': df1_all,
        'node2_all': df2_all,
        'node1_ntp': df1_ntp,
        'node2_ntp': df2_ntp,
        'start_offset': start_offset,
    }

def evaluate_bounded_clock(data, window_sizes=[100, 500, 1000, 5000], output_dir=None, exp_name=''):
    """
    Bounded clock evaluation.

    Compare:
    1. System clock (single-point, chrony-synchronized)
    2. ChronoTick (bounded clock with uncertainty)

    Metrics:
    - Agreement rate (single-point comparison)
    - Uncertainty-aware confidence (can ChronoTick identify ambiguous cases?)
    - Conservative correctness (if ChronoTick says "confident", is it right?)
    """
    print(f"\n{'='*80}")
    print(f"BOUNDED CLOCK EVALUATION: {exp_name}")
    print('='*80)

    start_offset = data['start_offset']
    node1_ntp = data['node1_ntp']
    node2_all = data['node2_all']

    all_results = {ws: [] for ws in window_sizes}

    for idx1, row1 in node1_ntp.iterrows():
        elapsed1 = row1['elapsed_seconds']
        elapsed2_target = elapsed1 - start_offset

        if elapsed2_target < 0:
            continue

        # Ground truth: Node 1 NTP time
        ntp1_time = row1['ntp_time']

        # Find Node 2's measurement at same wall-clock moment
        idx2_all = (node2_all['elapsed_seconds'] - elapsed2_target).abs().idxmin()
        if abs(node2_all.loc[idx2_all, 'elapsed_seconds'] - elapsed2_target) > 5:
            continue

        row2 = node2_all.loc[idx2_all]

        # Method 1: System clock (chrony-synchronized, single-point)
        system_time2 = row2['system_time']

        # Method 2: ChronoTick (bounded clock)
        chronotick_time2 = row2['chronotick_time']
        chronotick_unc2 = row2['chronotick_uncertainty_ms'] / 1000.0  # Convert to seconds

        # Calculate offsets relative to ground truth
        # (positive = ahead of ground truth, negative = behind)
        offset_system_ms = (system_time2 - ntp1_time) * 1000
        offset_chronotick_ms = (chronotick_time2 - ntp1_time) * 1000

        # ChronoTick bounds
        chronotick_lower_ms = offset_chronotick_ms - 3 * row2['chronotick_uncertainty_ms']
        chronotick_upper_ms = offset_chronotick_ms + 3 * row2['chronotick_uncertainty_ms']

        # Test for each window size
        for window_ms in window_sizes:
            # Calculate positions within window
            pos_truth = 0  # Ground truth is our reference (offset = 0)
            pos_system = offset_system_ms % window_ms
            pos_chronotick = offset_chronotick_ms % window_ms

            # Handle negative offsets
            if pos_system < 0:
                pos_system += window_ms
            if pos_chronotick < 0:
                pos_chronotick += window_ms

            # Calculate differences
            def window_diff(p1, p2, window_size):
                diff = abs(p1 - p2)
                if diff > window_size / 2:
                    diff = window_size - diff
                return diff

            diff_system = window_diff(0, pos_system, window_ms)
            diff_chronotick = window_diff(0, pos_chronotick, window_ms)

            # Agreement threshold
            threshold = min(10, window_ms * 0.01)

            agrees_system = (diff_system < threshold)
            agrees_chronotick = (diff_chronotick < threshold)

            # ChronoTick uncertainty-aware checks
            # Check if ENTIRE uncertainty range fits in one window
            pos_lower = chronotick_lower_ms % window_ms
            pos_upper = chronotick_upper_ms % window_ms

            if pos_lower < 0:
                pos_lower += window_ms
            if pos_upper < 0:
                pos_upper += window_ms

            # Check if bounds span window boundary
            # If lower and upper in same window → confident
            # If they span boundary → ambiguous
            diff_bounds = abs(pos_upper - pos_lower)
            if diff_bounds > window_ms / 2:
                diff_bounds = window_ms - diff_bounds

            # Ambiguous if uncertainty range is large relative to distance from boundary
            distance_from_boundary = min(pos_chronotick, window_ms - pos_chronotick)
            is_ambiguous = (3 * row2['chronotick_uncertainty_ms'] > distance_from_boundary)

            # Conservative correctness: If ChronoTick says "confident", is it correct?
            is_confident = not is_ambiguous
            confident_and_correct = is_confident and agrees_chronotick

            # Does ground truth (offset=0) fall within ChronoTick ±3σ bounds?
            truth_in_bounds = (chronotick_lower_ms <= 0 <= chronotick_upper_ms)

            all_results[window_ms].append({
                'elapsed_hours': elapsed1 / 3600,
                'offset_system_ms': offset_system_ms,
                'offset_chronotick_ms': offset_chronotick_ms,
                'chronotick_unc_ms': row2['chronotick_uncertainty_ms'],
                'chronotick_lower_ms': chronotick_lower_ms,
                'chronotick_upper_ms': chronotick_upper_ms,
                'diff_system': diff_system,
                'diff_chronotick': diff_chronotick,
                'agrees_system': agrees_system,
                'agrees_chronotick': agrees_chronotick,
                'is_ambiguous': is_ambiguous,
                'is_confident': is_confident,
                'confident_and_correct': confident_and_correct,
                'truth_in_bounds': truth_in_bounds,
                'distance_from_boundary': distance_from_boundary,
            })

    # Calculate metrics
    results_summary = {}

    for window_ms in window_sizes:
        df = pd.DataFrame(all_results[window_ms])

        if len(df) == 0:
            continue

        # Basic agreement rates
        agree_system = (df['agrees_system'].sum() / len(df) * 100)
        agree_chronotick = (df['agrees_chronotick'].sum() / len(df) * 100)

        # Bounded clock metrics
        confident_samples = df[df['is_confident'] == True]
        ambiguous_samples = df[df['is_ambiguous'] == True]

        if len(confident_samples) > 0:
            confident_correct_rate = (confident_samples['agrees_chronotick'].sum() / len(confident_samples) * 100)
        else:
            confident_correct_rate = 0

        # Truth in bounds rate
        truth_in_bounds_rate = (df['truth_in_bounds'].sum() / len(df) * 100)

        print(f"\n{'='*80}")
        print(f"Window {window_ms}ms ({len(df)} samples):")
        print('='*80)

        print(f"\n📊 SINGLE-POINT AGREEMENT (ignoring uncertainty):")
        print(f"  System Clock:  {df['agrees_system'].sum()}/{len(df)} = {agree_system:.1f}%")
        print(f"  ChronoTick:    {df['agrees_chronotick'].sum()}/{len(df)} = {agree_chronotick:.1f}%")
        print(f"  Difference:    {agree_chronotick - agree_system:+.1f}%")

        print(f"\n🎯 BOUNDED CLOCK METRICS (the value proposition!):")
        print(f"  Confident assignments:  {len(confident_samples)}/{len(df)} = {len(confident_samples)/len(df)*100:.1f}%")
        print(f"  Ambiguous (near boundary): {len(ambiguous_samples)}/{len(df)} = {len(ambiguous_samples)/len(df)*100:.1f}%")

        if len(confident_samples) > 0:
            print(f"  When confident, correct: {confident_samples['agrees_chronotick'].sum()}/{len(confident_samples)} = {confident_correct_rate:.1f}%")

        print(f"\n✅ UNCERTAINTY CALIBRATION:")
        print(f"  Truth within ±3σ bounds: {df['truth_in_bounds'].sum()}/{len(df)} = {truth_in_bounds_rate:.1f}%")

        # Median uncertainties
        median_unc = df['chronotick_unc_ms'].median()
        mean_unc = df['chronotick_unc_ms'].mean()
        print(f"  Median uncertainty: {median_unc:.2f}ms")
        print(f"  Mean uncertainty: {mean_unc:.2f}ms")

        results_summary[window_ms] = {
            'agree_system': agree_system,
            'agree_chronotick': agree_chronotick,
            'confident_rate': len(confident_samples) / len(df) * 100,
            'ambiguous_rate': len(ambiguous_samples) / len(df) * 100,
            'confident_correct_rate': confident_correct_rate,
            'truth_in_bounds_rate': truth_in_bounds_rate,
            'samples': len(df),
            'df': df
        }

    # Visualizations
    if output_dir and len(results_summary) > 0:
        create_bounded_clock_figure(results_summary, 1000, output_dir, exp_name)

    return results_summary

def create_bounded_clock_figure(results_summary, window_ms, output_dir, exp_name):
    """Create bounded clock evaluation figure."""
    if window_ms not in results_summary:
        return

    df = results_summary[window_ms]['df']

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # Panel 1: Agreement comparison
    ax1 = fig.add_subplot(gs[0, :])

    methods = ['System Clock\n(single-point)', 'ChronoTick\n(single-point,\nignoring uncertainty)']
    rates = [results_summary[window_ms]['agree_system'],
             results_summary[window_ms]['agree_chronotick']]
    colors = ['steelblue', 'green']

    bars = ax1.bar(methods, rates, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax1.axhline(90, color='red', linestyle='--', linewidth=2, label='90% target', alpha=0.7)
    ax1.set_ylabel('Window Assignment Agreement (%)', fontsize=12)
    ax1.set_title(f'(a) Single-Point Agreement: {window_ms}ms windows ({len(df)} samples)',
                  fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.3)

    # Panel 2: Bounded clock value proposition
    ax2 = fig.add_subplot(gs[1, 0])

    categories = ['Confident\nAssignments', 'Ambiguous\n(near boundary)']
    values = [results_summary[window_ms]['confident_rate'],
              results_summary[window_ms]['ambiguous_rate']]
    colors2 = ['green', 'orange']

    bars2 = ax2.bar(categories, values, color=colors2, alpha=0.7, edgecolor='black', linewidth=2)

    for bar, val in zip(bars2, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax2.set_ylabel('Percentage of Events (%)', fontsize=12)
    ax2.set_title('(b) ChronoTick Identifies Ambiguous Cases', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.grid(axis='y', alpha=0.3)

    # Panel 3: Confident correctness
    ax3 = fig.add_subplot(gs[1, 1])

    confident_correct = results_summary[window_ms]['confident_correct_rate']
    confident_incorrect = 100 - confident_correct

    ax3.bar(['Correct', 'Incorrect'], [confident_correct, confident_incorrect],
            color=['green', 'red'], alpha=0.7, edgecolor='black', linewidth=2)

    ax3.text(0, confident_correct + 2, f'{confident_correct:.1f}%',
             ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax3.text(1, confident_incorrect + 2, f'{confident_incorrect:.1f}%',
             ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax3.set_ylabel('Percentage (%)', fontsize=12)
    ax3.set_title(f'(c) When ChronoTick is "Confident", Correctness', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 105)
    ax3.grid(axis='y', alpha=0.3)

    # Panel 4: Timeline view with uncertainty bounds
    ax4 = fig.add_subplot(gs[2, :])

    # Plot offset errors
    ax4.scatter(df['elapsed_hours'], df['offset_system_ms'],
                color='steelblue', alpha=0.5, s=30, label='System clock offset')

    # ChronoTick with error bars
    confident_df = df[df['is_confident'] == True]
    ambiguous_df = df[df['is_ambiguous'] == True]

    ax4.errorbar(confident_df['elapsed_hours'], confident_df['offset_chronotick_ms'],
                 yerr=3*confident_df['chronotick_unc_ms'],
                 fmt='o', color='green', alpha=0.6, markersize=4, capsize=3,
                 label='ChronoTick (confident)', zorder=3)

    ax4.errorbar(ambiguous_df['elapsed_hours'], ambiguous_df['offset_chronotick_ms'],
                 yerr=3*ambiguous_df['chronotick_unc_ms'],
                 fmt='s', color='orange', alpha=0.7, markersize=6, capsize=3,
                 label='ChronoTick (ambiguous/near boundary)', zorder=4)

    ax4.axhline(0, color='black', linestyle='-', linewidth=2, label='Ground truth', zorder=1)
    ax4.axhline(10, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='±10ms threshold')
    ax4.axhline(-10, color='red', linestyle='--', linewidth=1.5, alpha=0.7)

    ax4.set_xlabel('Time (hours)', fontsize=12)
    ax4.set_ylabel('Offset from Ground Truth (ms)', fontsize=12)
    ax4.set_title('(d) Timeline: ChronoTick Identifies Ambiguous Cases with Uncertainty Bounds (±3σ)',
                  fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(alpha=0.3)
    ax4.set_yscale('symlog', linthresh=1)

    plt.suptitle(f'{exp_name}: Bounded Clock Evaluation (Google Spanner TrueTime Narrative)',
                 fontsize=14, fontweight='bold', y=0.995)

    output_path = Path(output_dir) / exp_name
    output_path.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path / f'{exp_name}_bounded_clock.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / f'{exp_name}_bounded_clock.pdf', bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}/{exp_name}_bounded_clock.pdf")
    plt.close()

def run_evaluation(exp_name, node1_csv, node2_csv, output_dir):
    """Run bounded clock evaluation for one experiment."""
    print(f"\n{'#'*80}")
    print(f"# {exp_name.upper()}")
    print(f"{'#'*80}")

    data = load_data(node1_csv, node2_csv)
    print(f"Loaded: {len(data['node1_ntp'])} NTP (Node 1), {len(data['node2_ntp'])} NTP (Node 2)")

    results = evaluate_bounded_clock(data, output_dir=output_dir, exp_name=exp_name)

    return results

def main():
    """Run bounded clock evaluation."""
    output_base = Path("results/figures/bounded_clock")

    experiments = {
        'experiment-5': {
            'node1': Path("results/experiment-5/ares-comp-11/data.csv"),
            'node2': Path("results/experiment-5/ares-comp-12/data.csv")
        },
        'experiment-7': {
            'node1': Path("results/experiment-7/ares-comp-11/chronotick_client_validation_20251020_220343.csv"),
            'node2': Path("results/experiment-7/ares-comp-12/chronotick_client_validation_20251020_220540.csv")
        },
        'experiment-10': {
            'node1': Path("results/experiment-10/ares-11/chronotick_client_validation_20251022_192420.csv"),
            'node2': Path("results/experiment-10/ares-12/chronotick_client_validation_20251022_192443.csv")
        }
    }

    all_results = {}

    for exp_name, paths in experiments.items():
        results = run_evaluation(exp_name, paths['node1'], paths['node2'], output_base)
        if results:
            all_results[exp_name] = results

    print(f"\n\n{'='*80}")
    print("BOUNDED CLOCK EVALUATION SUMMARY")
    print('='*80)
    print("\nThe Google Spanner TrueTime narrative:")
    print("- System clock: Single-point estimate (no uncertainty)")
    print("- ChronoTick: Bounded estimate with quantified uncertainty")
    print("\nValue proposition: ChronoTick KNOWS when it's uncertain!")
    print('='*80)

    for exp_name, results in all_results.items():
        if 1000 in results:
            r = results[1000]
            print(f"\n{exp_name}:")
            print(f"  System clock agreement:  {r['agree_system']:.1f}%")
            print(f"  ChronoTick agreement:    {r['agree_chronotick']:.1f}%")
            print(f"  Confident assignments:   {r['confident_rate']:.1f}%")
            print(f"  When confident, correct: {r['confident_correct_rate']:.1f}%")
            print(f"  Truth in ±3σ bounds:     {r['truth_in_bounds_rate']:.1f}%")

if __name__ == "__main__":
    main()
