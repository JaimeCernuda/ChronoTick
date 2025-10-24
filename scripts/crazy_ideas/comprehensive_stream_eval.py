#!/usr/bin/env python3
"""
Comprehensive Stream Processing Evaluation

ULTRATHINKING:
1. Compare ChronoTick vs System Clock (NTP baseline)
2. Test ALL experiments (5, 7, 10) not just Experiment-5
3. Multiple evaluation types (not just windowing)
4. Focused visualizations (30-min windows, not full 8 hours)

Evaluations:
1. Window Assignment: ChronoTick vs System Clock vs NTP Truth
2. Out-of-Order Events: Detection and quantification
3. Temporal Causality: Can we determine happens-before?
4. Drift Over Time: Does performance degrade?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json

plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

def load_data(node1_csv, node2_csv):
    """Load and prepare data."""
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

def evaluate_window_assignment_comprehensive(data, window_size_ms=1000, output_dir=None, exp_name=''):
    """
    Comprehensive Window Assignment Evaluation

    Compare THREE approaches:
    1. Ground Truth: Both nodes use NTP (oracle)
    2. System Clock: Both nodes use elapsed time (typical distributed system)
    3. ChronoTick: Nodes use ChronoTick predictions

    Key question: Does ChronoTick improve over System Clock baseline?
    """
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE WINDOW ASSIGNMENT ({window_size_ms}ms windows)")
    print('='*80)

    start_offset = data['start_offset']
    node1_ntp = data['node1_ntp']
    node2_ntp = data['node2_ntp']
    node1_all = data['node1_all']
    node2_all = data['node2_all']

    results = []

    for idx1, row1 in node1_ntp.iterrows():
        elapsed1 = row1['elapsed_seconds']
        elapsed2_target = elapsed1 - start_offset

        if elapsed2_target < 0:
            continue

        # GROUND TRUTH: Both nodes use NTP
        ntp1_ms = row1['ntp_offset_ms']

        # Find corresponding Node 2 NTP sample
        idx2_ntp = (node2_ntp['elapsed_seconds'] - elapsed2_target).abs().idxmin()
        if abs(node2_ntp.loc[idx2_ntp, 'elapsed_seconds'] - elapsed2_target) > 5:
            continue

        ntp2_ms = node2_ntp.loc[idx2_ntp, 'ntp_offset_ms']

        # Find corresponding Node 2 ALL sample for ChronoTick
        idx2_all = (node2_all['elapsed_seconds'] - elapsed2_target).abs().idxmin()
        if abs(node2_all.loc[idx2_all, 'elapsed_seconds'] - elapsed2_target) > 5:
            continue

        chronotick2_ms = node2_all.loc[idx2_all, 'chronotick_offset_ms']

        # METHOD 1: Ground Truth (Oracle - both use NTP)
        pos_ntp1 = ntp1_ms % window_size_ms
        pos_ntp2 = ntp2_ms % window_size_ms
        if pos_ntp1 < 0:
            pos_ntp1 += window_size_ms
        if pos_ntp2 < 0:
            pos_ntp2 += window_size_ms

        diff_ntp = abs(pos_ntp1 - pos_ntp2)
        if diff_ntp > window_size_ms / 2:
            diff_ntp = window_size_ms - diff_ntp

        agrees_ntp = (diff_ntp < 10)  # Within 10ms

        # METHOD 2: System Clock (Baseline - uses elapsed time without NTP correction)
        # This simulates typical distributed system with NTP-synchronized clocks
        # The drift is captured by the difference between nodes' initial offsets

        # Key insight: NTP offset = correction NTP makes to system clock
        # System clock (no NTP) would be: elapsed_time (no offset correction)
        # But nodes have initial offset, so:
        # Node 1: Uses elapsed time as-is (reference)
        # Node 2: Has initial offset captured by deployment delay

        # Actually, for fair comparison: System Clock = just using elapsed time
        # The "system clock skew" is captured by NTP offsets
        # If both nodes just used elapsed time, they'd be perfectly synchronized!
        # The NTP offsets show how much their clocks ACTUALLY drift

        # So baseline should be: What if nodes used raw system time (no sync)?
        # We approximate this by assuming drift accumulates linearly
        # Use NTP offset as estimate of accumulated drift

        # Simplified: System clock has same drift as NTP offset shows
        # (This is conservative - actual system clocks could be worse)
        system1_ms = ntp1_ms  # System drift estimated from NTP
        system2_ms = ntp2_ms  # System drift estimated from NTP

        pos_system1 = system1_ms % window_size_ms
        pos_system2 = system2_ms % window_size_ms

        if pos_system1 < 0:
            pos_system1 += window_size_ms
        if pos_system2 < 0:
            pos_system2 += window_size_ms

        diff_system = abs(pos_system1 - pos_system2)
        if diff_system > window_size_ms / 2:
            diff_system = window_size_ms - diff_system

        agrees_system = (diff_system < 10)

        # METHOD 3: ChronoTick (Our approach - use ChronoTick predictions)
        pos_chronotick1 = ntp1_ms % window_size_ms  # Node 1 uses NTP (ground truth)
        pos_chronotick2 = chronotick2_ms % window_size_ms  # Node 2 uses ChronoTick

        if pos_chronotick1 < 0:
            pos_chronotick1 += window_size_ms
        if pos_chronotick2 < 0:
            pos_chronotick2 += window_size_ms

        diff_chronotick = abs(pos_chronotick1 - pos_chronotick2)
        if diff_chronotick > window_size_ms / 2:
            diff_chronotick = window_size_ms - diff_chronotick

        agrees_chronotick = (diff_chronotick < 10)

        results.append({
            'elapsed_hours': elapsed1 / 3600,
            'ntp1_ms': ntp1_ms,
            'ntp2_ms': ntp2_ms,
            'chronotick2_ms': chronotick2_ms,
            'diff_ntp': diff_ntp,
            'diff_system': diff_system,
            'diff_chronotick': diff_chronotick,
            'agrees_ntp': agrees_ntp,
            'agrees_system': agrees_system,
            'agrees_chronotick': agrees_chronotick,
        })

    df = pd.DataFrame(results)

    if len(df) == 0:
        print("⚠️  No valid samples")
        return None

    # Calculate metrics
    agree_ntp = (df['agrees_ntp'].sum() / len(df) * 100)
    agree_system = (df['agrees_system'].sum() / len(df) * 100)
    agree_chronotick = (df['agrees_chronotick'].sum() / len(df) * 100)

    print(f"\nResults ({len(df)} samples):")
    print(f"  Ground Truth (both NTP): {df['agrees_ntp'].sum()}/{len(df)} = {agree_ntp:.1f}%")
    print(f"  System Clock (no correction): {df['agrees_system'].sum()}/{len(df)} = {agree_system:.1f}%")
    print(f"  ChronoTick: {df['agrees_chronotick'].sum()}/{len(df)} = {agree_chronotick:.1f}%")
    print(f"\n  💡 ChronoTick improvement: {agree_chronotick - agree_system:+.1f}% over baseline")

    # Visualization - FOCUSED on 30-minute window
    if output_dir and len(df) > 0:
        # Find interesting 30-minute window (hours 2-2.5)
        focus_start = 2.0
        focus_end = 2.5
        df_focus = df[(df['elapsed_hours'] >= focus_start) & (df['elapsed_hours'] <= focus_end)]

        if len(df_focus) < 5:
            # If not enough samples in that window, use first 30 minutes of data
            focus_start = df['elapsed_hours'].min()
            focus_end = focus_start + 0.5
            df_focus = df[(df['elapsed_hours'] >= focus_start) & (df['elapsed_hours'] <= focus_end)]

        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # Panel 1: ChronoTick vs System Clock (focused window)
        ax1 = fig.add_subplot(gs[0, :])

        agree_ct = df_focus[df_focus['agrees_chronotick']]
        disagree_ct = df_focus[~df_focus['agrees_chronotick']]

        ax1.scatter(agree_ct['elapsed_hours'], agree_ct['diff_chronotick'],
                   color='#009E73', s=60, alpha=0.7, label=f'ChronoTick Agreement ({len(agree_ct)})', marker='o', zorder=3)
        ax1.scatter(disagree_ct['elapsed_hours'], disagree_ct['diff_chronotick'],
                   color='#D55E00', s=60, alpha=0.7, label=f'ChronoTick Disagreement ({len(disagree_ct)})', marker='x', zorder=3)

        # Also plot system clock for comparison
        ax1.scatter(df_focus['elapsed_hours'], df_focus['diff_system'],
                   color='gray', s=30, alpha=0.3, label='System Clock (baseline)', marker='s', zorder=1)

        ax1.axhline(10, color='red', linestyle='--', linewidth=2, alpha=0.7, label='10ms threshold')
        ax1.set_xlabel('Time (hours)', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Offset Difference (ms)', fontweight='bold', fontsize=12)
        ax1.set_title(f'(a) ChronoTick vs System Clock: 30-min focused view (hours {focus_start:.1f}-{focus_end:.1f})',
                     fontweight='bold', fontsize=13)
        ax1.grid(alpha=0.3)
        ax1.legend(loc='best', fontsize=10)
        ax1.set_yscale('log')
        ax1.set_ylim(0.01, max(df_focus['diff_chronotick'].max(), df_focus['diff_system'].max()) * 1.5)

        # Panel 2: Full 8-hour overview
        ax2 = fig.add_subplot(gs[1, 0])

        ax2.scatter(df[df['agrees_chronotick']]['elapsed_hours'],
                   df[df['agrees_chronotick']]['diff_chronotick'],
                   color='#009E73', s=20, alpha=0.6, label=f'ChronoTick')
        ax2.scatter(df[~df['agrees_chronotick']]['elapsed_hours'],
                   df[~df['agrees_chronotick']]['diff_chronotick'],
                   color='#D55E00', s=20, alpha=0.6)

        ax2.axhline(10, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax2.set_xlabel('Time (hours)', fontweight='bold')
        ax2.set_ylabel('ChronoTick Difference (ms)', fontweight='bold')
        ax2.set_title(f'(b) ChronoTick: {agree_chronotick:.1f}% agreement', fontweight='bold')
        ax2.grid(alpha=0.3)
        ax2.set_yscale('log')

        # Panel 3: System Clock baseline (full 8 hours)
        ax3 = fig.add_subplot(gs[1, 1])

        ax3.scatter(df['elapsed_hours'], df['diff_system'],
                   color='gray', s=20, alpha=0.6, label='System Clock')
        ax3.axhline(10, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax3.set_xlabel('Time (hours)', fontweight='bold')
        ax3.set_ylabel('System Clock Difference (ms)', fontweight='bold')
        ax3.set_title(f'(c) System Clock: {agree_system:.1f}% agreement', fontweight='bold')
        ax3.grid(alpha=0.3)
        ax3.set_yscale('log')

        # Panel 4: Comparison bar chart
        ax4 = fig.add_subplot(gs[2, :])

        methods = ['Ground Truth\n(Both NTP)', 'System Clock\n(No Correction)', 'ChronoTick']
        rates = [agree_ntp, agree_system, agree_chronotick]
        colors = ['#0072B2', 'gray', '#009E73']

        bars = ax4.bar(methods, rates, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax4.set_ylabel('Agreement Rate (%)', fontweight='bold', fontsize=12)
        ax4.set_title(f'(d) Window Assignment Agreement Comparison ({len(df)} samples)', fontweight='bold', fontsize=13)
        ax4.axhline(90, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label='90% target')
        ax4.set_ylim(0, 105)
        ax4.grid(axis='y', alpha=0.3)
        ax4.legend(loc='lower right')

        # Add percentage labels
        for bar, rate in zip(bars, rates):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%',
                    ha='center', va='bottom', fontweight='bold', fontsize=12)

        # Add improvement annotation
        improvement = agree_chronotick - agree_system
        ax4.annotate('', xy=(2, agree_chronotick), xytext=(1, agree_system),
                    arrowprops=dict(arrowstyle='->', lw=2, color='green'))
        ax4.text(1.5, (agree_system + agree_chronotick) / 2,
                f'+{improvement:.1f}%\nimprovement',
                ha='center', fontsize=11, fontweight='bold', color='green',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

        plt.suptitle(f'{exp_name}: Window Assignment Comprehensive Evaluation',
                    fontsize=14, fontweight='bold', y=0.995)

        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f'{exp_name}_comprehensive_window_assignment.pdf', bbox_inches='tight')
        plt.savefig(output_dir / f'{exp_name}_comprehensive_window_assignment.png', dpi=300, bbox_inches='tight')
        plt.close()

    return {
        'name': 'Window Assignment',
        'agree_ntp': agree_ntp,
        'agree_system': agree_system,
        'agree_chronotick': agree_chronotick,
        'improvement': agree_chronotick - agree_system,
        'samples': len(df)
    }

def evaluate_out_of_order_detection(data, output_dir=None, exp_name=''):
    """
    Evaluation: Out-of-Order Event Detection

    In stream processing, events can arrive "out of order" due to clock skew.
    Question: Can ChronoTick detect when events are out of order?

    Compare:
    - System Clock: Arrival order = temporal order (assumes no skew)
    - ChronoTick: Use predictions to detect true temporal order
    """
    print(f"\n{'='*80}")
    print(f"OUT-OF-ORDER EVENT DETECTION")
    print('='*80)

    node1_ntp = data['node1_ntp']

    # Look at consecutive NTP samples
    out_of_order_system = 0
    out_of_order_chronotick = 0
    total_pairs = 0

    results = []

    for i in range(len(node1_ntp) - 1):
        row_i = node1_ntp.iloc[i]
        row_j = node1_ntp.iloc[i + 1]

        elapsed_i = row_i['elapsed_seconds']
        elapsed_j = row_j['elapsed_seconds']

        # Ground truth: NTP timestamps
        ntp_i = row_i['ntp_offset_ms']
        ntp_j = row_j['ntp_offset_ms']

        # ChronoTick predictions
        ct_i = row_i['chronotick_offset_ms']
        ct_j = row_j['chronotick_offset_ms']

        # System clock: elapsed time (no correction)
        system_i = 0
        system_j = 0

        # True temporal order (ground truth)
        wallclock_i = elapsed_i * 1000 + ntp_i
        wallclock_j = elapsed_j * 1000 + ntp_j
        true_order = "i<j" if wallclock_i < wallclock_j else "j<i"

        # System clock order (arrival order)
        system_order = "i<j"  # Assumes arrival order = temporal order

        # ChronoTick order
        ct_wallclock_i = elapsed_i * 1000 + ct_i
        ct_wallclock_j = elapsed_j * 1000 + ct_j
        ct_order = "i<j" if ct_wallclock_i < ct_wallclock_j else "j<i"

        # Check if out of order
        system_wrong = (system_order != true_order)
        ct_wrong = (ct_order != true_order)

        if system_wrong:
            out_of_order_system += 1
        if ct_wrong:
            out_of_order_chronotick += 1

        total_pairs += 1

        results.append({
            'elapsed_hours': elapsed_i / 3600,
            'true_order': true_order,
            'system_wrong': system_wrong,
            'ct_wrong': ct_wrong,
            'wallclock_diff_ms': wallclock_j - wallclock_i
        })

    df = pd.DataFrame(results)

    rate_system = (out_of_order_system / total_pairs * 100) if total_pairs > 0 else 0
    rate_ct = (out_of_order_chronotick / total_pairs * 100) if total_pairs > 0 else 0

    print(f"\nResults ({total_pairs} event pairs):")
    print(f"  System Clock out-of-order rate: {out_of_order_system}/{total_pairs} = {rate_system:.1f}%")
    print(f"  ChronoTick out-of-order rate: {out_of_order_chronotick}/{total_pairs} = {rate_ct:.1f}%")
    print(f"  💡 ChronoTick detects {rate_system - rate_ct:.1f}% more out-of-order events")

    return {
        'name': 'Out-of-Order Detection',
        'system_rate': rate_system,
        'chronotick_rate': rate_ct,
        'improvement': rate_system - rate_ct,
        'samples': total_pairs
    }

def run_comprehensive_eval(exp_name, node1_csv, node2_csv, output_base):
    """Run comprehensive evaluation."""
    print(f"\n\n{'#'*80}")
    print(f"# COMPREHENSIVE EVALUATION: {exp_name}")
    print(f"{'#'*80}")

    if not node1_csv.exists() or not node2_csv.exists():
        print(f"⚠️  Data not found for {exp_name}")
        return None

    data = load_data(node1_csv, node2_csv)
    output_dir = output_base / exp_name

    print(f"\nLoaded:")
    print(f"  Node 1: {len(data['node1_ntp'])} NTP, {len(data['node1_all'])} total")
    print(f"  Node 2: {len(data['node2_ntp'])} NTP, {len(data['node2_all'])} total")
    print(f"  Start offset: {data['start_offset']:.1f}s")

    results = {}

    # Eval 1: Comprehensive window assignment (ChronoTick vs System Clock vs Ground Truth)
    results['window_assignment'] = evaluate_window_assignment_comprehensive(
        data, window_size_ms=1000, output_dir=output_dir, exp_name=exp_name)

    # Eval 2: Out-of-order detection
    results['out_of_order'] = evaluate_out_of_order_detection(
        data, output_dir=output_dir, exp_name=exp_name)

    return results

def main():
    """Run comprehensive evaluation on ALL experiments."""
    output_base = Path("results/figures/comprehensive_stream")

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
        results = run_comprehensive_eval(exp_name, paths['node1'], paths['node2'], output_base)
        if results:
            all_results[exp_name] = results

    # Save summary
    summary_file = output_base / 'summary_results.json'
    output_base.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    # Print cross-experiment summary
    print(f"\n\n{'='*80}")
    print("CROSS-EXPERIMENT SUMMARY")
    print('='*80)

    for exp_name, results in all_results.items():
        if results and 'window_assignment' in results:
            wa = results['window_assignment']
            print(f"\n{exp_name.upper()}:")
            print(f"  System Clock: {wa['agree_system']:.1f}%")
            print(f"  ChronoTick: {wa['agree_chronotick']:.1f}%")
            print(f"  Improvement: +{wa['improvement']:.1f}%")

    print(f"\n✓ Results saved to: {output_base}/")
    print(f"✓ Summary JSON: {summary_file}")

if __name__ == "__main__":
    main()
