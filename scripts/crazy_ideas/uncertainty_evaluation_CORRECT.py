#!/usr/bin/env python3
"""
CORRECTED Uncertainty-Aware Distributed Coordination Evaluation

Fixed conceptual errors:
1. Consensus Zones: Both nodes use ChronoTick (not mixing NTP truth)
2. Distributed Lock: Both nodes use ChronoTick (not one using NTP truth)
3. Added bidirectional testing
4. Individual visualizations for each test
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

def load_and_prepare_data(node1_csv, node2_csv):
    """Load and align data from both nodes."""
    df1 = pd.read_csv(node1_csv)
    df2 = pd.read_csv(node2_csv)

    # Node 1: NTP samples (for ground truth)
    df1_ntp = df1[df1['has_ntp'] == True].copy()
    # Node 1: ALL samples (for ChronoTick predictions)
    df1_all = df1.copy()

    # Node 2: NTP samples (for ground truth)
    df2_ntp = df2[df2['has_ntp'] == True].copy()
    # Node 2: ALL samples (for ChronoTick predictions)
    df2_all = df2.copy()

    df1_ntp['timestamp'] = pd.to_datetime(df1_ntp['datetime'])
    df1_all['timestamp'] = pd.to_datetime(df1_all['datetime'])
    df2_ntp['timestamp'] = pd.to_datetime(df2_ntp['datetime'])
    df2_all['timestamp'] = pd.to_datetime(df2_all['datetime'])

    start1 = df1_ntp['timestamp'].iloc[0]
    start2 = df2_ntp['timestamp'].iloc[0]
    start_offset = (start2 - start1).total_seconds()

    df1_ntp['elapsed_seconds'] = (df1_ntp['timestamp'] - start1).dt.total_seconds()
    df1_all['elapsed_seconds'] = (df1_all['timestamp'] - start1).dt.total_seconds()
    df2_ntp['elapsed_seconds'] = (df2_ntp['timestamp'] - start2).dt.total_seconds()
    df2_all['elapsed_seconds'] = (df2_all['timestamp'] - start2).dt.total_seconds()

    return {
        'node1_ntp': df1_ntp,
        'node1_all': df1_all,
        'node2_ntp': df2_ntp,
        'node2_all': df2_all,
        'start_offset': start_offset,
        'start1': start1,
        'start2': start2
    }

def test_bidirectional_alignment(data, output_dir, exp_name):
    """
    Test 1: Bidirectional Timeline Alignment

    Test both directions:
    - N1 NTP truth → N2 ChronoTick prediction
    - N2 NTP truth → N1 ChronoTick prediction

    This validates that cross-node predictions are accurate.
    """
    print(f"\n{'='*80}")
    print("TEST 1: BIDIRECTIONAL TIMELINE ALIGNMENT")
    print('='*80)

    start_offset = data['start_offset']

    # Test 1→2: Node 1 NTP → Node 2 ChronoTick
    test1_data = []
    for idx1, row1 in data['node1_ntp'].iterrows():
        elapsed1 = row1['elapsed_seconds']
        elapsed2_target = elapsed1 - start_offset

        if elapsed2_target < 0:
            continue

        idx2 = (data['node2_all']['elapsed_seconds'] - elapsed2_target).abs().idxmin()
        if abs(data['node2_all'].loc[idx2, 'elapsed_seconds'] - elapsed2_target) > 5:
            continue

        ntp1 = row1['ntp_offset_ms']
        pred2 = data['node2_all'].loc[idx2, 'chronotick_offset_ms']
        unc2 = data['node2_all'].loc[idx2, 'chronotick_uncertainty_ms']

        agrees = (ntp1 >= pred2 - 3*unc2) and (ntp1 <= pred2 + 3*unc2)

        test1_data.append({
            'elapsed': elapsed1 / 3600,
            'ntp_truth': ntp1,
            'prediction': pred2,
            'uncertainty': unc2,
            'agrees': agrees
        })

    # Test 2→1: Node 2 NTP → Node 1 ChronoTick
    test2_data = []
    for idx2, row2 in data['node2_ntp'].iterrows():
        elapsed2 = row2['elapsed_seconds']
        elapsed1_target = elapsed2 + start_offset

        idx1 = (data['node1_all']['elapsed_seconds'] - elapsed1_target).abs().idxmin()
        if abs(data['node1_all'].loc[idx1, 'elapsed_seconds'] - elapsed1_target) > 5:
            continue

        ntp2 = row2['ntp_offset_ms']
        pred1 = data['node1_all'].loc[idx1, 'chronotick_offset_ms']
        unc1 = data['node1_all'].loc[idx1, 'chronotick_uncertainty_ms']

        agrees = (ntp2 >= pred1 - 3*unc1) and (ntp2 <= pred1 + 3*unc1)

        test2_data.append({
            'elapsed': elapsed1_target / 3600,
            'ntp_truth': ntp2,
            'prediction': pred1,
            'uncertainty': unc1,
            'agrees': agrees
        })

    df1 = pd.DataFrame(test1_data)
    df2 = pd.DataFrame(test2_data)

    rate1 = (df1['agrees'].sum() / len(df1) * 100) if len(df1) > 0 else 0
    rate2 = (df2['agrees'].sum() / len(df2) * 100) if len(df2) > 0 else 0

    print(f"Test 1→2: {df1['agrees'].sum()}/{len(df1)} = {rate1:.1f}%")
    print(f"Test 2→1: {df2['agrees'].sum()}/{len(df2)} = {rate2:.1f}%")

    # Visualization
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Panel 1: N1 → N2
    for idx, row in df1.iterrows():
        color = '#009E73' if row['agrees'] else '#D55E00'
        ax1.errorbar(row['elapsed'], row['prediction'], yerr=3*row['uncertainty'],
                    fmt='s', color=color, alpha=0.6, markersize=4)
        ax1.plot(row['elapsed'], row['ntp_truth'], 'o', color='#0072B2', markersize=3)

    ax1.set_xlabel('Time (hours)', fontweight='bold')
    ax1.set_ylabel('Clock Offset (ms)', fontweight='bold')
    ax1.set_title(f'(a) Node 1 NTP Truth → Node 2 ChronoTick: {rate1:.1f}% agreement ({len(df1)} samples)',
                 fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.legend(['NTP Truth (N1)', 'ChronoTick ±3σ (N2)'], loc='best')

    # Panel 2: N2 → N1
    for idx, row in df2.iterrows():
        color = '#009E73' if row['agrees'] else '#D55E00'
        ax2.errorbar(row['elapsed'], row['prediction'], yerr=3*row['uncertainty'],
                    fmt='s', color=color, alpha=0.6, markersize=4)
        ax2.plot(row['elapsed'], row['ntp_truth'], 'o', color='#D55E00', markersize=3)

    ax2.set_xlabel('Time (hours)', fontweight='bold')
    ax2.set_ylabel('Clock Offset (ms)', fontweight='bold')
    ax2.set_title(f'(b) Node 2 NTP Truth → Node 1 ChronoTick: {rate2:.1f}% agreement ({len(df2)} samples)',
                 fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.legend(['NTP Truth (N2)', 'ChronoTick ±3σ (N1)'], loc='best')

    plt.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'{exp_name}_test1_bidirectional_alignment.pdf', bbox_inches='tight')
    plt.savefig(output_dir / f'{exp_name}_test1_bidirectional_alignment.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'name': 'Bidirectional Timeline Alignment',
        'test1_rate': rate1,
        'test2_rate': rate2,
        'overall_rate': (rate1 + rate2) / 2,
        'test1_samples': len(df1),
        'test2_samples': len(df2)
    }

def test_consensus_windows_CORRECT(data, output_dir, exp_name):
    """
    Test 2: Consensus Windows (CORRECTED)

    BOTH nodes use ChronoTick predictions ±3σ at same wall-clock moments.
    Do their uncertainty ranges overlap?

    This tests if nodes can identify "consensus zones" where they agree
    events are concurrent within uncertainty.
    """
    print(f"\n{'='*80}")
    print("TEST 2: CONSENSUS WINDOWS (CORRECTED - BOTH USE CHRONOTICK)")
    print('='*80)

    start_offset = data['start_offset']

    consensus_data = []

    # Use Node 1 ALL samples and find corresponding Node 2 ALL samples
    for idx1, row1 in data['node1_all'].iterrows():
        elapsed1 = row1['elapsed_seconds']
        elapsed2_target = elapsed1 - start_offset

        if elapsed2_target < 0:
            continue

        idx2 = (data['node2_all']['elapsed_seconds'] - elapsed2_target).abs().idxmin()
        if abs(data['node2_all'].loc[idx2, 'elapsed_seconds'] - elapsed2_target) > 5:
            continue

        # BOTH use ChronoTick predictions
        pred1 = row1['chronotick_offset_ms']
        unc1 = row1['chronotick_uncertainty_ms']
        range1 = (pred1 - 3*unc1, pred1 + 3*unc1)

        pred2 = data['node2_all'].loc[idx2, 'chronotick_offset_ms']
        unc2 = data['node2_all'].loc[idx2, 'chronotick_uncertainty_ms']
        range2 = (pred2 - 3*unc2, pred2 + 3*unc2)

        # Check overlap
        overlaps = (range1[1] >= range2[0]) and (range2[1] >= range1[0])

        consensus_data.append({
            'elapsed': elapsed1 / 3600,
            'pred1': pred1,
            'unc1': unc1,
            'pred2': pred2,
            'unc2': unc2,
            'overlaps': overlaps
        })

    df = pd.DataFrame(consensus_data)
    rate = (df['overlaps'].sum() / len(df) * 100) if len(df) > 0 else 0

    print(f"Consensus windows: {df['overlaps'].sum()}/{len(df)} = {rate:.1f}%")

    # Visualization: Show first 100 samples
    fig, ax = plt.subplots(figsize=(14, 6))

    sample_df = df.head(100)

    for idx, row in sample_df.iterrows():
        color = '#009E73' if row['overlaps'] else '#D55E00'
        x = row['elapsed']

        # Node 1 range
        ax.plot([x, x], [row['pred1'] - 3*row['unc1'], row['pred1'] + 3*row['unc1']],
               color='#0072B2', linewidth=2, alpha=0.7)
        ax.plot(x, row['pred1'], 'o', color='#0072B2', markersize=4)

        # Node 2 range
        ax.plot([x+0.01, x+0.01], [row['pred2'] - 3*row['unc2'], row['pred2'] + 3*row['unc2']],
               color='#E69F00', linewidth=2, alpha=0.7)
        ax.plot(x+0.01, row['pred2'], 's', color='#E69F00', markersize=4)

        # Mark overlap/no overlap
        if row['overlaps']:
            ax.axvspan(x-0.005, x+0.015, alpha=0.1, color='green')

    ax.set_xlabel('Time (hours)', fontweight='bold')
    ax.set_ylabel('Clock Offset (ms)', fontweight='bold')
    ax.set_title(f'Consensus Windows: {rate:.1f}% overlap ({len(df)} total samples, showing first 100)',
                fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(['Node 1 ChronoTick ±3σ', 'Node 2 ChronoTick ±3σ'], loc='best')

    plt.tight_layout()
    plt.savefig(output_dir / f'{exp_name}_test2_consensus_windows.pdf', bbox_inches='tight')
    plt.savefig(output_dir / f'{exp_name}_test2_consensus_windows.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'name': 'Consensus Windows (ChronoTick vs ChronoTick)',
        'rate': rate,
        'samples': len(df)
    }

def test_uncertainty_calibration(data, output_dir, exp_name):
    """
    Test 3: Uncertainty Calibration

    Test both nodes' ChronoTick uncertainty calibration.
    Do ±1σ, ±2σ, ±3σ bounds contain ground truth at expected rates?
    """
    print(f"\n{'='*80}")
    print("TEST 3: UNCERTAINTY CALIBRATION")
    print('='*80)

    results_both = []

    for node_name, df_ntp in [('Node 1', data['node1_ntp']), ('Node 2', data['node2_ntp'])]:
        within_1sigma = 0
        within_2sigma = 0
        within_3sigma = 0
        total = 0

        for idx, row in df_ntp.iterrows():
            ntp_truth = row['ntp_offset_ms']
            chronotick_pred = row['chronotick_offset_ms']
            unc = row['chronotick_uncertainty_ms']

            error = abs(ntp_truth - chronotick_pred)

            if error <= 1 * unc:
                within_1sigma += 1
            if error <= 2 * unc:
                within_2sigma += 1
            if error <= 3 * unc:
                within_3sigma += 1

            total += 1

        rate_1s = within_1sigma / total * 100
        rate_2s = within_2sigma / total * 100
        rate_3s = within_3sigma / total * 100

        print(f"\n{node_name}:")
        print(f"  ±1σ: {within_1sigma}/{total} = {rate_1s:.1f}% (expect ~68%)")
        print(f"  ±2σ: {within_2sigma}/{total} = {rate_2s:.1f}% (expect ~95%)")
        print(f"  ±3σ: {within_3sigma}/{total} = {rate_3s:.1f}% (expect ~99.7%)")

        results_both.append({
            'node': node_name,
            'rate_1s': rate_1s,
            'rate_2s': rate_2s,
            'rate_3s': rate_3s,
            'total': total
        })

    # Visualization: Calibration plot
    fig, ax = plt.subplots(figsize=(10, 6))

    sigma_levels = [1, 2, 3]
    expected = [68.27, 95.45, 99.73]

    node1_rates = [results_both[0]['rate_1s'], results_both[0]['rate_2s'], results_both[0]['rate_3s']]
    node2_rates = [results_both[1]['rate_1s'], results_both[1]['rate_2s'], results_both[1]['rate_3s']]

    x = np.arange(len(sigma_levels))
    width = 0.25

    ax.bar(x - width, expected, width, label='Expected (Gaussian)', color='gray', alpha=0.5)
    ax.bar(x, node1_rates, width, label='Node 1 Observed', color='#0072B2')
    ax.bar(x + width, node2_rates, width, label='Node 2 Observed', color='#D55E00')

    ax.set_ylabel('Coverage (%)', fontweight='bold')
    ax.set_xlabel('Confidence Level', fontweight='bold')
    ax.set_title('Uncertainty Calibration: Expected vs Observed Coverage', fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['±1σ (68%)', '±2σ (95%)', '±3σ (99.7%)'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    plt.savefig(output_dir / f'{exp_name}_test3_calibration.pdf', bbox_inches='tight')
    plt.savefig(output_dir / f'{exp_name}_test3_calibration.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'name': 'Uncertainty Calibration',
        'node1_3sigma': results_both[0]['rate_3s'],
        'node2_3sigma': results_both[1]['rate_3s'],
        'samples': results_both[0]['total'] + results_both[1]['total']
    }

def test_distributed_lock_CORRECT(data, output_dir, exp_name):
    """
    Test 4: Distributed Lock Agreement (CORRECTED)

    BOTH nodes use ChronoTick pessimistic timestamps (pred + 3σ).
    Compare against ground truth lock ordering (both NTP values).

    Since we can't get both NTP at same moment, we use:
    - Ground truth: Node 1 NTP vs Node 2 ChronoTick (best available truth estimate)
    - ChronoTick: Node 1 ChronoTick+3σ vs Node 2 ChronoTick+3σ

    This tests if pessimistic coordination produces correct ordering.
    """
    print(f"\n{'='*80}")
    print("TEST 4: DISTRIBUTED LOCK AGREEMENT (CORRECTED - BOTH USE CHRONOTICK)")
    print('='*80)

    start_offset = data['start_offset']

    lock_data = []

    for idx1, row1 in data['node1_ntp'].iterrows():
        elapsed1 = row1['elapsed_seconds']
        elapsed2_target = elapsed1 - start_offset

        if elapsed2_target < 0:
            continue

        idx2 = (data['node2_all']['elapsed_seconds'] - elapsed2_target).abs().idxmin()
        if abs(data['node2_all'].loc[idx2, 'elapsed_seconds'] - elapsed2_target) > 5:
            continue

        # Ground truth: Node 1 NTP vs Node 2 ChronoTick (best estimate)
        ntp1 = row1['ntp_offset_ms']
        pred2 = data['node2_all'].loc[idx2, 'chronotick_offset_ms']

        # ChronoTick pessimistic: BOTH use ChronoTick + 3σ
        pred1 = row1['chronotick_offset_ms']
        unc1 = row1['chronotick_uncertainty_ms']
        unc2 = data['node2_all'].loc[idx2, 'chronotick_uncertainty_ms']

        time1_pessimistic = pred1 + 3 * unc1
        time2_pessimistic = pred2 + 3 * unc2

        # Determine winners
        winner_truth = "node1" if ntp1 < pred2 else "node2"
        winner_pessimistic = "node1" if time1_pessimistic < time2_pessimistic else "node2"

        agrees = (winner_truth == winner_pessimistic)

        lock_data.append({
            'elapsed': elapsed1 / 3600,
            'ntp1': ntp1,
            'pred1': pred1,
            'pred2': pred2,
            'pess1': time1_pessimistic,
            'pess2': time2_pessimistic,
            'winner_truth': winner_truth,
            'winner_pess': winner_pessimistic,
            'agrees': agrees
        })

    df = pd.DataFrame(lock_data)
    rate = (df['agrees'].sum() / len(df) * 100) if len(df) > 0 else 0

    print(f"Lock agreement: {df['agrees'].sum()}/{len(df)} = {rate:.1f}%")

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))

    agree_df = df[df['agrees']]
    disagree_df = df[~df['agrees']]

    ax.scatter(agree_df['elapsed'], agree_df['pess1'] - agree_df['pess2'],
              color='#009E73', s=30, alpha=0.6, label=f'Agreement ({len(agree_df)})')
    ax.scatter(disagree_df['elapsed'], disagree_df['pess1'] - disagree_df['pess2'],
              color='#D55E00', s=30, alpha=0.6, label=f'Disagreement ({len(disagree_df)})')

    ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Time (hours)', fontweight='bold')
    ax.set_ylabel('Pessimistic Time Difference (N1 - N2) [ms]', fontweight='bold')
    ax.set_title(f'Distributed Lock Agreement: {rate:.1f}% ({len(df)} samples)', fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_dir / f'{exp_name}_test4_distributed_lock.pdf', bbox_inches='tight')
    plt.savefig(output_dir / f'{exp_name}_test4_distributed_lock.png', dpi=300, bbox_inches='tight')
    plt.close()

    return {
        'name': 'Distributed Lock Agreement (ChronoTick vs ChronoTick)',
        'rate': rate,
        'samples': len(df)
    }

def run_experiment(exp_name, node1_csv, node2_csv, output_base):
    """Run all tests for one experiment."""
    print(f"\n\n{'#'*80}")
    print(f"# EXPERIMENT: {exp_name}")
    print(f"{'#'*80}")

    if not node1_csv.exists() or not node2_csv.exists():
        print(f"⚠️  Data not found for {exp_name}")
        return None

    data = load_and_prepare_data(node1_csv, node2_csv)

    print(f"\nLoaded:")
    print(f"  Node 1: {len(data['node1_ntp'])} NTP, {len(data['node1_all'])} total samples")
    print(f"  Node 2: {len(data['node2_ntp'])} NTP, {len(data['node2_all'])} total samples")
    print(f"  Start offset: {data['start_offset']:.1f}s")

    output_dir = output_base / exp_name

    results = {}
    results['test1'] = test_bidirectional_alignment(data, output_dir, exp_name)
    results['test2'] = test_consensus_windows_CORRECT(data, output_dir, exp_name)
    results['test3'] = test_uncertainty_calibration(data, output_dir, exp_name)
    results['test4'] = test_distributed_lock_CORRECT(data, output_dir, exp_name)

    return results

def main():
    """Run all experiments."""
    output_base = Path("results/figures/crazy_ideas_CORRECT")

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
        results = run_experiment(exp_name, paths['node1'], paths['node2'], output_base)
        if results:
            all_results[exp_name] = results

    # Save summary
    summary_file = output_base / 'summary_results.json'
    with open(summary_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\n\n{'='*80}")
    print("SUMMARY ACROSS ALL EXPERIMENTS")
    print('='*80)

    for exp_name, results in all_results.items():
        print(f"\n{exp_name.upper()}:")
        print(f"  Test 1 (Bidirectional): {results['test1']['overall_rate']:.1f}% avg")
        print(f"  Test 2 (Consensus): {results['test2']['rate']:.1f}%")
        print(f"  Test 3 (Calibration): {results['test3']['node1_3sigma']:.1f}% / {results['test3']['node2_3sigma']:.1f}%")
        print(f"  Test 4 (Lock): {results['test4']['rate']:.1f}%")

    print(f"\n✓ Results saved to: {output_base}/")
    print(f"✓ Summary JSON: {summary_file}")

if __name__ == "__main__":
    main()
