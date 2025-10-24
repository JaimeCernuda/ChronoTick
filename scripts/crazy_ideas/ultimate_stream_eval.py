#!/usr/bin/env python3
"""
ULTIMATE Stream Processing Evaluation

Clear comparison:
1. Ground Truth: Node 1 NTP (oracle)
2. Stale NTP: Node 2's last NTP measurement (realistic baseline)
3. ChronoTick: Node 2's current prediction (our approach)

Multiple tests:
- Window assignment (100ms, 500ms, 1000ms, 5000ms)
- Drift visualization
- Uncertainty bounds
- All 3 experiments

Goal: Show ChronoTick beats "Stale NTP" baseline
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

def find_last_ntp_before(target_elapsed, df_ntp):
    """Find the last NTP measurement before target time (for stale baseline)."""
    before = df_ntp[df_ntp['elapsed_seconds'] <= target_elapsed]
    if len(before) == 0:
        return None
    return before.iloc[-1]

def evaluate_window_assignment_ultimate(data, window_sizes=[100, 500, 1000, 5000],
                                        output_dir=None, exp_name=''):
    """
    ULTIMATE window assignment evaluation.

    For each Node 1 NTP sample (ground truth):
    - Compare against Node 2 ChronoTick prediction (our method)
    - Compare against Node 2's last NTP (stale baseline)
    - Compare against Node 2's current NTP if available (oracle)

    Test multiple window sizes to show sensitivity.
    """
    print(f"\n{'='*80}")
    print(f"ULTIMATE WINDOW ASSIGNMENT EVALUATION")
    print('='*80)

    start_offset = data['start_offset']
    node1_ntp = data['node1_ntp']
    node2_ntp = data['node2_ntp']
    node2_all = data['node2_all']

    all_results = {ws: [] for ws in window_sizes}

    for idx1, row1 in node1_ntp.iterrows():
        elapsed1 = row1['elapsed_seconds']
        elapsed2_target = elapsed1 - start_offset

        if elapsed2_target < 0:
            continue

        # Ground truth: Node 1 NTP
        ntp1_ms = row1['ntp_offset_ms']

        # Find Node 2's ChronoTick prediction at this moment
        idx2_all = (node2_all['elapsed_seconds'] - elapsed2_target).abs().idxmin()
        if abs(node2_all.loc[idx2_all, 'elapsed_seconds'] - elapsed2_target) > 5:
            continue

        chronotick2_ms = node2_all.loc[idx2_all, 'chronotick_offset_ms']
        unc2_ms = node2_all.loc[idx2_all, 'chronotick_uncertainty_ms']

        # Find Node 2's LAST NTP measurement (stale baseline)
        last_ntp2 = find_last_ntp_before(elapsed2_target, node2_ntp)
        if last_ntp2 is None:
            continue

        stale_ntp2_ms = last_ntp2['ntp_offset_ms']
        staleness_sec = elapsed2_target - last_ntp2['elapsed_seconds']

        # Find Node 2's CURRENT NTP if available (oracle)
        idx2_ntp = (node2_ntp['elapsed_seconds'] - elapsed2_target).abs().idxmin()
        has_current_ntp = abs(node2_ntp.loc[idx2_ntp, 'elapsed_seconds'] - elapsed2_target) < 5
        if has_current_ntp:
            current_ntp2_ms = node2_ntp.loc[idx2_ntp, 'ntp_offset_ms']
        else:
            current_ntp2_ms = None

        # Test agreement for each window size
        for window_ms in window_sizes:
            # Calculate positions within window
            pos1 = ntp1_ms % window_ms
            pos_chronotick = chronotick2_ms % window_ms
            pos_stale = stale_ntp2_ms % window_ms

            if pos1 < 0:
                pos1 += window_ms
            if pos_chronotick < 0:
                pos_chronotick += window_ms
            if pos_stale < 0:
                pos_stale += window_ms

            # Calculate differences
            def window_diff(p1, p2, window_size):
                diff = abs(p1 - p2)
                if diff > window_size / 2:
                    diff = window_size - diff
                return diff

            diff_chronotick = window_diff(pos1, pos_chronotick, window_ms)
            diff_stale = window_diff(pos1, pos_stale, window_ms)

            # Agreement threshold: 10ms for small windows, 1% of window for large
            threshold = min(10, window_ms * 0.01)

            agrees_chronotick = (diff_chronotick < threshold)
            agrees_stale = (diff_stale < threshold)

            # Oracle: current NTP if available
            agrees_oracle = None
            if current_ntp2_ms is not None:
                pos_oracle = current_ntp2_ms % window_ms
                if pos_oracle < 0:
                    pos_oracle += window_ms
                diff_oracle = window_diff(pos1, pos_oracle, window_ms)
                agrees_oracle = (diff_oracle < threshold)

            all_results[window_ms].append({
                'elapsed_hours': elapsed1 / 3600,
                'ntp1_ms': ntp1_ms,
                'chronotick2_ms': chronotick2_ms,
                'unc2_ms': unc2_ms,
                'stale_ntp2_ms': stale_ntp2_ms,
                'staleness_sec': staleness_sec,
                'current_ntp2_ms': current_ntp2_ms,
                'diff_chronotick': diff_chronotick,
                'diff_stale': diff_stale,
                'agrees_chronotick': agrees_chronotick,
                'agrees_stale': agrees_stale,
                'agrees_oracle': agrees_oracle,
            })

    # Calculate metrics for each window size
    results_summary = {}

    for window_ms in window_sizes:
        df = pd.DataFrame(all_results[window_ms])

        if len(df) == 0:
            print(f"\n⚠️  Window {window_ms}ms: No valid samples")
            continue

        agree_chronotick = (df['agrees_chronotick'].sum() / len(df) * 100)
        agree_stale = (df['agrees_stale'].sum() / len(df) * 100)

        # Oracle (only for samples with current NTP)
        df_oracle = df[df['agrees_oracle'].notna()]
        if len(df_oracle) > 0:
            agree_oracle = (df_oracle['agrees_oracle'].sum() / len(df_oracle) * 100)
        else:
            agree_oracle = None

        print(f"\nWindow {window_ms}ms ({len(df)} samples):")
        print(f"  Stale NTP (baseline): {df['agrees_stale'].sum()}/{len(df)} = {agree_stale:.1f}%")
        print(f"  ChronoTick: {df['agrees_chronotick'].sum()}/{len(df)} = {agree_chronotick:.1f}%")
        if agree_oracle is not None:
            print(f"  Oracle (current NTP): {df_oracle['agrees_oracle'].sum()}/{len(df_oracle)} = {agree_oracle:.1f}% ({len(df_oracle)} samples)")
        print(f"  💡 Improvement: {agree_chronotick - agree_stale:+.1f}%")

        results_summary[window_ms] = {
            'agree_chronotick': agree_chronotick,
            'agree_stale': agree_stale,
            'agree_oracle': agree_oracle,
            'improvement': agree_chronotick - agree_stale,
            'samples': len(df),
            'df': df
        }

    # Visualizations
    if output_dir and len(results_summary) > 0:
        # Figure 1: Comprehensive comparison for 1000ms window
        create_comprehensive_figure(results_summary, 1000, output_dir, exp_name)

        # Figure 2: Window size sensitivity
        create_sensitivity_figure(results_summary, output_dir, exp_name)

        # Figure 3: Focused timeline view
        create_timeline_figure(results_summary, 1000, output_dir, exp_name)

    return results_summary

def create_comprehensive_figure(results_summary, window_ms, output_dir, exp_name):
    """Create comprehensive comparison figure."""
    if window_ms not in results_summary:
        return

    df = results_summary[window_ms]['df']
    agree_stale = results_summary[window_ms]['agree_stale']
    agree_chronotick = results_summary[window_ms]['agree_chronotick']
    agree_oracle = results_summary[window_ms]['agree_oracle']

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)

    # Panel 1: Bar chart comparison
    ax1 = fig.add_subplot(gs[0, :])

    methods = ['Stale NTP\n(Baseline)', 'ChronoTick\n(Our Method)']
    rates = [agree_stale, agree_chronotick]
    colors = ['#CC79A7', '#009E73']

    if agree_oracle is not None:
        methods.insert(0, 'Oracle\n(Current NTP)')
        rates.insert(0, agree_oracle)
        colors.insert(0, '#0072B2')

    bars = ax1.bar(methods, rates, color=colors, alpha=0.85, edgecolor='black', linewidth=2.5)
    ax1.set_ylabel('Window Assignment Agreement (%)', fontweight='bold', fontsize=13)
    ax1.set_title(f'(a) Window Assignment Comparison: {window_ms}ms windows ({len(df)} samples)',
                 fontweight='bold', fontsize=14)
    ax1.axhline(90, color='red', linestyle='--', linewidth=2, alpha=0.7, label='90% target')
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', alpha=0.4)
    ax1.legend(loc='lower right', fontsize=11)

    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                f'{rate:.1f}%',
                ha='center', va='bottom', fontweight='bold', fontsize=14)

    # Improvement arrow
    if len(rates) >= 2:
        baseline_idx = 0 if agree_oracle is not None else 0
        chronotick_idx = 2 if agree_oracle is not None else 1
        improvement = rates[chronotick_idx] - rates[baseline_idx if agree_oracle is None else 1]

        ax1.annotate('', xy=(chronotick_idx, rates[chronotick_idx]),
                    xytext=(1 if agree_oracle is not None else 0, rates[1 if agree_oracle is not None else 0]),
                    arrowprops=dict(arrowstyle='->', lw=3, color='green'))
        ax1.text((chronotick_idx + (1 if agree_oracle is not None else 0)) / 2,
                (rates[chronotick_idx] + rates[1 if agree_oracle is not None else 0]) / 2,
                f'+{improvement:.1f}%',
                ha='center', fontsize=13, fontweight='bold', color='darkgreen',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Panel 2: Offset differences over time
    ax2 = fig.add_subplot(gs[1, :])

    # Plot stale NTP differences
    ax2.scatter(df['elapsed_hours'], df['diff_stale'],
               color='#CC79A7', s=40, alpha=0.6, label='Stale NTP', marker='s')

    # Plot ChronoTick differences
    ax2.scatter(df['elapsed_hours'], df['diff_chronotick'],
               color='#009E73', s=40, alpha=0.6, label='ChronoTick', marker='o')

    threshold = min(10, window_ms * 0.01)
    ax2.axhline(threshold, color='red', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Agreement threshold ({threshold:.1f}ms)')

    ax2.set_xlabel('Time (hours)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Offset Difference (ms)', fontweight='bold', fontsize=12)
    ax2.set_title('(b) Offset Differences Over Time', fontweight='bold', fontsize=13)
    ax2.grid(alpha=0.4)
    ax2.legend(loc='best', fontsize=11)
    ax2.set_yscale('log')

    # Panel 3: Staleness distribution
    ax3 = fig.add_subplot(gs[2, 0])

    ax3.hist(df['staleness_sec'], bins=30, color='#E69F00', alpha=0.7, edgecolor='black')
    ax3.set_xlabel('NTP Staleness (seconds)', fontweight='bold')
    ax3.set_ylabel('Count', fontweight='bold')
    ax3.set_title(f'(c) NTP Staleness\n(mean: {df["staleness_sec"].mean():.1f}s)', fontweight='bold')
    ax3.grid(axis='y', alpha=0.4)
    ax3.axvline(df['staleness_sec'].mean(), color='red', linestyle='--', linewidth=2, alpha=0.7)

    # Panel 4: Improvement by staleness
    ax4 = fig.add_subplot(gs[2, 1])

    # Bin by staleness
    df['staleness_bin'] = pd.cut(df['staleness_sec'], bins=5)
    grouped = df.groupby('staleness_bin').agg({
        'agrees_stale': 'mean',
        'agrees_chronotick': 'mean'
    }) * 100

    x = np.arange(len(grouped))
    width = 0.35

    ax4.bar(x - width/2, grouped['agrees_stale'], width, label='Stale NTP',
           color='#CC79A7', alpha=0.85, edgecolor='black')
    ax4.bar(x + width/2, grouped['agrees_chronotick'], width, label='ChronoTick',
           color='#009E73', alpha=0.85, edgecolor='black')

    ax4.set_xlabel('NTP Staleness Bin', fontweight='bold')
    ax4.set_ylabel('Agreement Rate (%)', fontweight='bold')
    ax4.set_title('(d) Performance vs Staleness', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{int(i.left)}-{int(i.right)}s' for i in grouped.index], rotation=45, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.4)

    # Panel 5: Histogram of differences
    ax5 = fig.add_subplot(gs[2, 2])

    bins = np.logspace(-2, 2, 30)
    ax5.hist(df['diff_stale'], bins=bins, alpha=0.6, label='Stale NTP', color='#CC79A7', edgecolor='black')
    ax5.hist(df['diff_chronotick'], bins=bins, alpha=0.6, label='ChronoTick', color='#009E73', edgecolor='black')
    ax5.axvline(threshold, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax5.set_xlabel('Offset Difference (ms)', fontweight='bold')
    ax5.set_ylabel('Count', fontweight='bold')
    ax5.set_title('(e) Difference Distribution', fontweight='bold')
    ax5.set_xscale('log')
    ax5.legend()
    ax5.grid(alpha=0.4)

    plt.suptitle(f'{exp_name}: ChronoTick vs Stale NTP Baseline',
                fontsize=16, fontweight='bold', y=0.995)

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / f'{exp_name}_ultimate_comprehensive.pdf', bbox_inches='tight')
    plt.savefig(output_dir / f'{exp_name}_ultimate_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✓ Saved comprehensive figure: {exp_name}_ultimate_comprehensive.pdf")

def create_sensitivity_figure(results_summary, output_dir, exp_name):
    """Create window size sensitivity figure."""
    window_sizes = sorted(results_summary.keys())

    stale_rates = [results_summary[ws]['agree_stale'] for ws in window_sizes]
    chronotick_rates = [results_summary[ws]['agree_chronotick'] for ws in window_sizes]
    improvements = [results_summary[ws]['improvement'] for ws in window_sizes]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Panel 1: Agreement rates
    ax1.plot(window_sizes, stale_rates, 'o-', linewidth=3, markersize=10,
            color='#CC79A7', label='Stale NTP (Baseline)')
    ax1.plot(window_sizes, chronotick_rates, 's-', linewidth=3, markersize=10,
            color='#009E73', label='ChronoTick')

    ax1.set_xlabel('Window Size (ms)', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Agreement Rate (%)', fontweight='bold', fontsize=12)
    ax1.set_title('(a) Window Size Sensitivity', fontweight='bold', fontsize=14)
    ax1.set_xscale('log')
    ax1.grid(alpha=0.4)
    ax1.legend(fontsize=12)
    ax1.axhline(90, color='red', linestyle='--', linewidth=2, alpha=0.7, label='90% target')

    # Panel 2: Improvement
    ax2.bar(range(len(window_sizes)), improvements, color='#009E73', alpha=0.85, edgecolor='black', linewidth=2)
    ax2.set_xlabel('Window Size (ms)', fontweight='bold', fontsize=12)
    ax2.set_ylabel('ChronoTick Improvement (%)', fontweight='bold', fontsize=12)
    ax2.set_title('(b) Improvement Over Baseline', fontweight='bold', fontsize=14)
    ax2.set_xticks(range(len(window_sizes)))
    ax2.set_xticklabels([f'{ws}' for ws in window_sizes])
    ax2.grid(axis='y', alpha=0.4)
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)

    for i, (ws, imp) in enumerate(zip(window_sizes, improvements)):
        ax2.text(i, imp + 0.5, f'+{imp:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.suptitle(f'{exp_name}: Window Size Sensitivity Analysis',
                fontsize=15, fontweight='bold')
    plt.tight_layout()

    plt.savefig(output_dir / f'{exp_name}_sensitivity.pdf', bbox_inches='tight')
    plt.savefig(output_dir / f'{exp_name}_sensitivity.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved sensitivity figure: {exp_name}_sensitivity.pdf")

def create_timeline_figure(results_summary, window_ms, output_dir, exp_name):
    """Create focused 5-10 minute timeline figure."""
    if window_ms not in results_summary:
        return

    df = results_summary[window_ms]['df']

    # Find interesting 10-minute window (hours 2-2.167)
    focus_start = 2.0
    focus_end = 2.167  # 10 minutes
    df_focus = df[(df['elapsed_hours'] >= focus_start) & (df['elapsed_hours'] <= focus_end)]

    if len(df_focus) < 3:
        # Use first 10 minutes if not enough data
        focus_start = df['elapsed_hours'].min()
        focus_end = focus_start + 0.167
        df_focus = df[(df['elapsed_hours'] >= focus_start) & (df['elapsed_hours'] <= focus_end)]

    if len(df_focus) == 0:
        return

    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot ground truth as reference line
    ax.axhline(0, color='#0072B2', linestyle='-', linewidth=3, alpha=0.8, label='Ground Truth (Node 1 NTP)', zorder=1)

    # Plot stale NTP offsets
    stale_errors = df_focus['stale_ntp2_ms'] - df_focus['ntp1_ms']
    ax.scatter(df_focus['elapsed_hours'], stale_errors,
              s=100, marker='s', color='#CC79A7', alpha=0.7, edgecolor='black', linewidth=1.5,
              label='Stale NTP', zorder=3)

    # Plot ChronoTick offsets with uncertainty
    ct_errors = df_focus['chronotick2_ms'] - df_focus['ntp1_ms']
    ax.errorbar(df_focus['elapsed_hours'], ct_errors, yerr=3*df_focus['unc2_ms'],
               fmt='o', markersize=10, color='#009E73', alpha=0.8, capsize=5, capthick=2,
               label='ChronoTick ±3σ', zorder=4)

    # Threshold lines
    threshold = min(10, window_ms * 0.01)
    ax.axhline(threshold, color='red', linestyle='--', linewidth=2, alpha=0.6, label=f'±{threshold:.1f}ms threshold')
    ax.axhline(-threshold, color='red', linestyle='--', linewidth=2, alpha=0.6)

    ax.set_xlabel('Time (hours)', fontweight='bold', fontsize=13)
    ax.set_ylabel('Offset Error vs Ground Truth (ms)', fontweight='bold', fontsize=13)
    ax.set_title(f'{exp_name}: Focused Timeline View (10 minutes, {len(df_focus)} events)',
                fontweight='bold', fontsize=14)
    ax.grid(alpha=0.4)
    ax.legend(fontsize=12, loc='best')
    ax.set_xlim(focus_start - 0.01, focus_end + 0.01)

    # Annotate key moments
    for idx, row in df_focus.iterrows():
        if abs(row['stale_ntp2_ms'] - row['ntp1_ms']) > threshold and \
           abs(row['chronotick2_ms'] - row['ntp1_ms']) < threshold:
            # ChronoTick saves the day!
            ax.annotate('ChronoTick\nprevents error',
                       xy=(row['elapsed_hours'], ct_errors[idx]),
                       xytext=(row['elapsed_hours'] + 0.02, ct_errors[idx] + 5),
                       arrowprops=dict(arrowstyle='->', color='green', lw=2),
                       fontsize=10, color='darkgreen', fontweight='bold',
                       bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
            break  # Just annotate one example

    plt.tight_layout()
    plt.savefig(output_dir / f'{exp_name}_timeline.pdf', bbox_inches='tight')
    plt.savefig(output_dir / f'{exp_name}_timeline.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved timeline figure: {exp_name}_timeline.pdf")

def run_ultimate_eval(exp_name, node1_csv, node2_csv, output_base):
    """Run ultimate evaluation."""
    print(f"\n\n{'#'*80}")
    print(f"# {exp_name.upper()}")
    print(f"{'#'*80}")

    if not node1_csv.exists() or not node2_csv.exists():
        print(f"⚠️  Data not found")
        return None

    data = load_data(node1_csv, node2_csv)
    output_dir = output_base / exp_name

    print(f"Loaded: {len(data['node1_ntp'])} NTP (Node 1), {len(data['node2_ntp'])} NTP (Node 2)")

    results = evaluate_window_assignment_ultimate(
        data,
        window_sizes=[100, 500, 1000, 5000],
        output_dir=output_dir,
        exp_name=exp_name
    )

    return results

def main():
    """Run ultimate evaluation."""
    output_base = Path("results/figures/ultimate_stream")

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
        results = run_ultimate_eval(exp_name, paths['node1'], paths['node2'], output_base)
        if results:
            all_results[exp_name] = results

    # Cross-experiment summary
    print(f"\n\n{'='*80}")
    print("CROSS-EXPERIMENT SUMMARY (1000ms windows)")
    print('='*80)

    for exp_name, results in all_results.items():
        if 1000 in results:
            r = results[1000]
            print(f"\n{exp_name}:")
            print(f"  Stale NTP: {r['agree_stale']:.1f}%")
            print(f"  ChronoTick: {r['agree_chronotick']:.1f}%")
            print(f"  Improvement: +{r['improvement']:.1f}% ({r['samples']} samples)")

    # Save summary
    summary = {}
    for exp_name, results in all_results.items():
        summary[exp_name] = {
            str(ws): {
                'agree_stale': r['agree_stale'],
                'agree_chronotick': r['agree_chronotick'],
                'improvement': r['improvement'],
                'samples': r['samples']
            }
            for ws, r in results.items()
        }

    summary_file = output_base / 'summary.json'
    output_base.mkdir(parents=True, exist_ok=True)
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Results saved to: {output_base}/")
    print(f"✓ Summary: {summary_file}")

if __name__ == "__main__":
    main()
