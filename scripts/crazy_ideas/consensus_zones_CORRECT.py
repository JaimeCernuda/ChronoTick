#!/usr/bin/env python3
"""
CONSENSUS ZONES - The Google Spanner TrueTime Story (CORRECT VERSION)

Cross-node comparison at same elapsed time:

Single-point clocks (problem):
- Node 1: "Time is T1"
- Node 2: "Time is T2"
- T1 ≠ T2 → DISAGREE! No way to reconcile!

Bounded clocks (solution):
- Node 1: "Time is in [T1-3σ, T1+3σ]"
- Node 2: "Time is in [T2-3σ, T2+3σ]"
- Ranges overlap? → CONSENSUS ZONE! Can agree event is "concurrent within uncertainty"

This enables stream processing semantics:
- Events in consensus zone → treat as concurrent
- Events outside consensus zone → definite ordering
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

def evaluate_consensus_zones(data, output_dir=None, exp_name=''):
    """
    Evaluate consensus zones.

    At same wall-clock moment (same elapsed time):
    1. Single-point: Do NTP estimates agree?
    2. Bounded: Do ChronoTick ±3σ ranges overlap?
    """
    print(f"\n{'='*80}")
    print(f"CONSENSUS ZONES EVALUATION: {exp_name}")
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

        # Node 1 estimates (at this elapsed time)
        ntp1_offset_ms = row1['ntp_offset_ms']

        # Find Node 1's ChronoTick at same moment
        idx1_all = (node1_all['elapsed_seconds'] - elapsed1).abs().idxmin()
        if abs(node1_all.loc[idx1_all, 'elapsed_seconds'] - elapsed1) > 5:
            continue

        chronotick1_offset_ms = node1_all.loc[idx1_all, 'chronotick_offset_ms']
        unc1_ms = node1_all.loc[idx1_all, 'chronotick_uncertainty_ms']

        # Node 2's ChronoTick at same wall-clock moment
        idx2_all = (node2_all['elapsed_seconds'] - elapsed2_target).abs().idxmin()
        if abs(node2_all.loc[idx2_all, 'elapsed_seconds'] - elapsed2_target) > 5:
            continue

        chronotick2_offset_ms = node2_all.loc[idx2_all, 'chronotick_offset_ms']
        unc2_ms = node2_all.loc[idx2_all, 'chronotick_uncertainty_ms']

        # Node 2's NTP at same moment (if available)
        idx2_ntp = (node2_ntp['elapsed_seconds'] - elapsed2_target).abs().idxmin()
        has_ntp2 = abs(node2_ntp.loc[idx2_ntp, 'elapsed_seconds'] - elapsed2_target) < 5

        if has_ntp2:
            ntp2_offset_ms = node2_ntp.loc[idx2_ntp, 'ntp_offset_ms']
        else:
            ntp2_offset_ms = None

        # SINGLE-POINT COMPARISON
        # Question: Do NTP point estimates agree?
        if ntp2_offset_ms is not None:
            ntp_diff_ms = abs(ntp1_offset_ms - ntp2_offset_ms)
            ntp_agree_10ms = (ntp_diff_ms < 10)
        else:
            ntp_diff_ms = None
            ntp_agree_10ms = None

        # BOUNDED CLOCK COMPARISON
        # Question: Do ChronoTick ±3σ ranges overlap?

        # Node 1 interval
        chronotick1_lower = chronotick1_offset_ms - 3 * unc1_ms
        chronotick1_upper = chronotick1_offset_ms + 3 * unc1_ms

        # Node 2 interval
        chronotick2_lower = chronotick2_offset_ms - 3 * unc2_ms
        chronotick2_upper = chronotick2_offset_ms + 3 * unc2_ms

        # Check for overlap
        # Ranges [A1, A2] and [B1, B2] overlap if:
        # A2 >= B1 AND B2 >= A1
        ranges_overlap = (chronotick1_upper >= chronotick2_lower) and \
                        (chronotick2_upper >= chronotick1_lower)

        # Calculate overlap amount
        if ranges_overlap:
            overlap_lower = max(chronotick1_lower, chronotick2_lower)
            overlap_upper = min(chronotick1_upper, chronotick2_upper)
            overlap_size_ms = overlap_upper - overlap_lower
        else:
            overlap_size_ms = 0

        # Total uncertainty span
        total_span_ms = (chronotick1_upper - chronotick1_lower) + \
                       (chronotick2_upper - chronotick2_lower)

        # What's the point-estimate difference?
        chronotick_diff_ms = abs(chronotick1_offset_ms - chronotick2_offset_ms)

        results.append({
            'elapsed_hours': elapsed1 / 3600,
            'ntp1_offset_ms': ntp1_offset_ms,
            'ntp2_offset_ms': ntp2_offset_ms,
            'chronotick1_offset_ms': chronotick1_offset_ms,
            'chronotick2_offset_ms': chronotick2_offset_ms,
            'unc1_ms': unc1_ms,
            'unc2_ms': unc2_ms,
            'chronotick1_lower': chronotick1_lower,
            'chronotick1_upper': chronotick1_upper,
            'chronotick2_lower': chronotick2_lower,
            'chronotick2_upper': chronotick2_upper,
            'ntp_diff_ms': ntp_diff_ms,
            'chronotick_diff_ms': chronotick_diff_ms,
            'ntp_agree_10ms': ntp_agree_10ms,
            'ranges_overlap': ranges_overlap,
            'overlap_size_ms': overlap_size_ms,
            'total_span_ms': total_span_ms,
        })

    df = pd.DataFrame(results)

    if len(df) == 0:
        print("⚠️  No valid cross-node comparisons!")
        return None

    # Calculate metrics
    print(f"\n📊 CROSS-NODE COMPARISON ({len(df)} wall-clock moments)")
    print('='*80)

    # Single-point (NTP)
    df_with_ntp = df[df['ntp_agree_10ms'].notna()]
    if len(df_with_ntp) > 0:
        ntp_agree_rate = (df_with_ntp['ntp_agree_10ms'].sum() / len(df_with_ntp) * 100)
        print(f"\n🔵 SINGLE-POINT (NTP offsets):")
        print(f"  Samples with both NTP: {len(df_with_ntp)}")
        print(f"  Agreement (<10ms):     {df_with_ntp['ntp_agree_10ms'].sum()}/{len(df_with_ntp)} = {ntp_agree_rate:.1f}%")
        print(f"  Median difference:     {df_with_ntp['ntp_diff_ms'].median():.2f}ms")
        print(f"  Mean difference:       {df_with_ntp['ntp_diff_ms'].mean():.2f}ms")
    else:
        ntp_agree_rate = None

    # Bounded clock (ChronoTick)
    overlap_rate = (df['ranges_overlap'].sum() / len(df) * 100)
    print(f"\n🟢 BOUNDED CLOCK (ChronoTick ±3σ ranges):")
    print(f"  Ranges overlap:        {df['ranges_overlap'].sum()}/{len(df)} = {overlap_rate:.1f}%")
    print(f"  Median overlap size:   {df[df['ranges_overlap']]['overlap_size_ms'].median():.2f}ms")
    print(f"  Point-estimate diff:   {df['chronotick_diff_ms'].median():.2f}ms (median)")
    print(f"  Combined uncertainty:  {df['total_span_ms'].median():.2f}ms (median total range)")

    print(f"\n🎯 THE VALUE PROPOSITION:")
    print(f"  Single-point NTP: {ntp_agree_rate:.1f}% agreement" if ntp_agree_rate else "  Single-point NTP: N/A (need simultaneous measurements)")
    print(f"  Bounded ChronoTick: {overlap_rate:.1f}% consensus zones")
    print(f"  → Bounded clock enables agreement even when points differ!")

    # Visualizations
    if output_dir:
        create_consensus_visualization(df, df_with_ntp if len(df_with_ntp) > 0 else None,
                                      output_dir, exp_name)

    return {
        'ntp_agree_rate': ntp_agree_rate,
        'overlap_rate': overlap_rate,
        'samples': len(df),
        'samples_with_ntp': len(df_with_ntp) if len(df_with_ntp) > 0 else 0,
        'df': df
    }

def create_consensus_visualization(df, df_with_ntp, output_dir, exp_name):
    """Create visualization showing consensus zones."""

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    # Panel 1: Timeline showing bounded intervals
    ax1 = fig.add_subplot(gs[0, :])

    # Take a focused window (10 minutes worth of samples)
    focus_start = df['elapsed_hours'].min() + 2.0
    focus_end = focus_start + (10/60)  # 10 minutes
    df_focus = df[(df['elapsed_hours'] >= focus_start) & (df['elapsed_hours'] <= focus_end)]

    if len(df_focus) > 10:
        df_focus = df_focus.iloc[:10]  # Limit to 10 samples for clarity

    for idx, row in df_focus.iterrows():
        t = row['elapsed_hours']

        # Node 1 interval (green)
        ax1.plot([t, t], [row['chronotick1_lower'], row['chronotick1_upper']],
                color='green', linewidth=3, alpha=0.7, label='Node 1 ±3σ' if idx == df_focus.index[0] else '')
        ax1.plot([t], [row['chronotick1_offset_ms']], 'o', color='darkgreen', markersize=8)

        # Node 2 interval (blue)
        ax1.plot([t, t], [row['chronotick2_lower'], row['chronotick2_upper']],
                color='blue', linewidth=3, alpha=0.7, label='Node 2 ±3σ' if idx == df_focus.index[0] else '')
        ax1.plot([t], [row['chronotick2_offset_ms']], 's', color='darkblue', markersize=8)

        # Highlight overlap
        if row['ranges_overlap']:
            overlap_lower = max(row['chronotick1_lower'], row['chronotick2_lower'])
            overlap_upper = min(row['chronotick1_upper'], row['chronotick2_upper'])
            ax1.fill_between([t-0.01, t+0.01], [overlap_lower]*2, [overlap_upper]*2,
                           color='gold', alpha=0.5, label='Consensus zone' if idx == df_focus.index[0] else '')

    ax1.set_xlabel('Time (hours)', fontsize=12)
    ax1.set_ylabel('Clock Offset (ms)', fontsize=12)
    ax1.set_title('(a) Bounded Clocks Enable Consensus Zones (10-min focused view)',
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3)

    # Panel 2: Agreement rates
    ax2 = fig.add_subplot(gs[1, 0])

    overlap_rate = (df['ranges_overlap'].sum() / len(df) * 100)

    if df_with_ntp is not None and len(df_with_ntp) > 0:
        ntp_rate = (df_with_ntp['ntp_agree_10ms'].sum() / len(df_with_ntp) * 100)
        categories = ['Single-Point\n(NTP)', 'Bounded\n(ChronoTick ±3σ)']
        values = [ntp_rate, overlap_rate]
        colors = ['steelblue', 'green']
    else:
        categories = ['Bounded\n(ChronoTick ±3σ)']
        values = [overlap_rate]
        colors = ['green']

    bars = ax2.bar(categories, values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)

    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 2,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax2.axhline(90, color='red', linestyle='--', linewidth=2, alpha=0.7, label='90% target')
    ax2.set_ylabel('Agreement / Overlap Rate (%)', fontsize=12)
    ax2.set_title('(b) Consensus Rate: Bounded vs Single-Point', fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 105)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Panel 3: Overlap size distribution
    ax3 = fig.add_subplot(gs[1, 1])

    df_overlap = df[df['ranges_overlap'] == True]
    if len(df_overlap) > 0:
        ax3.hist(df_overlap['overlap_size_ms'], bins=30, color='gold', alpha=0.7, edgecolor='black')
        ax3.axvline(df_overlap['overlap_size_ms'].median(), color='red', linestyle='--',
                   linewidth=2, label=f'Median: {df_overlap["overlap_size_ms"].median():.2f}ms')
        ax3.set_xlabel('Consensus Zone Size (ms)', fontsize=12)
        ax3.set_ylabel('Count', fontsize=12)
        ax3.set_title('(c) Size of Consensus Zones', fontsize=12, fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)

    # Panel 4: Timeline of all overlaps
    ax4 = fig.add_subplot(gs[2, :])

    # Scatter: overlapping vs non-overlapping
    df_yes = df[df['ranges_overlap'] == True]
    df_no = df[df['ranges_overlap'] == False]

    ax4.scatter(df_yes['elapsed_hours'], df_yes['chronotick_diff_ms'],
               color='green', alpha=0.6, s=50, label=f'Overlap ({len(df_yes)})', zorder=3)
    ax4.scatter(df_no['elapsed_hours'], df_no['chronotick_diff_ms'],
               color='red', alpha=0.6, s=50, marker='x', label=f'No overlap ({len(df_no)})', zorder=3)

    # Show uncertainty threshold
    median_total_unc = df['total_span_ms'].median()
    ax4.axhline(median_total_unc / 2, color='orange', linestyle='--', linewidth=2, alpha=0.7,
               label=f'Median combined uncertainty: {median_total_unc/2:.1f}ms')

    ax4.set_xlabel('Time (hours)', fontsize=12)
    ax4.set_ylabel('Point-Estimate Difference (ms)', fontsize=12)
    ax4.set_title('(d) Cross-Node Offset Differences Over Time', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper right')
    ax4.set_yscale('log')
    ax4.grid(alpha=0.3)

    plt.suptitle(f'{exp_name}: Consensus Zones (Google Spanner TrueTime Narrative)',
                 fontsize=14, fontweight='bold', y=0.995)

    output_path = Path(output_dir) / exp_name
    output_path.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path / f'{exp_name}_consensus_zones.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_path / f'{exp_name}_consensus_zones.pdf', bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}/{exp_name}_consensus_zones.pdf")
    plt.close()

def run_evaluation(exp_name, node1_csv, node2_csv, output_dir):
    """Run consensus zones evaluation."""
    print(f"\n{'#'*80}")
    print(f"# {exp_name.upper()}")
    print(f"{'#'*80}")

    data = load_data(node1_csv, node2_csv)
    print(f"Loaded: {len(data['node1_ntp'])} NTP (Node 1), {len(data['node2_ntp'])} NTP (Node 2)")

    results = evaluate_consensus_zones(data, output_dir=output_dir, exp_name=exp_name)

    return results

def main():
    """Run consensus zones evaluation."""
    output_base = Path("results/figures/consensus_zones")

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
    print("CONSENSUS ZONES SUMMARY")
    print('='*80)
    print("\nThe Google Spanner TrueTime narrative:")
    print("- Single-point clocks: Nodes disagree (T1 ≠ T2)")
    print("- Bounded clocks: Uncertainty ranges overlap → consensus zones!")
    print('='*80)

    for exp_name, results in all_results.items():
        print(f"\n{exp_name}:")
        if results['ntp_agree_rate']:
            print(f"  Single-point (NTP):       {results['ntp_agree_rate']:.1f}% agreement ({results['samples_with_ntp']} samples)")
        else:
            print(f"  Single-point (NTP):       N/A (insufficient simultaneous measurements)")
        print(f"  Bounded (ChronoTick):     {results['overlap_rate']:.1f}% overlap ({results['samples']} samples)")

    # Save summary
    summary = {}
    for exp_name, results in all_results.items():
        summary[exp_name] = {
            'ntp_agree_rate': results['ntp_agree_rate'],
            'overlap_rate': results['overlap_rate'],
            'samples': results['samples'],
            'samples_with_ntp': results['samples_with_ntp'],
        }

    with open(output_base / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved to: {output_base}/summary.json")

if __name__ == "__main__":
    main()
