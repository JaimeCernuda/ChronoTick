#!/usr/bin/env python3
"""
Paper-Quality Figure for Experiment-5 Multi-Node Temporal Alignment (FIXED)

FIXES:
1. Remove outliers for better visualization
2. Mark comparisons as "Cannot Compare" when Node 2 hasn't started yet
3. Show start2 marker on panel (a)
4. Panel (b) x-axis starts at deployment gap (not zero)
5. Remove total bar graph - show only Test 1 and Test 2
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Paper-quality matplotlib settings
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
    'axes.linewidth': 0.8,
    'grid.alpha': 0.3,
    'grid.linewidth': 0.5,
})

def find_by_elapsed_time(target_elapsed, df, elapsed_column='elapsed_seconds', tolerance_seconds=5):
    """Find nearest sample by elapsed time."""
    time_diffs = np.abs(df[elapsed_column] - target_elapsed)
    min_diff = time_diffs.min()

    if min_diff <= tolerance_seconds:
        return df.loc[time_diffs.idxmin()]
    return None

def generate_paper_figure_experiment5_fixed(node1_csv, node2_csv, output_dir):
    """Generate FIXED paper-quality figure for Experiment-5."""

    print("="*80)
    print("GENERATING FIXED PAPER-QUALITY FIGURE: EXPERIMENT-5")
    print("="*80)
    print("\nFixes applied:")
    print("  1. Remove outliers for better visualization")
    print("  2. Exclude invalid comparisons when nodes not running")
    print("  3. Show Node 2 start marker")
    print("  4. Panel (b) x-axis starts at deployment gap")
    print("  5. Remove total bar - show only Test 1 and Test 2")

    # Load data
    df1 = pd.read_csv(node1_csv)
    df2 = pd.read_csv(node2_csv)

    df1_ntp = df1[df1['has_ntp'] == True].copy()
    df2_ntp = df2[df2['has_ntp'] == True].copy()
    df1_all = df1.copy()
    df2_all = df2.copy()

    print(f"\nNode 1: {len(df1_ntp)} NTP samples, {len(df1_all)} total samples")
    print(f"Node 2: {len(df2_ntp)} NTP samples, {len(df2_all)} total samples")

    # Parse timestamps
    df1_ntp['timestamp'] = pd.to_datetime(df1_ntp['datetime'])
    df2_ntp['timestamp'] = pd.to_datetime(df2_ntp['datetime'])
    df1_all['timestamp'] = pd.to_datetime(df1_all['datetime'])
    df2_all['timestamp'] = pd.to_datetime(df2_all['datetime'])

    # Calculate start offset
    start1 = df1_ntp['timestamp'].iloc[0]
    start2 = df2_ntp['timestamp'].iloc[0]
    start_offset = (start2 - start1).total_seconds()

    print(f"\nStart offset: {start_offset:.1f}s ({start_offset/60:.2f} minutes)")
    print(f"Node 2 starts {start_offset:.1f}s AFTER Node 1")

    # Compute elapsed_seconds if missing
    if 'elapsed_seconds' not in df1_all.columns:
        df1_all['elapsed_seconds'] = (df1_all['timestamp'] - df1_all['timestamp'].iloc[0]).dt.total_seconds()
        df1_ntp['elapsed_seconds'] = (df1_ntp['timestamp'] - df1_ntp['timestamp'].iloc[0]).dt.total_seconds()
    if 'elapsed_seconds' not in df2_all.columns:
        df2_all['elapsed_seconds'] = (df2_all['timestamp'] - df2_all['timestamp'].iloc[0]).dt.total_seconds()
        df2_ntp['elapsed_seconds'] = (df2_ntp['timestamp'] - df2_ntp['timestamp'].iloc[0]).dt.total_seconds()

    # Get runtime ranges
    node1_max_elapsed = df1_all['elapsed_seconds'].max()
    node2_max_elapsed = df2_all['elapsed_seconds'].max()

    print(f"\nNode 1 runtime: 0 - {node1_max_elapsed:.1f}s ({node1_max_elapsed/3600:.2f}h)")
    print(f"Node 2 runtime: 0 - {node2_max_elapsed:.1f}s ({node2_max_elapsed/3600:.2f}h)")

    # Calculate hours for visualization
    reference_time = min(start1, start2)
    df1_ntp['hours_from_start'] = (df1_ntp['timestamp'] - reference_time).dt.total_seconds() / 3600
    df2_ntp['hours_from_start'] = (df2_ntp['timestamp'] - reference_time).dt.total_seconds() / 3600

    # TEST 1: Node 1 NTP → Node 2 ChronoTick
    # ONLY compare when BOTH nodes are running!
    print(f"\n{'='*80}")
    print("TEST 1: Node 1 NTP → Node 2 ChronoTick")
    print('='*80)

    test1_results = []
    test1_invalid = 0

    for idx1, row1 in df1_ntp.iterrows():
        elapsed1 = row1['elapsed_seconds']
        elapsed2_target = elapsed1 - start_offset
        ntp_truth1 = row1['ntp_offset_ms']

        # CHECK: Is Node 2 running at this point?
        if elapsed2_target < 0:
            test1_invalid += 1
            continue  # Node 2 hasn't started yet!

        if elapsed2_target > node2_max_elapsed:
            test1_invalid += 1
            continue  # Node 2 already stopped!

        nearest2 = find_by_elapsed_time(elapsed2_target, df2_all, tolerance_seconds=5)

        if nearest2 is not None:
            chronotick2 = nearest2['chronotick_offset_ms']
            uncertainty2 = nearest2['chronotick_uncertainty_ms']

            lower_bound = chronotick2 - 3 * uncertainty2
            upper_bound = chronotick2 + 3 * uncertainty2

            agrees = (ntp_truth1 >= lower_bound) and (ntp_truth1 <= upper_bound)

            test1_results.append({
                'hours': row1['hours_from_start'],
                'ntp_truth': ntp_truth1,
                'chronotick_pred': chronotick2,
                'lower': lower_bound,
                'upper': upper_bound,
                'agrees': agrees
            })

    agreement1 = sum([r['agrees'] for r in test1_results]) / len(test1_results) * 100 if test1_results else 0
    print(f"Valid comparisons: {len(test1_results)}")
    print(f"Invalid (Node 2 not running): {test1_invalid}")
    print(f"Agreement: {agreement1:.1f}%")

    # TEST 2: Node 2 NTP → Node 1 ChronoTick
    print(f"\n{'='*80}")
    print("TEST 2: Node 2 NTP → Node 1 ChronoTick")
    print('='*80)

    test2_results = []
    test2_invalid = 0

    for idx2, row2 in df2_ntp.iterrows():
        elapsed2 = row2['elapsed_seconds']
        elapsed1_target = elapsed2 + start_offset
        ntp_truth2 = row2['ntp_offset_ms']

        # CHECK: Is Node 1 running at this point?
        if elapsed1_target < 0:
            test2_invalid += 1
            continue  # Node 1 hasn't started yet (shouldn't happen)

        if elapsed1_target > node1_max_elapsed:
            test2_invalid += 1
            continue  # Node 1 already stopped!

        nearest1 = find_by_elapsed_time(elapsed1_target, df1_all, tolerance_seconds=5)

        if nearest1 is not None:
            chronotick1 = nearest1['chronotick_offset_ms']
            uncertainty1 = nearest1['chronotick_uncertainty_ms']

            lower_bound = chronotick1 - 3 * uncertainty1
            upper_bound = chronotick1 + 3 * uncertainty1

            agrees = (ntp_truth2 >= lower_bound) and (ntp_truth2 <= upper_bound)

            test2_results.append({
                'hours': row2['hours_from_start'],
                'ntp_truth': ntp_truth2,
                'chronotick_pred': chronotick1,
                'lower': lower_bound,
                'upper': upper_bound,
                'agrees': agrees
            })

    agreement2 = sum([r['agrees'] for r in test2_results]) / len(test2_results) * 100 if test2_results else 0
    print(f"Valid comparisons: {len(test2_results)}")
    print(f"Invalid (Node 1 not running): {test2_invalid}")
    print(f"Agreement: {agreement2:.1f}%")

    # Overall
    total_comparisons = len(test1_results) + len(test2_results)
    total_agreements = sum([r['agrees'] for r in test1_results]) + sum([r['agrees'] for r in test2_results])
    overall_agreement = total_agreements / total_comparisons * 100 if total_comparisons > 0 else 0

    print(f"\nOverall: {overall_agreement:.1f}% ({total_comparisons} valid comparisons)")

    # Outlier detection: Use percentile-based filtering
    all_offsets = []
    all_offsets.extend([r['ntp_truth'] for r in test1_results])
    all_offsets.extend([r['chronotick_pred'] for r in test1_results])
    all_offsets.extend([r['ntp_truth'] for r in test2_results])
    all_offsets.extend([r['chronotick_pred'] for r in test2_results])

    p1, p99 = np.percentile(all_offsets, [1, 99])
    print(f"\nOffset range (1st-99th percentile): [{p1:.1f}, {p99:.1f}] ms")

    # Use slightly wider range for visualization
    y_min = p1 - 5
    y_max = p99 + 5

    # Create paper-quality figure
    fig, axes = plt.subplots(3, 1, figsize=(7, 8))

    # Color scheme
    color_node1 = '#0072B2'  # Blue
    color_node2 = '#D55E00'  # Orange
    color_agree = '#009E73'  # Green
    color_disagree = '#E69F00'  # Yellow/Orange

    # Panel 1: Test 1 - Node 1 NTP Truth vs Node 2 ChronoTick
    ax1 = axes[0]

    for result in test1_results:
        # Skip outliers for visualization
        if result['ntp_truth'] < y_min or result['ntp_truth'] > y_max:
            continue
        if result['chronotick_pred'] < y_min or result['chronotick_pred'] > y_max:
            continue

        edge_color = color_agree if result['agrees'] else color_disagree
        edge_width = 1.5 if result['agrees'] else 2

        # Uncertainty band
        ax1.plot([result['hours'], result['hours']],
                [max(result['lower'], y_min), min(result['upper'], y_max)],
                color=color_node2, linewidth=4, alpha=0.25, zorder=2)

        # ChronoTick prediction
        ax1.scatter(result['hours'], result['chronotick_pred'],
                   color=color_node2, marker='s', s=40, zorder=3,
                   edgecolors='black', linewidths=0.5)

        # NTP truth
        ax1.scatter(result['hours'], result['ntp_truth'],
                   color=color_node1, marker='o', s=40, zorder=4,
                   edgecolors=edge_color, linewidths=edge_width)

    # Mark when Node 2 starts
    node2_start_hours = start_offset / 3600
    ax1.axvline(node2_start_hours, color=color_node2, linewidth=1.5,
               linestyle=':', alpha=0.6, zorder=1)
    ax1.text(node2_start_hours + 0.1, y_max * 0.9,
            f'Node 2\nstarts\n({start_offset:.0f}s)',
            fontsize=8, color=color_node2, alpha=0.8)

    ax1.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.4, zorder=1)
    ax1.set_ylabel('Offset (ms)', fontsize=11)
    ax1.set_xlabel('Time (hours)', fontsize=11)
    ax1.text(0.02, 0.98, f'(a) Node 1 NTP → Node 2 ChronoTick: {agreement1:.1f}% agreement',
            transform=ax1.transAxes, fontsize=10, fontweight='bold',
            verticalalignment='top')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 8)
    ax1.set_ylim(y_min, y_max)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements1 = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_node1,
               markersize=7, label='Node 1 NTP Truth', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=color_node2,
               markersize=7, label='Node 2 ChronoTick', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], color=color_node2, linewidth=4, alpha=0.25, label='±3σ'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markersize=7, label='Agree', markeredgecolor=color_agree, markeredgewidth=2),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markersize=7, label='Disagree', markeredgecolor=color_disagree, markeredgewidth=2),
    ]
    ax1.legend(handles=legend_elements1, loc='upper right', fontsize=8, ncol=2, framealpha=0.9)

    # Panel 2: Test 2 - Node 2 NTP Truth vs Node 1 ChronoTick
    # X-axis starts at deployment gap!
    ax2 = axes[1]

    for result in test2_results:
        # Skip outliers
        if result['ntp_truth'] < y_min or result['ntp_truth'] > y_max:
            continue
        if result['chronotick_pred'] < y_min or result['chronotick_pred'] > y_max:
            continue

        edge_color = color_agree if result['agrees'] else color_disagree
        edge_width = 1.5 if result['agrees'] else 2

        # Uncertainty band
        ax2.plot([result['hours'], result['hours']],
                [max(result['lower'], y_min), min(result['upper'], y_max)],
                color=color_node1, linewidth=4, alpha=0.25, zorder=2)

        # ChronoTick prediction
        ax2.scatter(result['hours'], result['chronotick_pred'],
                   color=color_node1, marker='o', s=40, zorder=3,
                   edgecolors='black', linewidths=0.5)

        # NTP truth
        ax2.scatter(result['hours'], result['ntp_truth'],
                   color=color_node2, marker='s', s=40, zorder=4,
                   edgecolors=edge_color, linewidths=edge_width)

    ax2.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.4, zorder=1)
    ax2.set_ylabel('Offset (ms)', fontsize=11)
    ax2.set_xlabel('Time (hours)', fontsize=11)
    ax2.text(0.02, 0.98, f'(b) Node 2 NTP → Node 1 ChronoTick: {agreement2:.1f}% agreement',
            transform=ax2.transAxes, fontsize=10, fontweight='bold',
            verticalalignment='top')
    ax2.grid(True, alpha=0.3)

    # X-axis starts at deployment gap (when Node 2 actually starts)
    ax2.set_xlim(node2_start_hours, 8)
    ax2.set_ylim(y_min, y_max)

    # Legend
    legend_elements2 = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=color_node2,
               markersize=7, label='Node 2 NTP Truth', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_node1,
               markersize=7, label='Node 1 ChronoTick', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], color=color_node1, linewidth=4, alpha=0.25, label='±3σ'),
    ]
    ax2.legend(handles=legend_elements2, loc='upper right', fontsize=8, ncol=2, framealpha=0.9)

    # Panel 3: Statistics - ONLY Test 1 and Test 2 (no Overall)
    ax3 = axes[2]

    categories = ['Test 1\n(Node 1→2)', 'Test 2\n(Node 2→1)']
    agree_counts = [
        sum([r['agrees'] for r in test1_results]),
        sum([r['agrees'] for r in test2_results])
    ]
    total_counts = [
        len(test1_results),
        len(test2_results)
    ]

    x = np.arange(len(categories))
    width = 0.5

    # Stacked bars
    disagree_counts = [t - a for t, a in zip(total_counts, agree_counts)]

    bars1 = ax3.bar(x, agree_counts, width, label='Agree',
                   color=color_agree, alpha=0.8, edgecolor='black', linewidth=0.8)
    bars2 = ax3.bar(x, disagree_counts, width, bottom=agree_counts,
                   label='Disagree', color=color_disagree, alpha=0.8,
                   edgecolor='black', linewidth=0.8)

    ax3.set_ylabel('Number of Comparisons', fontsize=11)
    ax3.text(0.02, 0.98, f'(c) Overall: {overall_agreement:.1f}% agreement ({total_comparisons} valid comparisons)',
            transform=ax3.transAxes, fontsize=10, fontweight='bold',
            verticalalignment='top')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, fontsize=10)
    ax3.legend(fontsize=9, framealpha=0.9)
    ax3.grid(True, alpha=0.3, axis='y')

    # Add percentage labels
    for i, (agree, total) in enumerate(zip(agree_counts, total_counts)):
        pct = agree / total * 100 if total > 0 else 0
        ax3.text(i, total + 3, f'{pct:.1f}%', ha='center', fontsize=9, fontweight='bold')

    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "5.12_multinode_temporal_alignment_experiment5_FIXED.pdf"
    png_path = output_dir / "5.12_multinode_temporal_alignment_experiment5_FIXED.png"

    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"\n✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")

    plt.close()

    # Summary
    print(f"\n{'='*80}")
    print("FIXED FIGURE STATISTICS")
    print('='*80)
    print(f"Experiment: 5 (Best Result - FIXED)")
    print(f"Overall Agreement: {overall_agreement:.1f}% ({total_comparisons} valid comparisons)")
    print(f"Test 1: {agreement1:.1f}% ({len(test1_results)} valid, {test1_invalid} invalid)")
    print(f"Test 2: {agreement2:.1f}% ({len(test2_results)} valid, {test2_invalid} invalid)")
    print(f"Start Offset: {start_offset:.1f}s")
    print(f"Y-axis range: [{y_min:.1f}, {y_max:.1f}] ms (outliers removed)")

def main():
    """Generate FIXED paper-quality figure for Experiment-5."""

    print("="*80)
    print("PAPER-QUALITY FIGURE: EXPERIMENT-5 TEMPORAL ALIGNMENT (FIXED)")
    print("="*80)

    node1_csv = Path("results/experiment-5/ares-comp-11/data.csv")
    node2_csv = Path("results/experiment-5/ares-comp-12/data.csv")
    output_dir = Path("results/figures/5/experiment-5")

    if not node1_csv.exists() or not node2_csv.exists():
        print(f"\n⚠️  Dataset files not found!")
        return

    generate_paper_figure_experiment5_fixed(node1_csv, node2_csv, output_dir)

    print("\n" + "="*80)
    print("FIXED PAPER-QUALITY FIGURE COMPLETE")
    print("="*80)
    print("\nFixes applied:")
    print("  ✓ Outliers removed for better visualization")
    print("  ✓ Invalid comparisons excluded (when nodes not both running)")
    print("  ✓ Node 2 start marker shown")
    print("  ✓ Panel (b) x-axis starts at deployment gap")
    print("  ✓ Total bar removed - showing only Test 1 and Test 2")

if __name__ == "__main__":
    main()
