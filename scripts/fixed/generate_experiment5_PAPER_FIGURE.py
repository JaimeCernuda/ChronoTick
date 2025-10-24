#!/usr/bin/env python3
"""
Paper-Quality Figure for Experiment-5 Multi-Node Temporal Alignment

Best result: 78% agreement
Publication-ready visualization with exact paper color scheme.
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

def generate_paper_figure_experiment5(node1_csv, node2_csv, output_dir):
    """Generate paper-quality figure for Experiment-5."""

    print("="*80)
    print("GENERATING PAPER-QUALITY FIGURE: EXPERIMENT-5")
    print("="*80)
    print("\nBest result: 78% agreement")
    print("Publication-ready styling")

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

    # Compute elapsed_seconds if missing
    if 'elapsed_seconds' not in df1_all.columns:
        df1_all['elapsed_seconds'] = (df1_all['timestamp'] - df1_all['timestamp'].iloc[0]).dt.total_seconds()
        df1_ntp['elapsed_seconds'] = (df1_ntp['timestamp'] - df1_ntp['timestamp'].iloc[0]).dt.total_seconds()
    if 'elapsed_seconds' not in df2_all.columns:
        df2_all['elapsed_seconds'] = (df2_all['timestamp'] - df2_all['timestamp'].iloc[0]).dt.total_seconds()
        df2_ntp['elapsed_seconds'] = (df2_ntp['timestamp'] - df2_ntp['timestamp'].iloc[0]).dt.total_seconds()

    # Calculate hours for visualization
    reference_time = min(start1, start2)
    df1_ntp['hours_from_start'] = (df1_ntp['timestamp'] - reference_time).dt.total_seconds() / 3600
    df2_ntp['hours_from_start'] = (df2_ntp['timestamp'] - reference_time).dt.total_seconds() / 3600

    # TEST 1: Node 1 NTP → Node 2 ChronoTick
    print(f"\nTest 1: Node 1 NTP → Node 2 ChronoTick")
    test1_results = []

    for idx1, row1 in df1_ntp.iterrows():
        elapsed1 = row1['elapsed_seconds']
        elapsed2_target = elapsed1 - start_offset
        ntp_truth1 = row1['ntp_offset_ms']

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
    print(f"  Agreement: {agreement1:.1f}% ({len(test1_results)} comparisons)")

    # TEST 2: Node 2 NTP → Node 1 ChronoTick
    print(f"\nTest 2: Node 2 NTP → Node 1 ChronoTick")
    test2_results = []

    for idx2, row2 in df2_ntp.iterrows():
        elapsed2 = row2['elapsed_seconds']
        elapsed1_target = elapsed2 + start_offset
        ntp_truth2 = row2['ntp_offset_ms']

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
    print(f"  Agreement: {agreement2:.1f}% ({len(test2_results)} comparisons)")

    # Overall
    total_comparisons = len(test1_results) + len(test2_results)
    total_agreements = sum([r['agrees'] for r in test1_results]) + sum([r['agrees'] for r in test2_results])
    overall_agreement = total_agreements / total_comparisons * 100 if total_comparisons > 0 else 0

    print(f"\nOverall: {overall_agreement:.1f}% ({total_comparisons} comparisons)")

    # Create paper-quality figure
    fig, axes = plt.subplots(3, 1, figsize=(7, 8))

    # Color scheme (exact paper colors)
    color_node1 = '#0072B2'  # Blue
    color_node2 = '#D55E00'  # Orange
    color_agree = '#009E73'  # Green
    color_disagree = '#E69F00'  # Yellow/Orange

    # Panel 1: Test 1 - Node 1 NTP Truth vs Node 2 ChronoTick
    ax1 = axes[0]

    for result in test1_results:
        edge_color = color_agree if result['agrees'] else color_disagree
        edge_width = 1.5 if result['agrees'] else 2

        # Uncertainty band (thicker line)
        ax1.plot([result['hours'], result['hours']],
                [result['lower'], result['upper']],
                color=color_node2, linewidth=4, alpha=0.25, zorder=2)

        # ChronoTick prediction (orange square)
        ax1.scatter(result['hours'], result['chronotick_pred'],
                   color=color_node2, marker='s', s=40, zorder=3,
                   edgecolors='black', linewidths=0.5)

        # NTP truth (blue circle with colored edge)
        ax1.scatter(result['hours'], result['ntp_truth'],
                   color=color_node1, marker='o', s=40, zorder=4,
                   edgecolors=edge_color, linewidths=edge_width)

    ax1.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.4, zorder=1)
    ax1.set_ylabel('Offset (ms)', fontsize=11)
    ax1.text(0.02, 0.98, f'(a) Node 1 NTP → Node 2 ChronoTick: {agreement1:.1f}% agreement',
            transform=ax1.transAxes, fontsize=10, fontweight='bold',
            verticalalignment='top')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 8)

    # Custom legend for panel 1
    from matplotlib.lines import Line2D
    legend_elements1 = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_node1,
               markersize=7, label='Node 1 NTP Truth', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=color_node2,
               markersize=7, label='Node 2 ChronoTick', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], color=color_node2, linewidth=4, alpha=0.25, label='±3σ Uncertainty'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markersize=7, label='Agreement', markeredgecolor=color_agree, markeredgewidth=2),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
               markersize=7, label='Disagreement', markeredgecolor=color_disagree, markeredgewidth=2),
    ]
    ax1.legend(handles=legend_elements1, loc='upper right', fontsize=8, ncol=2, framealpha=0.9)

    # Panel 2: Test 2 - Node 2 NTP Truth vs Node 1 ChronoTick
    ax2 = axes[1]

    for result in test2_results:
        edge_color = color_agree if result['agrees'] else color_disagree
        edge_width = 1.5 if result['agrees'] else 2

        # Uncertainty band
        ax2.plot([result['hours'], result['hours']],
                [result['lower'], result['upper']],
                color=color_node1, linewidth=4, alpha=0.25, zorder=2)

        # ChronoTick prediction (blue circle)
        ax2.scatter(result['hours'], result['chronotick_pred'],
                   color=color_node1, marker='o', s=40, zorder=3,
                   edgecolors='black', linewidths=0.5)

        # NTP truth (orange square with colored edge)
        ax2.scatter(result['hours'], result['ntp_truth'],
                   color=color_node2, marker='s', s=40, zorder=4,
                   edgecolors=edge_color, linewidths=edge_width)

    ax2.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.4, zorder=1)
    ax2.set_ylabel('Offset (ms)', fontsize=11)
    ax2.text(0.02, 0.98, f'(b) Node 2 NTP → Node 1 ChronoTick: {agreement2:.1f}% agreement',
            transform=ax2.transAxes, fontsize=10, fontweight='bold',
            verticalalignment='top')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 8)

    # Custom legend for panel 2
    legend_elements2 = [
        Line2D([0], [0], marker='s', color='w', markerfacecolor=color_node2,
               markersize=7, label='Node 2 NTP Truth', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_node1,
               markersize=7, label='Node 1 ChronoTick', markeredgecolor='black', markeredgewidth=0.5),
        Line2D([0], [0], color=color_node1, linewidth=4, alpha=0.25, label='±3σ Uncertainty'),
    ]
    ax2.legend(handles=legend_elements2, loc='upper right', fontsize=8, ncol=2, framealpha=0.9)

    # Panel 3: Overall statistics
    ax3 = axes[2]

    categories = ['Test 1\n(Node 1→2)', 'Test 2\n(Node 2→1)', 'Overall']
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

    # Stacked bars
    disagree_counts = [t - a for t, a in zip(total_counts, agree_counts)]

    bars1 = ax3.bar(x, agree_counts, width, label='Agree',
                   color=color_agree, alpha=0.8, edgecolor='black', linewidth=0.8)
    bars2 = ax3.bar(x, disagree_counts, width, bottom=agree_counts,
                   label='Disagree', color=color_disagree, alpha=0.8,
                   edgecolor='black', linewidth=0.8)

    ax3.set_ylabel('Number of Comparisons', fontsize=11)
    ax3.set_xlabel('Time (hours)', fontsize=11)
    ax3.text(0.02, 0.98, f'(c) Timeline Alignment: {overall_agreement:.1f}% agreement ({total_comparisons} comparisons)',
            transform=ax3.transAxes, fontsize=10, fontweight='bold',
            verticalalignment='top')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend(fontsize=9, framealpha=0.9)
    ax3.grid(True, alpha=0.3, axis='y')

    # Add percentage labels on bars
    for i, (agree, total) in enumerate(zip(agree_counts, total_counts)):
        pct = agree / total * 100 if total > 0 else 0
        ax3.text(i, total + 5, f'{pct:.1f}%', ha='center', fontsize=9, fontweight='bold')

    # Set common x-label for top two panels
    ax1.set_xlabel('Time (hours)', fontsize=11)
    ax2.set_xlabel('Time (hours)', fontsize=11)

    plt.tight_layout()

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = output_dir / "5.12_multinode_temporal_alignment_experiment5.pdf"
    png_path = output_dir / "5.12_multinode_temporal_alignment_experiment5.png"

    plt.savefig(pdf_path, bbox_inches='tight')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')

    print(f"\n✓ Saved: {pdf_path}")
    print(f"✓ Saved: {png_path}")

    plt.close()

    # Summary
    print(f"\n{'='*80}")
    print("FIGURE STATISTICS")
    print('='*80)
    print(f"Experiment: 5 (Best Result)")
    print(f"Overall Agreement: {overall_agreement:.1f}%")
    print(f"Test 1: {agreement1:.1f}% ({len(test1_results)} comparisons)")
    print(f"Test 2: {agreement2:.1f}% ({len(test2_results)} comparisons)")
    print(f"Total: {total_comparisons} comparisons")
    print(f"Start Offset: {start_offset:.1f}s")

def main():
    """Generate paper-quality figure for Experiment-5."""

    print("="*80)
    print("PAPER-QUALITY FIGURE: EXPERIMENT-5 TEMPORAL ALIGNMENT")
    print("="*80)

    node1_csv = Path("results/experiment-5/ares-comp-11/data.csv")
    node2_csv = Path("results/experiment-5/ares-comp-12/data.csv")
    output_dir = Path("results/figures/5/experiment-5")

    if not node1_csv.exists() or not node2_csv.exists():
        print(f"\n⚠️  Dataset files not found!")
        return

    generate_paper_figure_experiment5(node1_csv, node2_csv, output_dir)

    print("\n" + "="*80)
    print("PAPER-QUALITY FIGURE COMPLETE")
    print("="*80)
    print("\nReady for publication!")
    print("Figure saved in: results/figures/5/experiment-5/")

if __name__ == "__main__":
    main()
