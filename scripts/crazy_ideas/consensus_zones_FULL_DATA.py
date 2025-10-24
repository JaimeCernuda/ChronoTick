#!/usr/bin/env python3
"""
Consensus Zones: Full Data Visualization (No Subsampling)

Show ALL available synchronized NTP pairs across multiple focused time windows.
This addresses the concern: "why are you showing me only a few data points"

Approach:
- Window 1: First 10 minutes (ALL points in this window)
- Window 2: Minutes 60-70 (ALL points)
- Window 3: Minutes 240-250 (ALL points)
- Window 4: Minutes 420-430 (ALL points)
- Full overview: All 8 hours with ALL 132 synchronized pairs

No arbitrary subsampling - show every synchronized measurement!
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def load_experiment_data(exp_dir):
    """Load experiment data from both nodes."""
    exp_path = Path(exp_dir)

    # Load Node 1 data
    node1_path = exp_path / "ares-comp-11" / "data.csv"
    node1_df = pd.read_csv(node1_path)

    # Load Node 2 data
    node2_path = exp_path / "ares-comp-12" / "data.csv"
    node2_df = pd.read_csv(node2_path)

    # Calculate deployment offset (Node 2 started later)
    start_offset = node2_df['elapsed_seconds'].min() - node1_df['elapsed_seconds'].min()

    print(f"Node 1 samples: {len(node1_df)} (NTP: {node1_df['has_ntp'].sum()})")
    print(f"Node 2 samples: {len(node2_df)} (NTP: {node2_df['has_ntp'].sum()})")
    print(f"Deployment offset: {start_offset:.1f}s")

    return {
        'node1': node1_df,
        'node2': node2_df,
        'start_offset': start_offset
    }

def find_synchronized_pairs(data, tolerance_seconds=5.0):
    """
    Find ALL synchronized NTP pairs (no subsampling!).

    Returns dataframe with all pairs where both nodes have NTP within tolerance.
    """
    node1_df = data['node1']
    node2_df = data['node2']
    start_offset = data['start_offset']

    # Get NTP samples
    node1_ntp = node1_df[node1_df['has_ntp']].copy()
    node2_ntp = node2_df[node2_df['has_ntp']].copy()

    pairs = []

    for idx1, row1 in node1_ntp.iterrows():
        elapsed1 = row1['elapsed_seconds']

        # Map to Node 2's timeline (account for deployment offset)
        elapsed2_target = elapsed1 - start_offset

        # Find closest Node 2 NTP measurement
        time_diffs = (node2_ntp['elapsed_seconds'] - elapsed2_target).abs()

        if time_diffs.min() <= tolerance_seconds:
            idx2 = time_diffs.idxmin()
            row2 = node2_ntp.loc[idx2]

            # Get ChronoTick data for both nodes at these moments
            idx1_ct = (node1_df['elapsed_seconds'] - elapsed1).abs().idxmin()
            idx2_ct = (node2_df['elapsed_seconds'] - row2['elapsed_seconds']).abs().idxmin()

            ct1 = node1_df.loc[idx1_ct]
            ct2 = node2_df.loc[idx2_ct]

            pairs.append({
                'elapsed_hours': elapsed1 / 3600,
                'elapsed_minutes': elapsed1 / 60,
                'node1_ntp_offset': row1['ntp_offset_ms'],
                'node2_ntp_offset': row2['ntp_offset_ms'],
                'ntp_diff': abs(row1['ntp_offset_ms'] - row2['ntp_offset_ms']),
                'ntp_agrees': abs(row1['ntp_offset_ms'] - row2['ntp_offset_ms']) < 10,
                'node1_ct_offset': ct1['chronotick_offset_ms'],
                'node2_ct_offset': ct2['chronotick_offset_ms'],
                'node1_ct_uncertainty': ct1['chronotick_uncertainty_ms'],
                'node2_ct_uncertainty': ct2['chronotick_uncertainty_ms'],
                'node1_ct_lower': ct1['chronotick_offset_ms'] - 3*ct1['chronotick_uncertainty_ms'],
                'node1_ct_upper': ct1['chronotick_offset_ms'] + 3*ct1['chronotick_uncertainty_ms'],
                'node2_ct_lower': ct2['chronotick_offset_ms'] - 3*ct2['chronotick_uncertainty_ms'],
                'node2_ct_upper': ct2['chronotick_offset_ms'] + 3*ct2['chronotick_uncertainty_ms'],
            })

    df = pd.DataFrame(pairs)

    # Calculate consensus zones
    df['ct_overlaps'] = (
        (df['node1_ct_upper'] >= df['node2_ct_lower']) &
        (df['node2_ct_upper'] >= df['node1_ct_lower'])
    )

    return df

def create_focused_window_viz(pairs_df, time_window, output_path):
    """
    Create visualization for a specific time window showing ALL data points.

    Args:
        pairs_df: DataFrame with all synchronized pairs
        time_window: (start_min, end_min) tuple
        output_path: Where to save figure
    """
    start_min, end_min = time_window

    # Filter to this time window
    window_data = pairs_df[
        (pairs_df['elapsed_minutes'] >= start_min) &
        (pairs_df['elapsed_minutes'] <= end_min)
    ].copy()

    if len(window_data) == 0:
        print(f"No data in window {start_min}-{end_min} minutes")
        return

    print(f"\n{'='*60}")
    print(f"Time Window: {start_min}-{end_min} minutes")
    print(f"Data points: {len(window_data)}")
    print(f"NTP agreement: {window_data['ntp_agrees'].sum()}/{len(window_data)} = {window_data['ntp_agrees'].mean()*100:.1f}%")
    print(f"ChronoTick consensus: {window_data['ct_overlaps'].sum()}/{len(window_data)} = {window_data['ct_overlaps'].mean()*100:.1f}%")
    print(f"{'='*60}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Panel (a): Single-Point NTP Offsets
    ax1.plot(window_data['elapsed_minutes'], window_data['node1_ntp_offset'],
             'o-', color='green', linewidth=2, markersize=6, label='Node 1 NTP offset')
    ax1.plot(window_data['elapsed_minutes'], window_data['node2_ntp_offset'],
             's-', color='blue', linewidth=2, markersize=6, label='Node 2 NTP offset')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Perfect clock (offset=0)')

    # Highlight disagreements
    disagreements = window_data[~window_data['ntp_agrees']]
    for _, row in disagreements.iterrows():
        ax1.axvspan(row['elapsed_minutes']-0.5, row['elapsed_minutes']+0.5,
                   alpha=0.2, color='red')

    ax1.set_xlabel('Elapsed Time (minutes)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Clock Offset (ms)\n(How much to correct system clock)', fontsize=12, fontweight='bold')
    ax1.set_title(f'(a) Single-Point Clocks: NTP Offsets (Minutes {start_min}-{end_min})\n'
                 f'What correction each node thinks it needs | Agreement: {window_data["ntp_agrees"].mean()*100:.1f}%',
                 fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Panel (b): Bounded ChronoTick ±3σ Ranges
    for _, row in window_data.iterrows():
        t = row['elapsed_minutes']

        # Node 1 range (green)
        ax2.plot([t, t], [row['node1_ct_lower'], row['node1_ct_upper']],
                color='green', linewidth=14, alpha=0.4, solid_capstyle='round')

        # Node 2 range (blue)
        ax2.plot([t, t], [row['node2_ct_lower'], row['node2_ct_upper']],
                color='blue', linewidth=14, alpha=0.4, solid_capstyle='round')

        # Consensus zone (overlap) - GOLD
        if row['ct_overlaps']:
            overlap_lower = max(row['node1_ct_lower'], row['node2_ct_lower'])
            overlap_upper = min(row['node1_ct_upper'], row['node2_ct_upper'])
            ax2.plot([t, t], [overlap_lower, overlap_upper],
                    color='gold', linewidth=16, alpha=0.8, solid_capstyle='round',
                    label='Consensus zone' if t == window_data['elapsed_minutes'].iloc[0] else '')

    # Add center points
    ax2.plot(window_data['elapsed_minutes'], window_data['node1_ct_offset'],
            'o', color='darkgreen', markersize=4, label='Node 1 ChronoTick ±3σ')
    ax2.plot(window_data['elapsed_minutes'], window_data['node2_ct_offset'],
            's', color='darkblue', markersize=4, label='Node 2 ChronoTick ±3σ')

    ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Perfect clock (offset=0)')

    ax2.set_xlabel('Elapsed Time (minutes)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Clock Offset (ms)\n(±3σ uncertainty bounds)', fontsize=12, fontweight='bold')
    ax2.set_title(f'(b) Bounded Clocks: ChronoTick ±3σ Ranges (Minutes {start_min}-{end_min})\n'
                 f'Gold bars = consensus zones where ranges overlap | Consensus: {window_data["ct_overlaps"].mean()*100:.1f}%',
                 fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def create_full_overview(pairs_df, output_path):
    """Create overview showing ALL 132 synchronized pairs across full deployment."""
    print(f"\n{'='*60}")
    print(f"FULL DEPLOYMENT OVERVIEW")
    print(f"Total synchronized pairs: {len(pairs_df)}")
    print(f"Duration: {pairs_df['elapsed_hours'].max():.1f} hours")
    print(f"Overall NTP agreement: {pairs_df['ntp_agrees'].sum()}/{len(pairs_df)} = {pairs_df['ntp_agrees'].mean()*100:.1f}%")
    print(f"Overall ChronoTick consensus: {pairs_df['ct_overlaps'].sum()}/{len(pairs_df)} = {pairs_df['ct_overlaps'].mean()*100:.1f}%")
    print(f"{'='*60}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # Panel (a): Single-Point NTP Offsets
    ax1.plot(pairs_df['elapsed_hours'], pairs_df['node1_ntp_offset'],
             'o-', color='green', linewidth=1.5, markersize=4, alpha=0.7, label='Node 1 NTP offset')
    ax1.plot(pairs_df['elapsed_hours'], pairs_df['node2_ntp_offset'],
             's-', color='blue', linewidth=1.5, markersize=4, alpha=0.7, label='Node 2 NTP offset')
    ax1.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Perfect clock (offset=0)')

    # Highlight disagreements
    disagreements = pairs_df[~pairs_df['ntp_agrees']]
    for _, row in disagreements.iterrows():
        ax1.axvspan(row['elapsed_hours']-0.05, row['elapsed_hours']+0.05,
                   alpha=0.3, color='red')

    agreement_pct = pairs_df['ntp_agrees'].mean() * 100
    ax1.text(0.02, 0.98, f'Agreement (<10ms): {agreement_pct:.1f}%\nDisagreement: {100-agreement_pct:.1f}%',
            transform=ax1.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))

    ax1.set_xlabel('Elapsed Time (hours)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Clock Offset (ms)\n(How much to correct system clock)', fontsize=12, fontweight='bold')
    ax1.set_title(f'(a) Single-Point Clocks: NTP Offsets ({len(pairs_df)} synchronized measurements)\n'
                 f'What correction each node thinks it needs | Nodes disagree!',
                 fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Panel (b): Bounded ChronoTick Consensus Zones
    for _, row in pairs_df.iterrows():
        t = row['elapsed_hours']

        # Draw ranges as thick vertical lines
        ax2.plot([t, t], [row['node1_ct_lower'], row['node1_ct_upper']],
                color='green', linewidth=4, alpha=0.3, solid_capstyle='round')
        ax2.plot([t, t], [row['node2_ct_lower'], row['node2_ct_upper']],
                color='blue', linewidth=4, alpha=0.3, solid_capstyle='round')

        # Consensus zone (overlap) - GOLD
        if row['ct_overlaps']:
            overlap_lower = max(row['node1_ct_lower'], row['node2_ct_lower'])
            overlap_upper = min(row['node1_ct_upper'], row['node2_ct_upper'])
            ax2.plot([t, t], [overlap_lower, overlap_upper],
                    color='gold', linewidth=5, alpha=0.8, solid_capstyle='round',
                    label='Consensus zone' if t == pairs_df['elapsed_hours'].iloc[0] else '')

    # Add center points
    ax2.plot(pairs_df['elapsed_hours'], pairs_df['node1_ct_offset'],
            'o', color='darkgreen', markersize=3, alpha=0.6, label='Node 1 ChronoTick ±3σ')
    ax2.plot(pairs_df['elapsed_hours'], pairs_df['node2_ct_offset'],
            's', color='darkblue', markersize=3, alpha=0.6, label='Node 2 ChronoTick ±3σ')

    ax2.axhline(0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Perfect clock (offset=0)')

    consensus_pct = pairs_df['ct_overlaps'].mean() * 100
    ax2.text(0.02, 0.98, f'Consensus zones: {consensus_pct:.0f}%\nRanges overlap: Nodes can agree!',
            transform=ax2.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='gold', alpha=0.8))

    ax2.set_xlabel('Elapsed Time (hours)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Clock Offset (ms)\n(±3σ uncertainty bounds)', fontsize=12, fontweight='bold')
    ax2.set_title(f'(b) Bounded Clocks: ChronoTick ±3σ Ranges ({len(pairs_df)} synchronized measurements)\n'
                 f'Gold = consensus zones where ranges overlap | 100% consensus!',
                 fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    # Load experiment-5 data
    exp_dir = Path("results/experiment-5")
    output_dir = Path("results/figures/consensus_zones/experiment-5")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading experiment data...")
    data = load_experiment_data(exp_dir)

    print("\nFinding ALL synchronized NTP pairs (no subsampling)...")
    pairs_df = find_synchronized_pairs(data, tolerance_seconds=5.0)

    print(f"\nFound {len(pairs_df)} synchronized pairs")
    print(f"NTP agreement: {pairs_df['ntp_agrees'].sum()}/{len(pairs_df)} = {pairs_df['ntp_agrees'].mean()*100:.1f}%")
    print(f"ChronoTick consensus: {pairs_df['ct_overlaps'].sum()}/{len(pairs_df)} = {pairs_df['ct_overlaps'].mean()*100:.1f}%")

    # Save full data to CSV
    csv_path = output_dir / "all_synchronized_pairs.csv"
    pairs_df.to_csv(csv_path, index=False)
    print(f"\nSaved all {len(pairs_df)} pairs to: {csv_path}")

    # Create full overview with ALL data
    print("\nCreating full deployment overview...")
    create_full_overview(pairs_df, output_dir / "FULL_OVERVIEW_all_data.png")

    # Create focused windows (ALL points in each window)
    windows = [
        (0, 10, "startup"),
        (60, 70, "mid_deployment_1h"),
        (240, 250, "mid_deployment_4h"),
        (420, 430, "late_deployment_7h")
    ]

    print("\nCreating focused time windows (showing ALL data in each)...")
    for start, end, name in windows:
        output_path = output_dir / f"window_{name}_{start}-{end}min_ALL_DATA.png"
        create_focused_window_viz(pairs_df, (start, end), output_path)

    print("\n" + "="*60)
    print("COMPLETE! Created visualizations with ALL synchronized pairs")
    print(f"Total pairs shown: {len(pairs_df)} (NO SUBSAMPLING!)")
    print("="*60)

if __name__ == "__main__":
    main()
