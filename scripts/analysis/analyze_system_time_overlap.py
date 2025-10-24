#!/usr/bin/env python3
"""
Analyze why system_time matching only produces 25 matches in a narrow window.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def analyze_system_time_overlap():
    """Analyze system_time ranges and overlap between nodes."""

    print("="*80)
    print("SYSTEM TIME OVERLAP ANALYSIS")
    print("="*80)

    # Load data
    df1 = pd.read_csv("results/experiment-7/ares-comp-11/chronotick_client_validation_20251020_220343.csv")
    df2 = pd.read_csv("results/experiment-7/ares-comp-12/chronotick_client_validation_20251020_220540.csv")

    df1_ntp = df1[df1['has_ntp'] == True].copy()
    df2_all = df2.copy()

    print(f"\nNode 1 NTP samples: {len(df1_ntp)}")
    print(f"Node 2 ALL samples: {len(df2_all)}")

    # Analyze system_time ranges
    print(f"\n{'='*80}")
    print("SYSTEM TIME RANGES")
    print('='*80)

    node1_min = df1_ntp['system_time'].min()
    node1_max = df1_ntp['system_time'].max()
    node1_range = node1_max - node1_min

    node2_min = df2_all['system_time'].min()
    node2_max = df2_all['system_time'].max()
    node2_range = node2_max - node2_min

    print(f"\nNode 1 system_time range:")
    print(f"  Min: {node1_min:.2f}")
    print(f"  Max: {node1_max:.2f}")
    print(f"  Range: {node1_range:.2f} seconds ({node1_range/3600:.2f} hours)")

    print(f"\nNode 2 system_time range:")
    print(f"  Min: {node2_min:.2f}")
    print(f"  Max: {node2_max:.2f}")
    print(f"  Range: {node2_range:.2f} seconds ({node2_range/3600:.2f} hours)")

    # Calculate overlap
    overlap_start = max(node1_min, node2_min)
    overlap_end = min(node1_max, node2_max)
    overlap_range = overlap_end - overlap_start

    print(f"\n{'='*80}")
    print("OVERLAP ANALYSIS")
    print('='*80)
    print(f"\nOverlap start: {overlap_start:.2f}")
    print(f"Overlap end: {overlap_end:.2f}")
    print(f"Overlap range: {overlap_range:.2f} seconds ({overlap_range/3600:.2f} hours)")

    if overlap_range > 0:
        print(f"\n✓ Ranges DO overlap!")
        print(f"  Overlap is {overlap_range/node1_range*100:.1f}% of Node 1's range")
        print(f"  Overlap is {overlap_range/node2_range*100:.1f}% of Node 2's range")
    else:
        print(f"\n✗ Ranges DO NOT overlap!")
        print(f"  Gap: {abs(overlap_range):.2f} seconds")

    # Analyze where matches occur
    print(f"\n{'='*80}")
    print("MATCHING ANALYSIS")
    print('='*80)

    # Count how many Node 1 samples fall in overlap range
    node1_in_overlap = df1_ntp[(df1_ntp['system_time'] >= overlap_start) &
                                (df1_ntp['system_time'] <= overlap_end)]
    print(f"\nNode 1 samples in overlap range: {len(node1_in_overlap)} / {len(df1_ntp)} ({len(node1_in_overlap)/len(df1_ntp)*100:.1f}%)")

    # Count how many Node 2 samples fall in overlap range
    node2_in_overlap = df2_all[(df2_all['system_time'] >= overlap_start) &
                                (df2_all['system_time'] <= overlap_end)]
    print(f"Node 2 samples in overlap range: {len(node2_in_overlap)} / {len(df2_all)} ({len(node2_in_overlap)/len(df2_all)*100:.1f}%)")

    # Why only 25 matches?
    print(f"\n{'='*80}")
    print("WHY ONLY 25 MATCHES?")
    print('='*80)
    print(f"\nPotential matches from Node 1 in overlap: {len(node1_in_overlap)}")
    print(f"Actual matches found: 25")
    print(f"Match rate: {25/len(node1_in_overlap)*100:.1f}%")

    print(f"\nReasons for limited matches:")
    print(f"  1. Need exact system_time match (±1 second tolerance)")
    print(f"  2. Node 1 samples NTP every ~36 seconds")
    print(f"  3. Node 2 samples every 1 second")
    print(f"  4. Probability of overlap within 1-second window is low")

    # Calculate expected matches
    node1_sample_interval = node1_range / len(df1_ntp)
    node2_sample_interval = node2_range / len(df2_all)

    print(f"\n{'='*80}")
    print("SAMPLING RATES")
    print('='*80)
    print(f"\nNode 1 average sample interval: {node1_sample_interval:.2f} seconds")
    print(f"Node 2 average sample interval: {node2_sample_interval:.2f} seconds")

    # Expected matches
    expected_matches = len(node1_in_overlap) * (2.0 / node1_sample_interval)  # 2-second window (±1s)
    print(f"\nExpected matches (rough estimate): {expected_matches:.0f}")
    print(f"Actual matches: 25")
    print(f"Efficiency: {25/expected_matches*100:.1f}%")

if __name__ == "__main__":
    analyze_system_time_overlap()
