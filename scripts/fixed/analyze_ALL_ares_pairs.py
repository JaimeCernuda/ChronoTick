#!/usr/bin/env python3
"""
Analyze ALL ARES node pairs across experiments 1-10.

For each valid pair:
1. Check duration and configuration
2. Calculate start time offset
3. Run aligned truth test
4. Report agreement percentage
5. Calculate overall statistics (mean, std, best)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def analyze_dataset(csv_path):
    """Analyze a single dataset to get metadata."""
    try:
        df = pd.read_csv(csv_path)

        ntp_samples = len(df[df['has_ntp'] == True])
        total_samples = len(df)

        df['timestamp'] = pd.to_datetime(df['datetime'])
        start_time = df['timestamp'].iloc[0]
        end_time = df['timestamp'].iloc[-1]
        duration_hours = (end_time - start_time).total_seconds() / 3600

        # Check for elapsed_seconds
        if 'elapsed_seconds' not in df.columns:
            df['elapsed_seconds'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()

        return {
            'valid': True,
            'total_samples': total_samples,
            'ntp_samples': ntp_samples,
            'start_time': start_time,
            'end_time': end_time,
            'duration_hours': duration_hours,
            'has_elapsed_seconds': 'elapsed_seconds' in df.columns
        }
    except Exception as e:
        return {
            'valid': False,
            'error': str(e)
        }

def find_by_elapsed_time(target_elapsed, df, elapsed_column='elapsed_seconds', tolerance_seconds=5):
    """Find nearest sample by elapsed time."""
    time_diffs = np.abs(df[elapsed_column] - target_elapsed)
    min_diff = time_diffs.min()

    if min_diff <= tolerance_seconds:
        return df.loc[time_diffs.idxmin()]
    return None

def calculate_agreement(node1_csv, node2_csv):
    """Calculate aligned truth test agreement for a node pair."""

    try:
        # Load data
        df1 = pd.read_csv(node1_csv)
        df2 = pd.read_csv(node2_csv)

        df1_ntp = df1[df1['has_ntp'] == True].copy()
        df2_ntp = df2[df2['has_ntp'] == True].copy()
        df1_all = df1.copy()
        df2_all = df2.copy()

        # Parse timestamps
        df1_ntp['timestamp'] = pd.to_datetime(df1_ntp['datetime'])
        df2_ntp['timestamp'] = pd.to_datetime(df2_ntp['datetime'])
        df1_all['timestamp'] = pd.to_datetime(df1_all['datetime'])
        df2_all['timestamp'] = pd.to_datetime(df2_all['datetime'])

        # Calculate start offset
        start1 = df1_ntp['timestamp'].iloc[0]
        start2 = df2_ntp['timestamp'].iloc[0]
        start_offset = (start2 - start1).total_seconds()

        # Compute elapsed_seconds if missing
        if 'elapsed_seconds' not in df1_all.columns:
            df1_all['elapsed_seconds'] = (df1_all['timestamp'] - df1_all['timestamp'].iloc[0]).dt.total_seconds()
            df1_ntp['elapsed_seconds'] = (df1_ntp['timestamp'] - df1_ntp['timestamp'].iloc[0]).dt.total_seconds()
        if 'elapsed_seconds' not in df2_all.columns:
            df2_all['elapsed_seconds'] = (df2_all['timestamp'] - df2_all['timestamp'].iloc[0]).dt.total_seconds()
            df2_ntp['elapsed_seconds'] = (df2_ntp['timestamp'] - df2_ntp['timestamp'].iloc[0]).dt.total_seconds()

        # TEST 1: Node 1 NTP → Node 2 ChronoTick
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
                test1_results.append(agrees)

        # TEST 2: Node 2 NTP → Node 1 ChronoTick
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
                test2_results.append(agrees)

        # Overall agreement
        total_comparisons = len(test1_results) + len(test2_results)
        total_agreements = sum(test1_results) + sum(test2_results)
        overall_agreement = total_agreements / total_comparisons * 100 if total_comparisons > 0 else 0

        agreement1 = sum(test1_results) / len(test1_results) * 100 if test1_results else 0
        agreement2 = sum(test2_results) / len(test2_results) * 100 if test2_results else 0

        return {
            'success': True,
            'start_offset_seconds': start_offset,
            'test1_agreement': agreement1,
            'test1_comparisons': len(test1_results),
            'test2_agreement': agreement2,
            'test2_comparisons': len(test2_results),
            'overall_agreement': overall_agreement,
            'total_comparisons': total_comparisons
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def main():
    """Analyze all ARES node pairs across experiments."""

    print("="*80)
    print("ANALYZING ALL ARES NODE PAIRS ACROSS EXPERIMENTS")
    print("="*80)

    # Define all valid pairs (excluding FAILED)
    pairs = [
        {
            'experiment': 'experiment-5',
            'node1': 'results/experiment-5/ares-comp-11/data.csv',
            'node2': 'results/experiment-5/ares-comp-12/data.csv'
        },
        {
            'experiment': 'experiment-6',
            'node1': 'results/experiment-6/ares-comp-11/data.csv',
            'node2': 'results/experiment-6/ares-comp-12/data.csv'
        },
        {
            'experiment': 'experiment-7',
            'node1': 'results/experiment-7/ares-comp-11/chronotick_client_validation_20251020_220343.csv',
            'node2': 'results/experiment-7/ares-comp-12/chronotick_client_validation_20251020_220540.csv'
        },
        {
            'experiment': 'experiment-9',
            'node1': 'results/experiment-9/ares-11/chronotick_client_validation_20251022_105514.csv',
            'node2': 'results/experiment-9/ares-12/chronotick_client_validation_20251022_105732.csv'
        },
        {
            'experiment': 'experiment-10',
            'node1': 'results/experiment-10/ares-11/chronotick_client_validation_20251022_192420.csv',
            'node2': 'results/experiment-10/ares-12/chronotick_client_validation_20251022_192443.csv'
        }
    ]

    results = []

    for pair in pairs:
        print(f"\n{'='*80}")
        print(f"{pair['experiment'].upper()}")
        print('='*80)

        node1_path = Path(pair['node1'])
        node2_path = Path(pair['node2'])

        if not node1_path.exists():
            print(f"⚠️  Node 1 not found: {node1_path}")
            continue
        if not node2_path.exists():
            print(f"⚠️  Node 2 not found: {node2_path}")
            continue

        # Analyze metadata
        print(f"\nNode 1: {node1_path.name}")
        meta1 = analyze_dataset(node1_path)
        if not meta1['valid']:
            print(f"  ✗ Error: {meta1['error']}")
            continue
        print(f"  Duration: {meta1['duration_hours']:.2f} hours")
        print(f"  NTP samples: {meta1['ntp_samples']}")
        print(f"  Total samples: {meta1['total_samples']}")

        print(f"\nNode 2: {node2_path.name}")
        meta2 = analyze_dataset(node2_path)
        if not meta2['valid']:
            print(f"  ✗ Error: {meta2['error']}")
            continue
        print(f"  Duration: {meta2['duration_hours']:.2f} hours")
        print(f"  NTP samples: {meta2['ntp_samples']}")
        print(f"  Total samples: {meta2['total_samples']}")

        # Check if both ~8 hours
        if meta1['duration_hours'] < 7.5 or meta2['duration_hours'] < 7.5:
            print(f"\n⚠️  Skipping: Duration too short (<7.5 hours)")
            continue

        # Check if same configuration (similar sample counts)
        sample_ratio = meta1['total_samples'] / meta2['total_samples'] if meta2['total_samples'] > 0 else 0
        if sample_ratio < 0.9 or sample_ratio > 1.1:
            print(f"\n⚠️  Warning: Sample counts differ significantly (ratio: {sample_ratio:.2f})")

        # Calculate agreement
        print(f"\nCalculating aligned truth test agreement...")
        agreement_result = calculate_agreement(node1_path, node2_path)

        if not agreement_result['success']:
            print(f"  ✗ Error: {agreement_result['error']}")
            continue

        print(f"\n✓ Results:")
        print(f"  Start offset: {agreement_result['start_offset_seconds']:.1f}s ({agreement_result['start_offset_seconds']/60:.2f} min)")
        print(f"  Test 1 (N1→N2): {agreement_result['test1_agreement']:.1f}% ({agreement_result['test1_comparisons']} comparisons)")
        print(f"  Test 2 (N2→N1): {agreement_result['test2_agreement']:.1f}% ({agreement_result['test2_comparisons']} comparisons)")
        print(f"  Overall Agreement: {agreement_result['overall_agreement']:.1f}% ({agreement_result['total_comparisons']} total)")

        # Store result
        results.append({
            'experiment': pair['experiment'],
            'node1_duration': meta1['duration_hours'],
            'node2_duration': meta2['duration_hours'],
            'start_offset_seconds': agreement_result['start_offset_seconds'],
            'overall_agreement': agreement_result['overall_agreement'],
            'test1_agreement': agreement_result['test1_agreement'],
            'test2_agreement': agreement_result['test2_agreement'],
            'total_comparisons': agreement_result['total_comparisons']
        })

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print('='*80)

    if len(results) == 0:
        print("\n⚠️  No valid results to analyze!")
        return

    agreements = [r['overall_agreement'] for r in results]
    mean_agreement = np.mean(agreements)
    std_agreement = np.std(agreements)
    min_agreement = np.min(agreements)
    max_agreement = np.max(agreements)

    print(f"\nValid experiments analyzed: {len(results)}")
    print(f"\nOverall Agreement Statistics:")
    print(f"  Mean: {mean_agreement:.2f}%")
    print(f"  Std Dev: {std_agreement:.2f}%")
    print(f"  Min: {min_agreement:.2f}%")
    print(f"  Max: {max_agreement:.2f}%")

    # Sort by agreement
    results_sorted = sorted(results, key=lambda x: x['overall_agreement'], reverse=True)

    print(f"\n{'='*80}")
    print("RANKING BY AGREEMENT")
    print('='*80)
    print(f"\n{'Rank':<6} {'Experiment':<15} {'Agreement':<12} {'Comparisons':<12} {'Duration':<12}")
    print('-'*80)

    for i, result in enumerate(results_sorted, 1):
        print(f"{i:<6} {result['experiment']:<15} {result['overall_agreement']:>10.1f}% "
              f"{result['total_comparisons']:>11} {result['node1_duration']:>10.1f}h")

    # Best experiment
    best = results_sorted[0]
    print(f"\n{'='*80}")
    print(f"BEST RESULT: {best['experiment']}")
    print('='*80)
    print(f"Agreement: {best['overall_agreement']:.1f}%")
    print(f"Total comparisons: {best['total_comparisons']}")
    print(f"Start offset: {best['start_offset_seconds']:.1f}s")
    print(f"Test 1 (N1→N2): {best['test1_agreement']:.1f}%")
    print(f"Test 2 (N2→N1): {best['test2_agreement']:.1f}%")

    # Save results
    output_dir = Path("results/figures/5/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "all_ares_pairs_analysis.json", 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved detailed results to: {output_dir / 'all_ares_pairs_analysis.json'}")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print('='*80)

if __name__ == "__main__":
    main()
