#!/usr/bin/env python3
"""
Comprehensive analysis of all ChronoTick experiments to select best datasets for paper figures.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def analyze_dataset(csv_path):
    """Analyze a single dataset and return metrics."""
    try:
        df = pd.read_csv(csv_path)

        if 'has_ntp' not in df.columns or df['has_ntp'].sum() == 0:
            return None

        df_ntp = df[df['has_ntp'] == True].copy()

        # Calculate error
        df_ntp['error_ms'] = df_ntp['chronotick_offset_ms'] - df_ntp['ntp_offset_ms']

        # Compute metrics
        metrics = {
            'path': str(csv_path),
            'total_samples': len(df),
            'ntp_measurements': len(df_ntp),
            'duration_hours': df['elapsed_seconds'].max() / 3600,
            'ntp_interval_minutes': (df['elapsed_seconds'].max() / 60) / len(df_ntp) if len(df_ntp) > 0 else 0,
            'mae_ms': df_ntp['error_ms'].abs().mean(),
            'rmse_ms': np.sqrt((df_ntp['error_ms'] ** 2).mean()),
            'max_error_ms': df_ntp['error_ms'].abs().max(),
            'mean_offset_ms': df_ntp['ntp_offset_ms'].mean(),
            'std_offset_ms': df_ntp['ntp_offset_ms'].std(),
        }

        # Determine platform
        path_str = str(csv_path).lower()
        if 'homelab' in path_str:
            metrics['platform'] = 'homelab'
        elif 'wsl' in path_str or 'local' in path_str:
            metrics['platform'] = 'wsl2'
        elif 'ares' in path_str or 'comp-11' in path_str or 'comp-12' in path_str:
            metrics['platform'] = 'ares'
        else:
            metrics['platform'] = 'unknown'

        # Determine node for ARES
        if 'comp-11' in path_str or 'ares-11' in path_str:
            metrics['node'] = 'comp-11'
        elif 'comp-12' in path_str or 'ares-12' in path_str:
            metrics['node'] = 'comp-12'
        else:
            metrics['node'] = None

        # Determine experiment
        if '/experiment-' in path_str:
            exp_num = path_str.split('/experiment-')[1].split('/')[0]
            metrics['experiment'] = f'experiment-{exp_num}'
        else:
            metrics['experiment'] = 'unknown'

        return metrics
    except Exception as e:
        print(f"Error analyzing {csv_path}: {e}")
        return None

def main():
    """Analyze all experiments."""
    results_dir = Path('results')

    # Find all CSV files
    csv_files = list(results_dir.glob('**/data.csv'))
    csv_files.extend(results_dir.glob('**/chronotick_client_validation_*.csv'))

    print(f"Found {len(csv_files)} CSV files\n")

    # Analyze each dataset
    all_metrics = []
    for csv_file in csv_files:
        metrics = analyze_dataset(csv_file)
        if metrics:
            all_metrics.append(metrics)

    # Group by NTP interval category
    interval_2min = []  # 1.5-3 minutes
    interval_10min = []  # 8-12 minutes
    interval_1hour = []  # 45-75 minutes

    for m in all_metrics:
        interval = m['ntp_interval_minutes']
        if 1.5 <= interval <= 3:
            interval_2min.append(m)
        elif 8 <= interval <= 12:
            interval_10min.append(m)
        elif 45 <= interval <= 75:
            interval_1hour.append(m)

    print("=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)

    print(f"\n2-MINUTE INTERVAL DATASETS ({len(interval_2min)} found):")
    print("-" * 80)
    for m in sorted(interval_2min, key=lambda x: x['mae_ms']):
        node_str = m.get('node') or 'N/A'
        print(f"{m['experiment']:15} {m['platform']:10} {node_str:10} "
              f"{m['duration_hours']:5.1f}h  {m['ntp_measurements']:4} NTP  "
              f"MAE: {m['mae_ms']:7.3f} ms")

    print(f"\n10-MINUTE INTERVAL DATASETS ({len(interval_10min)} found):")
    print("-" * 80)
    for m in sorted(interval_10min, key=lambda x: x['mae_ms']):
        node_str = m.get('node') or 'N/A'
        print(f"{m['experiment']:15} {m['platform']:10} {node_str:10} "
              f"{m['duration_hours']:5.1f}h  {m['ntp_measurements']:4} NTP  "
              f"MAE: {m['mae_ms']:7.3f} ms")

    print(f"\n1-HOUR INTERVAL DATASETS ({len(interval_1hour)} found):")
    print("-" * 80)
    if interval_1hour:
        for m in sorted(interval_1hour, key=lambda x: x['mae_ms']):
            node_str = m.get('node') or 'N/A'
            print(f"{m['experiment']:15} {m['platform']:10} {node_str:10} "
                  f"{m['duration_hours']:5.1f}h  {m['ntp_measurements']:4} NTP  "
                  f"MAE: {m['mae_ms']:7.3f} ms")
    else:
        print("No 1-hour interval datasets found")

    # Best datasets for each category
    print("\n" + "=" * 80)
    print("RECOMMENDED DATASETS FOR FIGURES")
    print("=" * 80)

    if interval_2min:
        best_2min = min(interval_2min, key=lambda x: x['mae_ms'])
        print(f"\nBest 2-minute interval dataset:")
        print(f"  {best_2min['path']}")
        print(f"  MAE: {best_2min['mae_ms']:.3f} ms, Duration: {best_2min['duration_hours']:.1f}h")

    if interval_10min:
        best_10min = min(interval_10min, key=lambda x: x['mae_ms'])
        print(f"\nBest 10-minute interval dataset:")
        print(f"  {best_10min['path']}")
        print(f"  MAE: {best_10min['mae_ms']:.3f} ms, Duration: {best_10min['duration_hours']:.1f}h")

    if interval_1hour:
        best_1hour = min(interval_1hour, key=lambda x: x['mae_ms'])
        print(f"\nBest 1-hour interval dataset:")
        print(f"  {best_1hour['path']}")
        print(f"  MAE: {best_1hour['mae_ms']:.3f} ms, Duration: {best_1hour['duration_hours']:.1f}h")

    # Multi-platform datasets (prefer same experiment)
    print(f"\nMulti-platform datasets (for cross-platform comparison):")
    print("-" * 80)

    # Group by experiment
    by_experiment = {}
    for m in all_metrics:
        exp = m['experiment']
        if exp not in by_experiment:
            by_experiment[exp] = []
        by_experiment[exp].append(m)

    # Find experiments with multiple platforms
    for exp, datasets in sorted(by_experiment.items()):
        platforms = set(m['platform'] for m in datasets)
        if len(platforms) >= 2:
            print(f"\n{exp}: {len(platforms)} platforms")
            for m in sorted(datasets, key=lambda x: (x['platform'], x.get('node') or '')):
                node_str = m.get('node') or 'N/A'
                print(f"    {m['platform']:10} {node_str:10} "
                      f"{m['duration_hours']:5.1f}h  MAE: {m['mae_ms']:7.3f} ms")

    # Multi-node datasets (ARES only)
    print(f"\nMulti-node datasets (for node agreement analysis):")
    print("-" * 80)

    for exp, datasets in sorted(by_experiment.items()):
        ares_nodes = [m for m in datasets if m.get('node')]
        if len(ares_nodes) >= 2:
            print(f"\n{exp}: {len(ares_nodes)} ARES nodes")
            for m in sorted(ares_nodes, key=lambda x: x['node']):
                print(f"    {m['node']:10} {m['duration_hours']:5.1f}h  "
                      f"MAE: {m['mae_ms']:7.3f} ms  NTP interval: {m['ntp_interval_minutes']:.1f} min")

    # Save results to JSON
    output_file = Path('results/dataset_analysis.json')
    with open(output_file, 'w') as f:
        json.dump({
            'all_datasets': all_metrics,
            'by_interval': {
                '2min': interval_2min,
                '10min': interval_10min,
                '1hour': interval_1hour
            },
            'by_experiment': by_experiment
        }, f, indent=2)

    print(f"\n\nFull analysis saved to: {output_file}")

if __name__ == '__main__':
    main()
