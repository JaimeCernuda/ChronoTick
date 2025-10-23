#!/usr/bin/env python3
"""
Analysis script for ChronoTick validation experiments
Analyzes logs and CSVs from homelab, ARES-11, and ARES-12
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re
from datetime import datetime

# Setup
data_dir = Path(__file__).parent
experiments = {
    'homelab': {
        'csv': data_dir / 'homelab' / 'chronotick_client_validation_20251022_094657.csv',
        'log': data_dir / 'homelab' / 'client_validation_20251022_094656.log'
    },
    'ares-11': {
        'csv': data_dir / 'ares-11' / 'chronotick_client_validation_20251022_105514.csv',
        'log': data_dir / 'ares-11' / 'client_validation_20251022_105511.log'
    },
    'ares-12': {
        'csv': data_dir / 'ares-12' / 'chronotick_client_validation_20251022_105732.csv',
        'log': data_dir / 'ares-12' / 'client_validation_20251022_105730.log'
    }
}

def analyze_csv(csv_path, name):
    """Analyze CSV data from experiment"""
    print(f"\n{'='*80}")
    print(f"ANALYZING {name.upper()} CSV")
    print(f"{'='*80}")

    df = pd.read_csv(csv_path)

    print(f"Total samples: {len(df)}")
    print(f"Duration: {df['elapsed_seconds'].max():.1f} seconds ({df['elapsed_seconds'].max()/60:.1f} minutes)")

    # Source distribution
    sources = df['chronotick_source'].value_counts()
    print(f"\nSource distribution:")
    for source, count in sources.items():
        pct = 100 * count / len(df)
        print(f"  {source}: {count} ({pct:.1f}%)")

    # NTP measurements
    ntp_samples = df[df['has_ntp'] == True]
    print(f"\nNTP measurements: {len(ntp_samples)}")
    if len(ntp_samples) > 1:
        ntp_intervals = ntp_samples['sample_number'].diff().dropna()
        print(f"  Mean interval: {ntp_intervals.mean():.1f} samples ({ntp_intervals.mean() * 10:.0f}s)")
        print(f"  Std interval: {ntp_intervals.std():.1f} samples")

    # Offset statistics
    print(f"\nChronoTick offset statistics:")
    print(f"  Mean: {df['chronotick_offset_ms'].mean():.3f} ms")
    print(f"  Std: {df['chronotick_offset_ms'].std():.3f} ms")
    print(f"  Min: {df['chronotick_offset_ms'].min():.3f} ms")
    print(f"  Max: {df['chronotick_offset_ms'].max():.3f} ms")

    if len(ntp_samples) > 0:
        print(f"\nNTP offset statistics:")
        ntp_offsets = ntp_samples['ntp_offset_ms'].dropna()
        if len(ntp_offsets) > 0:
            print(f"  Mean: {ntp_offsets.mean():.3f} ms")
            print(f"  Std: {ntp_offsets.std():.3f} ms")
            print(f"  Min: {ntp_offsets.min():.3f} ms")
            print(f"  Max: {ntp_offsets.max():.3f} ms")

    return df

def analyze_log(log_path, name):
    """Analyze log file for key metrics"""
    print(f"\n{'='*80}")
    print(f"ANALYZING {name.upper()} LOG")
    print(f"{'='*80}")

    stats = {
        'fusion_events': 0,
        'dataset_writes': 0,
        'ntp_accepted': 0,
        'ntp_rejected': 0,
        'ntp_additions': 0,
        'errors': 0
    }

    with open(log_path, 'r') as f:
        for line in f:
            if 'FUSION HAPPENING' in line:
                stats['fusion_events'] += 1
            elif 'Wrote 30/30 predictions' in line:
                stats['dataset_writes'] += 1
            elif '✓ Dataset now has 2 total measurements' in line:
                stats['ntp_additions'] += 1
            elif 'accepted' in line and 'rejected' in line and 'rejection rate' in line:
                # Extract acceptance/rejection counts
                match = re.search(r'(\d+) accepted, (\d+) rejected', line)
                if match:
                    stats['ntp_accepted'] = int(match.group(1))
                    stats['ntp_rejected'] = int(match.group(2))
            elif 'ERROR' in line or 'CRITICAL' in line:
                stats['errors'] += 1

    print(f"Fusion events: {stats['fusion_events']}")
    print(f"Dataset writes (30/30): {stats['dataset_writes']}")
    print(f"NTP dataset additions: {stats['ntp_additions']}")
    if stats['ntp_accepted'] > 0 or stats['ntp_rejected'] > 0:
        total = stats['ntp_accepted'] + stats['ntp_rejected']
        rejection_rate = 100 * stats['ntp_rejected'] / total if total > 0 else 0
        print(f"NTP acceptance: {stats['ntp_accepted']} accepted, {stats['ntp_rejected']} rejected ({rejection_rate:.1f}% rejection)")
    print(f"Errors/Critical: {stats['errors']}")

    return stats

def create_plots(dfs):
    """Create comprehensive analysis plots"""
    print(f"\n{'='*80}")
    print(f"CREATING ANALYSIS PLOTS")
    print(f"{'='*80}")

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('ChronoTick Validation Experiments Analysis', fontsize=16, fontweight='bold')

    colors = {'homelab': '#1f77b4', 'ares-11': '#ff7f0e', 'ares-12': '#2ca02c'}

    for idx, (name, df) in enumerate(dfs.items()):
        color = colors[name]

        # Row 1: Offset over time
        ax = axes[0, idx]
        ax.plot(df['elapsed_seconds'] / 60, df['chronotick_offset_ms'],
                label='ChronoTick', alpha=0.7, linewidth=0.5, color=color)
        ntp_data = df[df['has_ntp'] == True]
        if len(ntp_data) > 0:
            ax.scatter(ntp_data['elapsed_seconds'] / 60, ntp_data['ntp_offset_ms'],
                      label='NTP', color='red', alpha=0.6, s=20, marker='x')
        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Offset (ms)')
        ax.set_title(f'{name.upper()}: Offset Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Row 2: Source distribution
        ax = axes[1, idx]
        sources = df['chronotick_source'].value_counts()
        ax.bar(sources.index, sources.values, color=color, alpha=0.7)
        ax.set_xlabel('Source')
        ax.set_ylabel('Count')
        ax.set_title(f'{name.upper()}: Prediction Sources')
        ax.grid(True, alpha=0.3, axis='y')

        # Row 3: Offset histogram
        ax = axes[2, idx]
        ax.hist(df['chronotick_offset_ms'], bins=50, color=color, alpha=0.7, edgecolor='black')
        ax.axvline(df['chronotick_offset_ms'].mean(), color='red',
                   linestyle='--', linewidth=2, label=f'Mean: {df["chronotick_offset_ms"].mean():.2f}ms')
        ax.set_xlabel('Offset (ms)')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{name.upper()}: Offset Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = data_dir / 'analysis_plots.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Plots saved to: {output_path}")

    # Create comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('ChronoTick Cross-Experiment Comparison', fontsize=16, fontweight='bold')

    # Offset comparison over time
    ax = axes[0, 0]
    for name, df in dfs.items():
        ax.plot(df['elapsed_seconds'] / 60, df['chronotick_offset_ms'],
                label=name, alpha=0.7, linewidth=1, color=colors[name])
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Offset (ms)')
    ax.set_title('ChronoTick Offset: All Experiments')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # NTP offset comparison
    ax = axes[0, 1]
    for name, df in dfs.items():
        ntp_data = df[df['has_ntp'] == True]
        if len(ntp_data) > 0:
            ax.plot(ntp_data['elapsed_seconds'] / 60, ntp_data['ntp_offset_ms'],
                   label=name, alpha=0.7, marker='o', markersize=3, color=colors[name])
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('NTP Offset (ms)')
    ax.set_title('NTP Ground Truth: All Experiments')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Offset distribution comparison
    ax = axes[1, 0]
    for name, df in dfs.items():
        ax.hist(df['chronotick_offset_ms'], bins=50, alpha=0.5,
                label=f'{name} (μ={df["chronotick_offset_ms"].mean():.2f}ms)',
                color=colors[name], edgecolor='black')
    ax.set_xlabel('Offset (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('Offset Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Source distribution comparison
    ax = axes[1, 1]
    source_data = []
    labels = []
    for name, df in dfs.items():
        sources = df['chronotick_source'].value_counts()
        fusion_pct = 100 * sources.get('fusion', 0) / len(df)
        cpu_pct = 100 * sources.get('cpu', 0) / len(df)
        source_data.append([fusion_pct, cpu_pct])
        labels.append(name)

    x = np.arange(len(labels))
    width = 0.35
    source_data = np.array(source_data)

    ax.bar(x - width/2, source_data[:, 0], width, label='Fusion', alpha=0.7)
    ax.bar(x + width/2, source_data[:, 1], width, label='CPU', alpha=0.7)
    ax.set_ylabel('Percentage (%)')
    ax.set_title('Prediction Source Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = data_dir / 'comparison_plots.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison plots saved to: {output_path}")

def main():
    """Main analysis function"""
    print("="*80)
    print("CHRONOTICK VALIDATION EXPERIMENTS - COMPREHENSIVE ANALYSIS")
    print("="*80)
    print(f"Analysis timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Analyze CSVs
    dfs = {}
    for name, paths in experiments.items():
        df = analyze_csv(paths['csv'], name)
        dfs[name] = df

    # Analyze logs
    stats = {}
    for name, paths in experiments.items():
        stat = analyze_log(paths['log'], name)
        stats[name] = stat

    # Create plots
    create_plots(dfs)

    # Summary table
    print(f"\n{'='*80}")
    print(f"SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Metric':<30} {'Homelab':<15} {'ARES-11':<15} {'ARES-12':<15}")
    print(f"{'-'*75}")

    for name in ['homelab', 'ares-11', 'ares-12']:
        df = dfs[name]
        st = stats[name]

        if name == 'homelab':
            print(f"{'Total samples':<30} {len(df):<15} {len(dfs['ares-11']):<15} {len(dfs['ares-12']):<15}")
            print(f"{'Duration (min)':<30} {df['elapsed_seconds'].max()/60:<15.1f} {dfs['ares-11']['elapsed_seconds'].max()/60:<15.1f} {dfs['ares-12']['elapsed_seconds'].max()/60:<15.1f}")
            print(f"{'Fusion events':<30} {st['fusion_events']:<15} {stats['ares-11']['fusion_events']:<15} {stats['ares-12']['fusion_events']:<15}")
            print(f"{'Dataset writes':<30} {st['dataset_writes']:<15} {stats['ares-11']['dataset_writes']:<15} {stats['ares-12']['dataset_writes']:<15}")
            print(f"{'NTP additions':<30} {st['ntp_additions']:<15} {stats['ares-11']['ntp_additions']:<15} {stats['ares-12']['ntp_additions']:<15}")

            ntp_homelab = len(dfs['homelab'][dfs['homelab']['has_ntp'] == True])
            ntp_ares11 = len(dfs['ares-11'][dfs['ares-11']['has_ntp'] == True])
            ntp_ares12 = len(dfs['ares-12'][dfs['ares-12']['has_ntp'] == True])
            print(f"{'NTP measurements':<30} {ntp_homelab:<15} {ntp_ares11:<15} {ntp_ares12:<15}")

            fusion_homelab = df['chronotick_source'].value_counts().get('fusion', 0)
            fusion_ares11 = dfs['ares-11']['chronotick_source'].value_counts().get('fusion', 0)
            fusion_ares12 = dfs['ares-12']['chronotick_source'].value_counts().get('fusion', 0)
            print(f"{'Fusion predictions':<30} {fusion_homelab:<15} {fusion_ares11:<15} {fusion_ares12:<15}")

            mean_homelab = df['chronotick_offset_ms'].mean()
            mean_ares11 = dfs['ares-11']['chronotick_offset_ms'].mean()
            mean_ares12 = dfs['ares-12']['chronotick_offset_ms'].mean()
            print(f"{'Mean offset (ms)':<30} {mean_homelab:<15.3f} {mean_ares11:<15.3f} {mean_ares12:<15.3f}")

            std_homelab = df['chronotick_offset_ms'].std()
            std_ares11 = dfs['ares-11']['chronotick_offset_ms'].std()
            std_ares12 = dfs['ares-12']['chronotick_offset_ms'].std()
            print(f"{'Std offset (ms)':<30} {std_homelab:<15.3f} {std_ares11:<15.3f} {std_ares12:<15.3f}")

            break

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")
    print(f"Output files:")
    print(f"  - {data_dir / 'analysis_plots.png'}")
    print(f"  - {data_dir / 'comparison_plots.png'}")

if __name__ == '__main__':
    main()
