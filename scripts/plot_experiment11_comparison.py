#!/usr/bin/env python3
"""
Plot Experiment-11 Comparison: ChronoTick vs System Clock vs NTP
Shows offset errors for all three platforms (homelab, ares-comp-11, ares-comp-12)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10

def load_data(csv_path):
    """Load and parse validation CSV"""
    df = pd.read_csv(csv_path)

    # Use column names directly
    df['elapsed'] = df['elapsed_seconds']
    df['timestamp'] = pd.to_datetime(df['datetime'])
    df['system_time'] = df['system_time']
    df['chronotick_time'] = df['chronotick_time']
    df['chronotick_offset_ms'] = df['chronotick_offset_ms']

    # NTP columns (only present when has_ntp is True)
    df['ntp_offset_ms'] = df['ntp_offset_ms']
    df['is_ntp'] = df['has_ntp']

    # Filter to NTP measurements only
    ntp_df = df[df['is_ntp'] == True].copy()

    if len(ntp_df) == 0:
        print(f"WARNING: No NTP measurements found in {csv_path}")
        return None

    # Calculate ground truth (NTP reference time)
    # NTP offset tells us: true_time = system_time + offset
    ntp_df['ntp_reference_time'] = ntp_df['system_time'] + (ntp_df['ntp_offset_ms'] / 1000.0)

    # Calculate errors (in milliseconds)
    # ChronoTick error: how far ChronoTick is from NTP ground truth
    ntp_df['chronotick_error_ms'] = (ntp_df['chronotick_time'] - ntp_df['ntp_reference_time']) * 1000

    # System clock error: how far system clock is from NTP ground truth
    # This should equal the NTP offset (but with opposite sign for plotting)
    ntp_df['system_error_ms'] = (ntp_df['system_time'] - ntp_df['ntp_reference_time']) * 1000

    return ntp_df

def plot_comparison(homelab_df, comp11_df, comp12_df, output_path):
    """Create 3x3 grid comparison plot"""

    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Experiment-11: ChronoTick vs System Clock Performance\n5-Server NTP Averaging with MAD Outlier Rejection',
                 fontsize=16, fontweight='bold', y=0.995)

    platforms = [
        ('Homelab', homelab_df),
        ('ARES comp-11', comp11_df),
        ('ARES comp-12', comp12_df)
    ]

    for row, (platform_name, df) in enumerate(platforms):
        if df is None or len(df) == 0:
            for col in range(3):
                axes[row, col].text(0.5, 0.5, f'No data for {platform_name}',
                                   ha='center', va='center', fontsize=12)
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
            continue

        elapsed_min = df['elapsed'] / 60.0

        # Calculate statistics
        chronotick_mean = df['chronotick_error_ms'].abs().mean()
        chronotick_std = df['chronotick_error_ms'].std()
        system_mean = df['system_error_ms'].abs().mean()
        system_std = df['system_error_ms'].std()
        ntp_mean = df['ntp_offset_ms'].mean()
        ntp_std = df['ntp_offset_ms'].std()

        improvement = ((system_mean - chronotick_mean) / system_mean) * 100

        # Column 1: ChronoTick Error vs NTP
        ax = axes[row, 0]
        ax.plot(elapsed_min, df['chronotick_error_ms'], 'o-', label='ChronoTick Error',
               color='blue', alpha=0.6, markersize=4)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.axhline(y=chronotick_mean, color='blue', linestyle='--', alpha=0.5, linewidth=1.5,
                  label=f'Mean: {chronotick_mean:.2f}ms')
        ax.fill_between(elapsed_min,
                        chronotick_mean - chronotick_std,
                        chronotick_mean + chronotick_std,
                        alpha=0.2, color='blue', label=f'±1σ: {chronotick_std:.2f}ms')

        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('ChronoTick Error (ms)')
        ax.set_title(f'{platform_name}: ChronoTick Error\n'
                    f'Mean Absolute Error: {chronotick_mean:.2f}ms ± {chronotick_std:.2f}ms',
                    fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Column 2: System Clock Error vs NTP
        ax = axes[row, 1]
        ax.plot(elapsed_min, df['system_error_ms'], 'o-', label='System Clock Error',
               color='red', alpha=0.6, markersize=4)
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3, linewidth=0.8)
        ax.axhline(y=system_mean, color='red', linestyle='--', alpha=0.5, linewidth=1.5,
                  label=f'Mean: {system_mean:.2f}ms')
        ax.fill_between(elapsed_min,
                        system_mean - system_std,
                        system_mean + system_std,
                        alpha=0.2, color='red', label=f'±1σ: {system_std:.2f}ms')

        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('System Clock Error (ms)')
        ax.set_title(f'{platform_name}: System Clock Error\n'
                    f'Mean Absolute Error: {system_mean:.2f}ms ± {system_std:.2f}ms',
                    fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

        # Column 3: Direct Comparison
        ax = axes[row, 2]
        ax.plot(elapsed_min, df['chronotick_error_ms'].abs(), 'o-',
               label=f'ChronoTick: {chronotick_mean:.2f}ms',
               color='blue', alpha=0.6, markersize=4)
        ax.plot(elapsed_min, df['system_error_ms'].abs(), 'o-',
               label=f'System Clock: {system_mean:.2f}ms',
               color='red', alpha=0.6, markersize=4)

        # Add improvement annotation
        winner = 'ChronoTick ✅' if chronotick_mean < system_mean else 'System Clock ⚠️'
        improvement_text = f'{abs(improvement):.1f}%'

        ax.text(0.5, 0.95,
               f'{winner}\nImprovement: {improvement_text}',
               transform=ax.transAxes,
               fontsize=11,
               fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgreen' if chronotick_mean < system_mean else 'lightyellow',
                        alpha=0.8),
               ha='center', va='top')

        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Absolute Error (ms)')
        ax.set_title(f'{platform_name}: ChronoTick vs System\n'
                    f'Samples: {len(df)} NTP measurements',
                    fontweight='bold')
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved comparison plot to: {output_path}")

    return fig

def print_summary(homelab_df, comp11_df, comp12_df):
    """Print summary statistics"""

    print("\n" + "="*80)
    print("EXPERIMENT-11 SUMMARY: 5-Server NTP Averaging + 5s Prediction Interval")
    print("="*80)

    platforms = [
        ('Homelab', homelab_df),
        ('ARES comp-11', comp11_df),
        ('ARES comp-12', comp12_df)
    ]

    for platform_name, df in platforms:
        if df is None or len(df) == 0:
            print(f"\n{platform_name}: NO DATA")
            continue

        chronotick_mean = df['chronotick_error_ms'].abs().mean()
        chronotick_std = df['chronotick_error_ms'].std()
        system_mean = df['system_error_ms'].abs().mean()
        system_std = df['system_error_ms'].std()
        ntp_mean = df['ntp_offset_ms'].mean()
        ntp_std = df['ntp_offset_ms'].std()

        improvement = ((system_mean - chronotick_mean) / system_mean) * 100

        runtime_hours = df['elapsed'].max() / 3600

        print(f"\n{platform_name}:")
        print(f"  Runtime: {runtime_hours:.2f} hours ({len(df)} NTP measurements)")
        print(f"  ChronoTick Mean Error: {chronotick_mean:.2f} ± {chronotick_std:.2f} ms")
        print(f"  System Clock Mean Error: {system_mean:.2f} ± {system_std:.2f} ms")
        print(f"  NTP Offset: {ntp_mean:.2f} ± {ntp_std:.2f} ms")

        if chronotick_mean < system_mean:
            print(f"  ✅ ChronoTick WINS by {improvement:.1f}%")
        else:
            print(f"  ⚠️  System Clock WINS by {abs(improvement):.1f}%")

        # NTP stability check
        ntp_range = df['ntp_offset_ms'].max() - df['ntp_offset_ms'].min()
        print(f"  NTP Stability: {ntp_std:.2f}ms std, {ntp_range:.2f}ms range")

        if ntp_std > 2.0:
            print(f"  ⚠️  High NTP variability detected")
        else:
            print(f"  ✅ Stable NTP measurements")

    print("\n" + "="*80)

def main():
    # Load data
    print("Loading Experiment-11 data...")

    homelab_df = load_data('results/experiment-11/homelab/chronotick_client_validation_20251023_124918.csv')
    comp11_df = load_data('results/experiment-11/ares-comp-11/chronotick_client_validation_20251023_134440.csv')
    comp12_df = load_data('results/experiment-11/ares-comp-12/chronotick_client_validation_20251023_134702.csv')

    # Print summary
    print_summary(homelab_df, comp11_df, comp12_df)

    # Create plots
    output_path = 'results/experiment-11/experiment11_comparison.png'
    plot_comparison(homelab_df, comp11_df, comp12_df, output_path)

    print(f"\n✅ Analysis complete!")

if __name__ == '__main__':
    main()
