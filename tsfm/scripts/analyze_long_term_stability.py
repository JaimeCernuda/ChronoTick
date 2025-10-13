#!/usr/bin/env python3
"""
Analyze long-term ChronoTick stability test results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def load_data(csv_path):
    """Load and prepare the stability test data."""
    df = pd.read_csv(csv_path)

    # Convert elapsed_seconds to hours for plotting
    df['elapsed_hours'] = df['elapsed_seconds'] / 3600

    # Convert datetime string to pandas datetime
    df['datetime'] = pd.to_datetime(df['datetime'])

    return df

def calculate_metrics(df):
    """Calculate stability metrics."""
    metrics = {}

    # ChronoTick offset statistics
    metrics['chronotick_offset_mean'] = df['chronotick_offset_ms'].mean()
    metrics['chronotick_offset_std'] = df['chronotick_offset_ms'].std()
    metrics['chronotick_offset_range'] = (
        df['chronotick_offset_ms'].max() - df['chronotick_offset_ms'].min()
    )

    # Calculate drift rate (ms/hour)
    hours = df['elapsed_hours'].iloc[-1] - df['elapsed_hours'].iloc[0]
    offset_change = df['chronotick_offset_ms'].iloc[-1] - df['chronotick_offset_ms'].iloc[0]
    metrics['drift_rate_ms_per_hour'] = offset_change / hours if hours > 0 else 0

    # NTP comparison (when available)
    ntp_valid = df['ntp_offset_ms'].notna()
    if ntp_valid.any():
        metrics['ntp_measurements'] = ntp_valid.sum()
        metrics['chronotick_error_vs_ntp_mean'] = df.loc[ntp_valid, 'chronotick_error_vs_ntp_ms'].mean()
        metrics['chronotick_error_vs_ntp_std'] = df.loc[ntp_valid, 'chronotick_error_vs_ntp_ms'].std()
        metrics['system_error_vs_ntp_mean'] = df.loc[ntp_valid, 'system_error_vs_ntp_ms'].mean()
        metrics['system_error_vs_ntp_std'] = df.loc[ntp_valid, 'system_error_vs_ntp_ms'].std()

        # Calculate RMS error
        metrics['chronotick_rms_error'] = np.sqrt(
            (df.loc[ntp_valid, 'chronotick_error_vs_ntp_ms'] ** 2).mean()
        )
        metrics['system_rms_error'] = np.sqrt(
            (df.loc[ntp_valid, 'system_error_vs_ntp_ms'] ** 2).mean()
        )

    # Stability score (lower is better)
    metrics['stability_score'] = (
        abs(metrics['chronotick_offset_mean']) +
        metrics['chronotick_offset_std'] +
        metrics['chronotick_offset_range'] / 10
    )

    return metrics

def create_visualizations(df, metrics, output_dir):
    """Create comprehensive stability visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.suptitle('ChronoTick Long-Term Stability Analysis (8.5 Hours)', fontsize=14, fontweight='bold')

    # 1. ChronoTick offset over time
    ax = axes[0, 0]
    ax.plot(df['elapsed_hours'], df['chronotick_offset_ms'], 'b-', linewidth=0.5, alpha=0.6)
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Zero offset')
    ax.set_xlabel('Elapsed Time (hours)')
    ax.set_ylabel('ChronoTick Offset (ms)')
    ax.set_title('ChronoTick Offset Evolution')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add trend line
    z = np.polyfit(df['elapsed_hours'], df['chronotick_offset_ms'], 1)
    p = np.poly1d(z)
    ax.plot(df['elapsed_hours'], p(df['elapsed_hours']), "r--", alpha=0.8,
            label=f'Trend: {z[0]:.2f} ms/hr')
    ax.legend()

    # 2. ChronoTick vs NTP comparison
    ax = axes[0, 1]
    ntp_valid = df['ntp_offset_ms'].notna()
    if ntp_valid.any():
        ax.scatter(df.loc[ntp_valid, 'elapsed_hours'],
                  df.loc[ntp_valid, 'chronotick_error_vs_ntp_ms'],
                  c='blue', alpha=0.5, s=20, label='ChronoTick error vs NTP')
        ax.scatter(df.loc[ntp_valid, 'elapsed_hours'],
                  df.loc[ntp_valid, 'system_error_vs_ntp_ms'],
                  c='red', alpha=0.5, s=20, label='System clock error vs NTP')
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel('Elapsed Time (hours)')
        ax.set_ylabel('Error vs NTP (ms)')
        ax.set_title('ChronoTick vs System Clock Accuracy (NTP Reference)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No NTP measurements available',
                ha='center', va='center', transform=ax.transAxes)

    # 3. Offset distribution histogram
    ax = axes[1, 0]
    ax.hist(df['chronotick_offset_ms'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax.axvline(x=metrics['chronotick_offset_mean'], color='r', linestyle='--',
               linewidth=2, label=f"Mean: {metrics['chronotick_offset_mean']:.2f} ms")
    ax.set_xlabel('ChronoTick Offset (ms)')
    ax.set_ylabel('Frequency')
    ax.set_title('Offset Distribution')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 4. Rolling statistics (1-hour window)
    ax = axes[1, 1]
    window_size = int(360)  # 1 hour = 360 samples at 10s intervals
    rolling_mean = df['chronotick_offset_ms'].rolling(window=window_size, center=True).mean()
    rolling_std = df['chronotick_offset_ms'].rolling(window=window_size, center=True).std()

    ax.plot(df['elapsed_hours'], rolling_mean, 'b-', linewidth=2, label='1-hour rolling mean')
    ax.fill_between(df['elapsed_hours'],
                     rolling_mean - rolling_std,
                     rolling_mean + rolling_std,
                     alpha=0.3, color='blue', label='±1 std')
    ax.set_xlabel('Elapsed Time (hours)')
    ax.set_ylabel('ChronoTick Offset (ms)')
    ax.set_title('Rolling Statistics (1-Hour Window)')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # 5. Uncertainty over time
    ax = axes[2, 0]
    ax.plot(df['elapsed_hours'], df['chronotick_uncertainty_ms'], 'g-', linewidth=0.5, alpha=0.6)
    ax.set_xlabel('Elapsed Time (hours)')
    ax.set_ylabel('Uncertainty (ms)')
    ax.set_title('ChronoTick Uncertainty Evolution')
    ax.grid(True, alpha=0.3)

    # 6. Metrics summary table
    ax = axes[2, 1]
    ax.axis('off')

    # Format metrics for display
    summary_text = f"""
    Long-Term Stability Metrics (8.5 hours)
    {'='*45}

    ChronoTick Offset:
      Mean: {metrics['chronotick_offset_mean']:.2f} ms
      Std Dev: {metrics['chronotick_offset_std']:.2f} ms
      Range: {metrics['chronotick_offset_range']:.2f} ms

    Drift Rate: {metrics['drift_rate_ms_per_hour']:.2f} ms/hour

    Stability Score: {metrics['stability_score']:.2f}
    """

    if 'chronotick_rms_error' in metrics:
        summary_text += f"""
    Accuracy vs NTP ({metrics['ntp_measurements']:.0f} measurements):
      ChronoTick RMS: {metrics['chronotick_rms_error']:.2f} ms
      System Clock RMS: {metrics['system_rms_error']:.2f} ms

      ChronoTick Mean: {metrics['chronotick_error_vs_ntp_mean']:.2f} ± {metrics['chronotick_error_vs_ntp_std']:.2f} ms
      System Clock Mean: {metrics['system_error_vs_ntp_mean']:.2f} ± {metrics['system_error_vs_ntp_std']:.2f} ms
        """

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(output_dir / 'long_term_stability_analysis.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {output_dir / 'long_term_stability_analysis.png'}")

    # Create focused plot: offset over time with key events
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df['elapsed_hours'], df['chronotick_offset_ms'], 'b-', linewidth=1, alpha=0.7)

    # Mark NTP measurements
    ntp_valid = df['ntp_offset_ms'].notna()
    if ntp_valid.any():
        ax.scatter(df.loc[ntp_valid, 'elapsed_hours'],
                  df.loc[ntp_valid, 'chronotick_offset_ms'],
                  c='red', marker='x', s=50, alpha=0.7, label='NTP measurement')

    # Add trend line
    z = np.polyfit(df['elapsed_hours'], df['chronotick_offset_ms'], 1)
    p = np.poly1d(z)
    ax.plot(df['elapsed_hours'], p(df['elapsed_hours']), "r--", alpha=0.8, linewidth=2,
            label=f'Linear trend: {z[0]:.2f} ms/hr')

    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3, label='Zero offset')
    ax.set_xlabel('Elapsed Time (hours)', fontsize=12)
    ax.set_ylabel('ChronoTick Offset (ms)', fontsize=12)
    ax.set_title('ChronoTick Offset Evolution - 8.5 Hour Stability Test', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'offset_evolution_focused.png', dpi=150, bbox_inches='tight')
    print(f"Saved visualization: {output_dir / 'offset_evolution_focused.png'}")

    plt.close('all')

def main():
    # Find the most recent CSV file
    results_dir = Path('results/long_term_stability')
    csv_files = list(results_dir.glob('chronotick_stability_*.csv'))

    if not csv_files:
        print("ERROR: No stability test CSV files found!")
        sys.exit(1)

    csv_path = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"Analyzing: {csv_path}")
    print(f"File size: {csv_path.stat().st_size / 1024:.1f} KB")

    # Load data
    df = load_data(csv_path)
    print(f"\nLoaded {len(df)} samples")
    print(f"Time range: {df['datetime'].iloc[0]} to {df['datetime'].iloc[-1]}")
    print(f"Duration: {df['elapsed_hours'].iloc[-1]:.2f} hours")

    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = calculate_metrics(df)

    # Print metrics
    print("\n" + "="*60)
    print("LONG-TERM STABILITY ANALYSIS RESULTS")
    print("="*60)
    print(f"\nTest Duration: {df['elapsed_hours'].iloc[-1]:.2f} hours")
    print(f"Samples Collected: {len(df)}")
    print(f"\nChronoTick Offset Statistics:")
    print(f"  Mean: {metrics['chronotick_offset_mean']:.3f} ms")
    print(f"  Std Dev: {metrics['chronotick_offset_std']:.3f} ms")
    print(f"  Range: {metrics['chronotick_offset_range']:.3f} ms")
    print(f"\nDrift Rate: {metrics['drift_rate_ms_per_hour']:.3f} ms/hour")
    print(f"Stability Score: {metrics['stability_score']:.3f} (lower is better)")

    if 'chronotick_rms_error' in metrics:
        print(f"\nAccuracy vs NTP ({metrics['ntp_measurements']:.0f} measurements):")
        print(f"  ChronoTick RMS Error: {metrics['chronotick_rms_error']:.3f} ms")
        print(f"  System Clock RMS Error: {metrics['system_rms_error']:.3f} ms")
        print(f"  ChronoTick Mean Error: {metrics['chronotick_error_vs_ntp_mean']:.3f} ± {metrics['chronotick_error_vs_ntp_std']:.3f} ms")
        print(f"  System Clock Mean Error: {metrics['system_error_vs_ntp_mean']:.3f} ± {metrics['system_error_vs_ntp_std']:.3f} ms")

        improvement = (metrics['system_rms_error'] - metrics['chronotick_rms_error']) / metrics['system_rms_error'] * 100
        print(f"\n  ChronoTick RMS Improvement: {improvement:.1f}%")

    print("="*60)

    # Create visualizations
    print("\nGenerating visualizations...")
    output_dir = Path('results/long_term_stability')
    create_visualizations(df, metrics, output_dir)

    # Save metrics to file
    metrics_file = output_dir / 'stability_metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("LONG-TERM STABILITY ANALYSIS RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Test Duration: {df['elapsed_hours'].iloc[-1]:.2f} hours\n")
        f.write(f"Samples Collected: {len(df)}\n\n")
        f.write("ChronoTick Offset Statistics:\n")
        f.write(f"  Mean: {metrics['chronotick_offset_mean']:.3f} ms\n")
        f.write(f"  Std Dev: {metrics['chronotick_offset_std']:.3f} ms\n")
        f.write(f"  Range: {metrics['chronotick_offset_range']:.3f} ms\n\n")
        f.write(f"Drift Rate: {metrics['drift_rate_ms_per_hour']:.3f} ms/hour\n")
        f.write(f"Stability Score: {metrics['stability_score']:.3f}\n")

        if 'chronotick_rms_error' in metrics:
            f.write(f"\nAccuracy vs NTP ({metrics['ntp_measurements']:.0f} measurements):\n")
            f.write(f"  ChronoTick RMS Error: {metrics['chronotick_rms_error']:.3f} ms\n")
            f.write(f"  System Clock RMS Error: {metrics['system_rms_error']:.3f} ms\n")
            f.write(f"  ChronoTick Mean Error: {metrics['chronotick_error_vs_ntp_mean']:.3f} ± {metrics['chronotick_error_vs_ntp_std']:.3f} ms\n")
            f.write(f"  System Clock Mean Error: {metrics['system_error_vs_ntp_mean']:.3f} ± {metrics['system_error_vs_ntp_std']:.3f} ms\n")

            improvement = (metrics['system_rms_error'] - metrics['chronotick_rms_error']) / metrics['system_rms_error'] * 100
            f.write(f"\n  ChronoTick RMS Improvement: {improvement:.1f}%\n")

    print(f"Saved metrics: {metrics_file}")
    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()
