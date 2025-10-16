#!/usr/bin/env python3
"""
Visualize ChronoTick Performance

Shows:
1. System clock offset errors vs ChronoTick offset errors
2. TimesFM uncertainty bounds for ChronoTick
3. Performance improvement metrics

Usage:
    python visualize_chronotick_performance.py results/ntp_correction_experiment/overnight_8hr_20251013/visualization_data/
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def load_data(data_dir):
    """Load all CSVs from the directory"""
    data_dir = Path(data_dir)

    # Find the CSV files
    summary_files = list(data_dir.glob("summary_*.csv"))
    client_pred_files = list(data_dir.glob("client_predictions_*.csv"))

    if not summary_files:
        print(f"No summary files found in {data_dir}")
        return None, None

    # Load most recent files
    summary_df = pd.read_csv(sorted(summary_files)[-1])
    client_df = pd.read_csv(sorted(client_pred_files)[-1]) if client_pred_files else None

    print(f"Loaded summary: {sorted(summary_files)[-1].name}")
    if client_df is not None:
        print(f"Loaded client predictions: {sorted(client_pred_files)[-1].name}")

    return summary_df, client_df

def plot_error_comparison(summary_df, client_df, output_dir):
    """Plot system clock error vs ChronoTick error with uncertainty bounds"""

    # Filter to rows where we have NTP ground truth
    df = summary_df[summary_df['has_ntp'] == True].copy()

    if len(df) == 0:
        print("No NTP ground truth data available for comparison")
        return

    # Convert timestamps to elapsed time in hours
    start_time = df['timestamp'].min()
    df['elapsed_hours'] = (df['timestamp'] - start_time) / 3600

    # Merge with client predictions to get uncertainty
    if client_df is not None:
        # Match timestamps (allowing small tolerance)
        df['timestamp_rounded'] = df['timestamp'].round(0)
        client_df['timestamp_rounded'] = client_df['timestamp'].round(0)
        df = df.merge(
            client_df[['timestamp_rounded', 'offset_uncertainty_ms']],
            on='timestamp_rounded',
            how='left'
        )

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 10))

    # Main plot: Error comparison with uncertainty bands
    ax1 = plt.subplot(2, 2, (1, 2))

    # Plot system clock error
    ax1.plot(df['elapsed_hours'], df['system_error_ms'],
             'o-', color='red', alpha=0.6, linewidth=2, markersize=4,
             label='System Clock Error (no ChronoTick)')

    # Plot ChronoTick error
    ax1.plot(df['elapsed_hours'], df['chronotick_error_ms'],
             's-', color='green', alpha=0.8, linewidth=2, markersize=4,
             label='ChronoTick Error')

    # Add uncertainty bands if available
    if 'offset_uncertainty_ms' in df.columns and df['offset_uncertainty_ms'].notna().any():
        uncertainty = df['offset_uncertainty_ms'].fillna(0)
        ax1.fill_between(
            df['elapsed_hours'],
            df['chronotick_error_ms'] - uncertainty,
            df['chronotick_error_ms'] + uncertainty,
            color='green', alpha=0.2,
            label='TimesFM Uncertainty (±1σ)'
        )

    ax1.set_xlabel('Time (hours)', fontsize=12)
    ax1.set_ylabel('Absolute Error vs NTP Ground Truth (ms)', fontsize=12)
    ax1.set_title('ChronoTick vs System Clock Performance', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    # Statistics subplot
    ax2 = plt.subplot(2, 2, 3)

    stats_text = []
    stats_text.append("PERFORMANCE STATISTICS\n" + "="*50)
    stats_text.append(f"\nSystem Clock (no ChronoTick):")
    stats_text.append(f"  Mean Error:   {df['system_error_ms'].mean():.3f} ms")
    stats_text.append(f"  Median Error: {df['system_error_ms'].median():.3f} ms")
    stats_text.append(f"  Std Dev:      {df['system_error_ms'].std():.3f} ms")
    stats_text.append(f"  Max Error:    {df['system_error_ms'].max():.3f} ms")

    stats_text.append(f"\nChronoTick:")
    stats_text.append(f"  Mean Error:   {df['chronotick_error_ms'].mean():.3f} ms")
    stats_text.append(f"  Median Error: {df['chronotick_error_ms'].median():.3f} ms")
    stats_text.append(f"  Std Dev:      {df['chronotick_error_ms'].std():.3f} ms")
    stats_text.append(f"  Max Error:    {df['chronotick_error_ms'].max():.3f} ms")

    improvement = (1 - df['chronotick_error_ms'].mean() / df['system_error_ms'].mean()) * 100
    stats_text.append(f"\nImprovement:")
    stats_text.append(f"  Mean Error Reduction: {improvement:.1f}%")

    error_reduction = df['system_error_ms'].mean() - df['chronotick_error_ms'].mean()
    stats_text.append(f"  Absolute Reduction:   {error_reduction:.3f} ms")

    if 'offset_uncertainty_ms' in df.columns:
        avg_uncertainty = df['offset_uncertainty_ms'].mean()
        stats_text.append(f"\nTimesFM Uncertainty:")
        stats_text.append(f"  Average:  {avg_uncertainty:.3f} ms")
        stats_text.append(f"  Min:      {df['offset_uncertainty_ms'].min():.3f} ms")
        stats_text.append(f"  Max:      {df['offset_uncertainty_ms'].max():.3f} ms")

    ax2.text(0.05, 0.95, '\n'.join(stats_text),
             transform=ax2.transAxes,
             fontsize=9, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax2.axis('off')

    # Error distribution histogram
    ax3 = plt.subplot(2, 2, 4)

    bins = np.linspace(0, max(df['system_error_ms'].max(), df['chronotick_error_ms'].max()), 30)
    ax3.hist(df['system_error_ms'], bins=bins, alpha=0.5, color='red', label='System Clock', edgecolor='black')
    ax3.hist(df['chronotick_error_ms'], bins=bins, alpha=0.5, color='green', label='ChronoTick', edgecolor='black')

    ax3.set_xlabel('Absolute Error (ms)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save
    output_path = output_dir / 'chronotick_performance_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")

    plt.close()

    # Also print summary to console
    print("\n" + "="*60)
    print("PERFORMANCE SUMMARY")
    print("="*60)
    print(f"Test duration: {df['elapsed_hours'].max():.2f} hours")
    print(f"Ground truth samples: {len(df)}")
    print(f"\nSystem Clock Error:    {df['system_error_ms'].mean():.3f} ± {df['system_error_ms'].std():.3f} ms")
    print(f"ChronoTick Error:      {df['chronotick_error_ms'].mean():.3f} ± {df['chronotick_error_ms'].std():.3f} ms")
    print(f"Improvement:           {improvement:.1f}% error reduction")
    if 'offset_uncertainty_ms' in df.columns:
        print(f"TimesFM Uncertainty:   {avg_uncertainty:.3f} ms average")
    print("="*60)

def plot_detailed_timeline(summary_df, output_dir):
    """Plot detailed timeline showing offset corrections and sources"""

    df = summary_df.copy()
    start_time = df['timestamp'].min()
    df['elapsed_hours'] = (df['timestamp'] - start_time) / 3600

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Plot 1: Offset corrections by source
    sources = df['client_source'].unique()
    colors = {'ntp': 'blue', 'ntp_warm_up': 'cyan', 'cpu': 'orange', 'gpu': 'purple', 'fusion': 'green'}

    for source in sources:
        mask = df['client_source'] == source
        ax1.scatter(df.loc[mask, 'elapsed_hours'],
                   df.loc[mask, 'client_offset_ms'],
                   label=source.upper(),
                   color=colors.get(source, 'gray'),
                   alpha=0.6, s=20)

    ax1.set_ylabel('Offset Correction (ms)', fontsize=12)
    ax1.set_title('ChronoTick Offset Corrections Over Time', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.3)

    # Plot 2: Dataset size and corrections
    ax2_twin = ax2.twinx()

    ax2.plot(df['elapsed_hours'], df['dataset_size'],
             color='blue', linewidth=2, label='Dataset Size')
    ax2_twin.plot(df['elapsed_hours'], df['corrections_applied'],
                  color='red', linewidth=2, label='Corrections Applied', linestyle='--')

    ax2.set_xlabel('Time (hours)', fontsize=12)
    ax2.set_ylabel('Dataset Size (measurements)', fontsize=12, color='blue')
    ax2_twin.set_ylabel('Corrections Applied', fontsize=12, color='red')
    ax2.set_title('Dataset Growth and NTP Corrections', fontsize=12, fontweight='bold')

    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper left', fontsize=10)
    ax2_twin.legend(loc='upper right', fontsize=10)

    plt.tight_layout()

    output_path = output_dir / 'chronotick_detailed_timeline.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")

    plt.close()

def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_chronotick_performance.py <data_directory>")
        print("Example: python visualize_chronotick_performance.py results/ntp_correction_experiment/overnight_8hr_20251013/visualization_data/")
        sys.exit(1)

    data_dir = Path(sys.argv[1])

    if not data_dir.exists():
        print(f"Error: Directory not found: {data_dir}")
        sys.exit(1)

    print(f"Loading data from: {data_dir}")
    summary_df, client_df = load_data(data_dir)

    if summary_df is None:
        print("Failed to load data")
        sys.exit(1)

    print(f"\nTotal samples: {len(summary_df)}")
    print(f"Samples with NTP ground truth: {summary_df['has_ntp'].sum()}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_error_comparison(summary_df, client_df, data_dir)
    plot_detailed_timeline(summary_df, data_dir)

    print("\n✓ Visualization complete!")

if __name__ == "__main__":
    main()
