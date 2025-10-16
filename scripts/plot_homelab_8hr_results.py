#!/usr/bin/env python3
"""
Create publication-quality plots for homelab 8-hour experiment results.
Matches the plots from overnight_8hr_backtracking_ENHANCED.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_data(summary_file):
    """Load the summary CSV file."""
    df = pd.read_csv(summary_file)

    # Filter to only rows where we have ground truth (has_ntp=True)
    df_with_truth = df[df['has_ntp'] == True].copy()

    # Convert elapsed seconds to hours
    df_with_truth['elapsed_hours'] = df_with_truth['elapsed_seconds'] / 3600

    # Get absolute errors
    df_with_truth['chronotick_abs_error'] = df_with_truth['chronotick_error_ms'].abs()
    df_with_truth['system_abs_error'] = df_with_truth['system_error_ms'].abs()

    return df_with_truth

def create_smooth_trend(x, y, window_size=20):
    """Create a smooth trend using rolling average to show overall trends."""
    # Create a DataFrame for easier rolling window calculation
    df_temp = pd.DataFrame({'x': x, 'y': y})
    df_temp = df_temp.sort_values('x')

    # Apply rolling mean
    df_temp['y_smooth'] = df_temp['y'].rolling(window=window_size, center=True, min_periods=1).mean()

    return df_temp['x'].values, df_temp['y_smooth'].values

def plot_offset_comparison(df, output_path):
    """Plot system clock vs ChronoTick offset errors over time."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Get data
    time_hours = df['elapsed_hours'].values
    system_error = df['system_error_ms'].values  # Keep sign for offset
    chronotick_error = df['chronotick_error_ms'].values
    chronotick_uncertainty = df['ntp_uncertainty_ms'].values  # Using NTP uncertainty as proxy

    # Create smooth trends (rolling average to show overall trends)
    time_smooth_sys, system_smooth = create_smooth_trend(time_hours, system_error, window_size=15)
    time_smooth_ct, chronotick_smooth = create_smooth_trend(time_hours, chronotick_error, window_size=15)
    time_smooth_unc, unc_smooth = create_smooth_trend(time_hours, chronotick_uncertainty, window_size=15)

    # Plot system clock error (red)
    ax.plot(time_smooth_sys, system_smooth, color='#E74C3C', linewidth=2.5,
            label='System Clock Error', alpha=0.9)

    # Plot ChronoTick error (blue)
    ax.plot(time_smooth_ct, chronotick_smooth, color='#3498DB', linewidth=2.5,
            label='ChronoTick Error', alpha=0.9)

    # Add uncertainty band around ChronoTick (lighter blue)
    ax.fill_between(time_smooth_ct,
                     chronotick_smooth - unc_smooth,
                     chronotick_smooth + unc_smooth,
                     color='#3498DB', alpha=0.2, label='ChronoTick Uncertainty')

    # Customize plot
    ax.set_xlabel('Time (hours)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Offset Error (ms)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Offset comparison plot saved to: {output_path}")

    return fig

def plot_accumulated_error(df, output_path):
    """Plot evolution of accumulated error over time."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Calculate accumulated errors
    df_sorted = df.sort_values('elapsed_hours')
    time_hours = df_sorted['elapsed_hours'].values

    chronotick_accumulated = df_sorted['chronotick_abs_error'].cumsum().values
    system_accumulated = df_sorted['system_abs_error'].cumsum().values

    # Create smooth trends (rolling average to show overall trends)
    time_smooth_sys, system_acc_smooth = create_smooth_trend(time_hours, system_accumulated, window_size=15)
    time_smooth_ct, chronotick_acc_smooth = create_smooth_trend(time_hours, chronotick_accumulated, window_size=15)

    # Plot system clock accumulated error (red)
    ax.plot(time_smooth_sys, system_acc_smooth, color='#E74C3C', linewidth=2.5,
            label='System Clock Accumulated Error', alpha=0.9)

    # Plot ChronoTick accumulated error (green)
    ax.plot(time_smooth_ct, chronotick_acc_smooth, color='#27AE60', linewidth=2.5,
            label='ChronoTick Accumulated Error', alpha=0.9)

    # Customize plot
    ax.set_xlabel('Time (hours)', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accumulated Absolute Error (ms)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Accumulated error plot saved to: {output_path}")

    return fig

def plot_composite_analysis(df, output_path):
    """Create composite plot with multiple analysis panels."""
    fig = plt.figure(figsize=(16, 12))

    # Create 3x2 grid
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    # Panel 1: Offset comparison (top, spans both columns)
    ax1 = fig.add_subplot(gs[0, :])
    time_hours = df['elapsed_hours'].values
    system_error = df['system_error_ms'].values
    chronotick_error = df['chronotick_error_ms'].values
    chronotick_uncertainty = df['ntp_uncertainty_ms'].values

    time_smooth_sys, system_smooth = create_smooth_trend(time_hours, system_error, window_size=15)
    time_smooth_ct, chronotick_smooth = create_smooth_trend(time_hours, chronotick_error, window_size=15)
    time_smooth_unc, unc_smooth = create_smooth_trend(time_hours, chronotick_uncertainty, window_size=15)

    ax1.plot(time_smooth_sys, system_smooth, color='#E74C3C', linewidth=2.5, label='System Clock Error', alpha=0.9)
    ax1.plot(time_smooth_ct, chronotick_smooth, color='#3498DB', linewidth=2.5, label='ChronoTick Error', alpha=0.9)
    ax1.fill_between(time_smooth_ct, chronotick_smooth - unc_smooth, chronotick_smooth + unc_smooth,
                     color='#3498DB', alpha=0.2, label='ChronoTick Uncertainty')
    ax1.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Offset Error (ms)', fontsize=12, fontweight='bold')
    ax1.set_title('Offset Error Over Time', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel 2: Accumulated error
    ax2 = fig.add_subplot(gs[1, :])
    df_sorted = df.sort_values('elapsed_hours')
    time_hours_sorted = df_sorted['elapsed_hours'].values
    chronotick_accumulated = df_sorted['chronotick_abs_error'].cumsum().values
    system_accumulated = df_sorted['system_abs_error'].cumsum().values

    time_smooth_sys, system_acc_smooth = create_smooth_trend(time_hours_sorted, system_accumulated, window_size=15)
    time_smooth_ct, chronotick_acc_smooth = create_smooth_trend(time_hours_sorted, chronotick_accumulated, window_size=15)

    ax2.plot(time_smooth_sys, system_acc_smooth, color='#E74C3C', linewidth=2.5, label='System Clock', alpha=0.9)
    ax2.plot(time_smooth_ct, chronotick_acc_smooth, color='#27AE60', linewidth=2.5, label='ChronoTick', alpha=0.9)
    ax2.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accumulated Absolute Error (ms)', fontsize=12, fontweight='bold')
    ax2.set_title('Accumulated Error Evolution', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Panel 3: Error distribution histogram
    ax3 = fig.add_subplot(gs[2, 0])
    bins = np.linspace(0, max(df['system_abs_error'].max(), df['chronotick_abs_error'].max()), 30)
    ax3.hist(df['system_abs_error'], bins=bins, alpha=0.6, color='#E74C3C', label='System Clock', edgecolor='black')
    ax3.hist(df['chronotick_abs_error'], bins=bins, alpha=0.6, color='#3498DB', label='ChronoTick', edgecolor='black')
    ax3.set_xlabel('Absolute Error (ms)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax3.set_title('Error Distribution', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # Panel 4: Performance statistics
    ax4 = fig.add_subplot(gs[2, 1])
    stats_text = []
    stats_text.append("PERFORMANCE STATISTICS")
    stats_text.append("=" * 45)
    stats_text.append(f"\nTest Duration: {df['elapsed_hours'].max():.2f} hours")
    stats_text.append(f"Ground Truth Samples: {len(df)}")
    stats_text.append(f"\nSystem Clock:")
    stats_text.append(f"  Mean Error:   {df['system_abs_error'].mean():.3f} ms")
    stats_text.append(f"  Median Error: {df['system_abs_error'].median():.3f} ms")
    stats_text.append(f"  Std Dev:      {df['system_abs_error'].std():.3f} ms")
    stats_text.append(f"  Max Error:    {df['system_abs_error'].max():.3f} ms")
    stats_text.append(f"\nChronoTick:")
    stats_text.append(f"  Mean Error:   {df['chronotick_abs_error'].mean():.3f} ms")
    stats_text.append(f"  Median Error: {df['chronotick_abs_error'].median():.3f} ms")
    stats_text.append(f"  Std Dev:      {df['chronotick_abs_error'].std():.3f} ms")
    stats_text.append(f"  Max Error:    {df['chronotick_abs_error'].max():.3f} ms")

    improvement = (1 - df['chronotick_abs_error'].mean() / df['system_abs_error'].mean()) * 100
    error_reduction = df['system_abs_error'].mean() - df['chronotick_abs_error'].mean()

    stats_text.append(f"\nImprovement:")
    stats_text.append(f"  Error Reduction: {improvement:.1f}%")
    stats_text.append(f"  Absolute Reduction: {error_reduction:.3f} ms")

    ax4.text(0.05, 0.95, '\n'.join(stats_text),
             transform=ax4.transAxes, fontsize=8.5, verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    ax4.axis('off')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Composite plot saved to: {output_path}")

    return fig

def main():
    """Main function."""
    # Load data
    data_dir = Path("homelab-execution-1/ntp_correction_experiment/overnight_8hr_20251014_192151")
    summary_file = data_dir / "summary_backtracking_20251014_192212.csv"

    print("Loading homelab data...")
    df = load_data(summary_file)
    print(f"Loaded {len(df)} data points with ground truth")

    # Create plots
    print("\nCreating offset comparison plot...")
    offset_output = data_dir / "offset_comparison_smooth.png"
    plot_offset_comparison(df, offset_output)

    print("\nCreating accumulated error plot...")
    accumulated_output = data_dir / "accumulated_error_evolution.png"
    plot_accumulated_error(df, accumulated_output)

    print("\nCreating composite analysis plot...")
    composite_output = data_dir / "overnight_8hr_homelab_plots.png"
    plot_composite_analysis(df, composite_output)

    print("\nDone!")
    print(f"\nAll plots saved to: {data_dir}")

if __name__ == "__main__":
    main()
