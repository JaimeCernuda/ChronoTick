#!/usr/bin/env python3
"""
Create three plots for 8-hour FIXED results:
a) Offset comparison without smoothing
b) Offset comparison with smoothing
c) Accumulated error evolution
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
    """Create a smooth trend using rolling average."""
    df_temp = pd.DataFrame({'x': x, 'y': y})
    df_temp = df_temp.sort_values('x')
    df_temp['y_smooth'] = df_temp['y'].rolling(window=window_size, center=True, min_periods=1).mean()
    return df_temp['x'].values, df_temp['y_smooth'].values

def plot_offset_raw(df, output_path):
    """Plot system clock vs ChronoTick offset errors WITHOUT smoothing."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    time_hours = df['elapsed_hours'].values
    system_error = df['system_error_ms'].values
    chronotick_error = df['chronotick_error_ms'].values
    chronotick_uncertainty = df['ntp_uncertainty_ms'].values

    # Plot system clock error (red) - raw data
    ax.plot(time_hours, system_error, color='#E74C3C', linewidth=1.5,
            label='System Clock Error', alpha=0.8)

    # Plot ChronoTick error (blue) - raw data
    ax.plot(time_hours, chronotick_error, color='#3498DB', linewidth=1.5,
            label='ChronoTick Error', alpha=0.8)

    # Add uncertainty band
    ax.fill_between(time_hours,
                     chronotick_error - chronotick_uncertainty,
                     chronotick_error + chronotick_uncertainty,
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
    print(f"Raw offset plot saved to: {output_path}")
    plt.close()

def plot_offset_smooth(df, output_path):
    """Plot system clock vs ChronoTick offset errors WITH smoothing."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    time_hours = df['elapsed_hours'].values
    system_error = df['system_error_ms'].values
    chronotick_error = df['chronotick_error_ms'].values
    chronotick_uncertainty = df['ntp_uncertainty_ms'].values

    # Create smooth trends
    time_smooth_sys, system_smooth = create_smooth_trend(time_hours, system_error, window_size=15)
    time_smooth_ct, chronotick_smooth = create_smooth_trend(time_hours, chronotick_error, window_size=15)
    time_smooth_unc, unc_smooth = create_smooth_trend(time_hours, chronotick_uncertainty, window_size=15)

    # Plot system clock error (red)
    ax.plot(time_smooth_sys, system_smooth, color='#E74C3C', linewidth=2.5,
            label='System Clock Error', alpha=0.9)

    # Plot ChronoTick error (blue)
    ax.plot(time_smooth_ct, chronotick_smooth, color='#3498DB', linewidth=2.5,
            label='ChronoTick Error', alpha=0.9)

    # Add uncertainty band
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
    print(f"Smooth offset plot saved to: {output_path}")
    plt.close()

def plot_accumulated_error(df, output_path):
    """Plot evolution of accumulated error over time."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Calculate accumulated errors
    df_sorted = df.sort_values('elapsed_hours')
    time_hours = df_sorted['elapsed_hours'].values
    chronotick_accumulated = df_sorted['chronotick_abs_error'].cumsum().values
    system_accumulated = df_sorted['system_abs_error'].cumsum().values

    # Create smooth trends
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
    plt.close()

def main():
    """Main function."""
    # Load data
    data_dir = Path("tsfm/results/ntp_correction_experiment/overnight_8hr_FIXED_20251014")
    summary_file = data_dir / "summary_backtracking_20251014_155930.csv"

    print("Loading data...")
    df = load_data(summary_file)
    print(f"Loaded {len(df)} data points with ground truth")

    # Create all three plots
    print("\n1. Creating raw offset comparison plot...")
    raw_output = data_dir / "offset_comparison_raw.png"
    plot_offset_raw(df, raw_output)

    print("\n2. Creating smooth offset comparison plot...")
    smooth_output = data_dir / "offset_comparison_smooth.png"
    plot_offset_smooth(df, smooth_output)

    print("\n3. Creating accumulated error evolution plot...")
    accumulated_output = data_dir / "accumulated_error_evolution.png"
    plot_accumulated_error(df, accumulated_output)

    print("\n" + "="*60)
    print("All plots created successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
