#!/usr/bin/env python3
"""
Diagnostic analysis of 8-hour test degradation.
Investigate why ChronoTick performance degraded after ~4 hours.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_full_data(summary_file):
    """Load the summary CSV file with all data."""
    df = pd.read_csv(summary_file)
    df['elapsed_hours'] = df['elapsed_seconds'] / 3600
    return df

def analyze_sign_changes(df):
    """Check if ChronoTick predictions oscillate between positive/negative."""
    df_with_truth = df[df['has_ntp'] == True].copy()

    chronotick_errors = df_with_truth['chronotick_error_ms'].values
    system_errors = df_with_truth['system_error_ms'].values
    times = df_with_truth['elapsed_hours'].values

    # Count sign changes
    chronotick_sign_changes = np.sum(np.diff(np.sign(chronotick_errors)) != 0)
    system_sign_changes = np.sum(np.diff(np.sign(system_errors)) != 0)

    print("\n" + "="*70)
    print("SIGN OSCILLATION ANALYSIS")
    print("="*70)
    print(f"ChronoTick sign changes: {chronotick_sign_changes} out of {len(chronotick_errors)-1} transitions")
    print(f"System Clock sign changes: {system_sign_changes} out of {len(system_errors)-1} transitions")
    print(f"ChronoTick oscillation rate: {(chronotick_sign_changes/(len(chronotick_errors)-1))*100:.1f}%")
    print(f"System Clock oscillation rate: {(system_sign_changes/(len(system_errors)-1))*100:.1f}%")

    return df_with_truth

def analyze_uncertainty(df):
    """Analyze NTP uncertainty values."""
    df_with_truth = df[df['has_ntp'] == True].copy()

    uncertainties = df_with_truth['ntp_uncertainty_ms'].values
    times = df_with_truth['elapsed_hours'].values

    print("\n" + "="*70)
    print("UNCERTAINTY ANALYSIS")
    print("="*70)
    print(f"Mean uncertainty: {uncertainties.mean():.2f} ms")
    print(f"Median uncertainty: {np.median(uncertainties):.2f} ms")
    print(f"Min uncertainty: {uncertainties.min():.2f} ms")
    print(f"Max uncertainty: {uncertainties.max():.2f} ms")
    print(f"Std uncertainty: {uncertainties.std():.2f} ms")

    # Check if uncertainties are unusually low
    low_uncertainty_count = np.sum(uncertainties < 10)
    print(f"\nMeasurements with uncertainty < 10ms: {low_uncertainty_count}/{len(uncertainties)} ({(low_uncertainty_count/len(uncertainties))*100:.1f}%)")

    return df_with_truth

def analyze_error_by_time_period(df):
    """Break down error statistics by time period."""
    df_with_truth = df[df['has_ntp'] == True].copy()

    print("\n" + "="*70)
    print("ERROR BY TIME PERIOD")
    print("="*70)

    periods = [
        (0, 2, "Hours 0-2"),
        (2, 4, "Hours 2-4"),
        (4, 6, "Hours 4-6"),
        (6, 8, "Hours 6-8"),
    ]

    for start, end, label in periods:
        period_data = df_with_truth[(df_with_truth['elapsed_hours'] >= start) &
                                     (df_with_truth['elapsed_hours'] < end)]

        if len(period_data) > 0:
            ct_mean = period_data['chronotick_error_ms'].abs().mean()
            ct_median = period_data['chronotick_error_ms'].abs().median()
            ct_max = period_data['chronotick_error_ms'].abs().max()
            ct_std = period_data['chronotick_error_ms'].std()

            sys_mean = period_data['system_error_ms'].abs().mean()

            print(f"\n{label} ({len(period_data)} measurements):")
            print(f"  ChronoTick: mean={ct_mean:.1f}ms, median={ct_median:.1f}ms, max={ct_max:.1f}ms, std={ct_std:.1f}ms")
            print(f"  System Clock: mean={sys_mean:.1f}ms")
            print(f"  Ratio (CT/Sys): {ct_mean/sys_mean:.2f}x")

def find_degradation_point(df):
    """Identify when performance started degrading."""
    df_with_truth = df[df['has_ntp'] == True].copy()

    # Calculate rolling mean of absolute error
    window = 5
    df_with_truth['ct_rolling_error'] = df_with_truth['chronotick_error_ms'].abs().rolling(window=window, min_periods=1).mean()
    df_with_truth['sys_rolling_error'] = df_with_truth['system_error_ms'].abs().rolling(window=window, min_periods=1).mean()

    # Find where ChronoTick becomes worse than system clock
    df_with_truth['ct_worse'] = df_with_truth['ct_rolling_error'] > df_with_truth['sys_rolling_error']

    # Find first sustained period where CT is worse
    first_worse_idx = None
    sustained_count = 0
    required_sustained = 3  # Need 3 consecutive measurements

    for idx, row in df_with_truth.iterrows():
        if row['ct_worse']:
            sustained_count += 1
            if sustained_count >= required_sustained and first_worse_idx is None:
                first_worse_idx = idx
        else:
            sustained_count = 0

    print("\n" + "="*70)
    print("DEGRADATION POINT ANALYSIS")
    print("="*70)

    if first_worse_idx is not None:
        degradation_row = df_with_truth.loc[first_worse_idx]
        print(f"ChronoTick became consistently worse than system clock at:")
        print(f"  Time: {degradation_row['elapsed_hours']:.2f} hours ({degradation_row['elapsed_seconds']/60:.0f} minutes)")
        print(f"  ChronoTick error: {degradation_row['chronotick_error_ms']:.1f} ms")
        print(f"  System error: {degradation_row['system_error_ms']:.1f} ms")
        print(f"  NTP uncertainty: {degradation_row['ntp_uncertainty_ms']:.1f} ms")
        print(f"  Dataset size: {degradation_row['dataset_size']}")
    else:
        print("No sustained degradation point found where ChronoTick became worse.")

    return df_with_truth

def check_for_anomalies(df):
    """Check for specific anomalies or issues."""
    df_with_truth = df[df['has_ntp'] == True].copy()

    print("\n" + "="*70)
    print("ANOMALY DETECTION")
    print("="*70)

    # Check for very large errors
    large_errors = df_with_truth[df_with_truth['chronotick_error_ms'].abs() > 300]
    print(f"\nMeasurements with |ChronoTick error| > 300ms: {len(large_errors)}")
    if len(large_errors) > 0:
        print("  Times when this occurred:")
        for _, row in large_errors.iterrows():
            print(f"    {row['elapsed_hours']:.2f}h: error={row['chronotick_error_ms']:.1f}ms, uncertainty={row['ntp_uncertainty_ms']:.1f}ms")

    # Check for sudden jumps
    df_with_truth['error_change'] = df_with_truth['chronotick_error_ms'].diff().abs()
    large_jumps = df_with_truth[df_with_truth['error_change'] > 200]
    print(f"\nSudden error jumps (>200ms change): {len(large_jumps)}")
    if len(large_jumps) > 0:
        print("  Times when this occurred:")
        for _, row in large_jumps.head(10).iterrows():
            print(f"    {row['elapsed_hours']:.2f}h: jump={row['error_change']:.1f}ms")

    # Check dataset size changes
    df_all = df.copy()
    df_all['dataset_size_change'] = df_all['dataset_size'].diff()
    large_size_changes = df_all[df_all['dataset_size_change'].abs() > 50]
    print(f"\nLarge dataset size changes (>50): {len(large_size_changes)}")
    if len(large_size_changes) > 0:
        print("  Sample of changes:")
        for _, row in large_size_changes.head(5).iterrows():
            print(f"    {row['elapsed_hours']:.2f}h: change={row['dataset_size_change']:.0f}, new size={row['dataset_size']}")

def create_diagnostic_plots(df, output_dir):
    """Create detailed diagnostic plots."""
    df_with_truth = df[df['has_ntp'] == True].copy()

    fig, axes = plt.subplots(4, 1, figsize=(14, 16))

    # Plot 1: Signed errors (not absolute)
    ax1 = axes[0]
    ax1.plot(df_with_truth['elapsed_hours'], df_with_truth['system_error_ms'],
             'r-', linewidth=1.5, label='System Clock Error', alpha=0.7)
    ax1.plot(df_with_truth['elapsed_hours'], df_with_truth['chronotick_error_ms'],
             'b-', linewidth=1.5, label='ChronoTick Error', alpha=0.7)
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_ylabel('Offset Error (ms)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Signed Offset Errors (showing oscillations)', fontsize=13, fontweight='bold')

    # Plot 2: NTP Uncertainty over time
    ax2 = axes[1]
    ax2.plot(df_with_truth['elapsed_hours'], df_with_truth['ntp_uncertainty_ms'],
             'g-', linewidth=2, label='NTP Uncertainty', alpha=0.7)
    ax2.set_ylabel('NTP Uncertainty (ms)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    ax2.set_title('NTP Uncertainty Over Time', fontsize=13, fontweight='bold')

    # Plot 3: Dataset size over time
    ax3 = axes[2]
    ax3.plot(df['elapsed_hours'], df['dataset_size'],
             'purple', linewidth=2, alpha=0.7)
    ax3.set_ylabel('Dataset Size', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.set_title('ML Model Dataset Size Evolution', fontsize=13, fontweight='bold')

    # Plot 4: Rolling error ratio (ChronoTick / System Clock)
    ax4 = axes[3]
    df_with_truth['error_ratio'] = df_with_truth['chronotick_error_ms'].abs() / df_with_truth['system_error_ms'].abs()
    # Apply rolling mean to smooth
    df_with_truth['error_ratio_smooth'] = df_with_truth['error_ratio'].rolling(window=5, min_periods=1).mean()
    ax4.plot(df_with_truth['elapsed_hours'], df_with_truth['error_ratio_smooth'],
             'orange', linewidth=2, alpha=0.7)
    ax4.axhline(y=1.0, color='k', linestyle='--', linewidth=2, alpha=0.5, label='Parity (CT = Sys)')
    ax4.set_ylabel('Error Ratio (CT/Sys)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Time (hours)', fontsize=12, fontweight='bold')
    ax4.legend(loc='best')
    ax4.grid(True, alpha=0.3)
    ax4.set_title('Performance Ratio (values > 1 = ChronoTick worse)', fontsize=13, fontweight='bold')
    ax4.set_ylim([0, max(3, df_with_truth['error_ratio_smooth'].max() * 1.1)])

    plt.tight_layout()
    output_path = output_dir / "diagnostic_analysis.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nDiagnostic plots saved to: {output_path}")
    plt.close()

def main():
    """Main diagnostic function."""
    data_dir = Path("tsfm/results/ntp_correction_experiment/overnight_8hr_FIXED_20251014")
    summary_file = data_dir / "summary_backtracking_20251014_155930.csv"

    print("\n" + "="*70)
    print("8-HOUR TEST DEGRADATION DIAGNOSTIC ANALYSIS")
    print("="*70)

    # Load data
    print("\nLoading data...")
    df = load_full_data(summary_file)
    print(f"Total measurements: {len(df)}")
    print(f"Measurements with NTP ground truth: {len(df[df['has_ntp'] == True])}")

    # Run analyses
    analyze_sign_changes(df)
    analyze_uncertainty(df)
    analyze_error_by_time_period(df)
    find_degradation_point(df)
    check_for_anomalies(df)

    # Create diagnostic plots
    print("\n" + "="*70)
    print("Creating diagnostic plots...")
    create_diagnostic_plots(df, data_dir)

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nCheck the diagnostic_analysis.png file for visual insights.")

if __name__ == "__main__":
    main()
