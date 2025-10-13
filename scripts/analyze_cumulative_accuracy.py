#!/usr/bin/env python3
"""
Analyze cumulative accuracy evolution over time for long-term stability test.
Shows how average error from [0, x] evolves as x increases.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

def load_data(csv_path):
    """Load and prepare the stability test data."""
    df = pd.read_csv(csv_path)
    df['elapsed_hours'] = df['elapsed_seconds'] / 3600
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

def calculate_cumulative_metrics(df):
    """Calculate cumulative accuracy metrics from time 0 to each point x."""
    # Filter only rows with NTP measurements
    ntp_valid = df['ntp_offset_ms'].notna()
    df_ntp = df[ntp_valid].copy().reset_index(drop=True)

    if len(df_ntp) == 0:
        print("ERROR: No NTP measurements found!")
        return None

    # Calculate absolute errors
    df_ntp['chronotick_abs_error'] = df_ntp['chronotick_error_vs_ntp_ms'].abs()
    df_ntp['system_abs_error'] = df_ntp['system_error_vs_ntp_ms'].abs()

    # Calculate cumulative mean absolute error from [0, x]
    df_ntp['chronotick_cumulative_mae'] = df_ntp['chronotick_abs_error'].expanding().mean()
    df_ntp['system_cumulative_mae'] = df_ntp['system_abs_error'].expanding().mean()

    # Calculate cumulative RMS error from [0, x]
    df_ntp['chronotick_cumulative_rms'] = np.sqrt(
        (df_ntp['chronotick_error_vs_ntp_ms'] ** 2).expanding().mean()
    )
    df_ntp['system_cumulative_rms'] = np.sqrt(
        (df_ntp['system_error_vs_ntp_ms'] ** 2).expanding().mean()
    )

    # Calculate rolling window metrics (1 hour = ~36 NTP samples at 100s intervals)
    window_size = 36
    df_ntp['chronotick_rolling_mae'] = df_ntp['chronotick_abs_error'].rolling(
        window=window_size, min_periods=1, center=True
    ).mean()
    df_ntp['system_rolling_mae'] = df_ntp['system_abs_error'].rolling(
        window=window_size, min_periods=1, center=True
    ).mean()

    return df_ntp

def create_cumulative_accuracy_plots(df, df_ntp, output_dir):
    """Create cumulative accuracy evolution visualizations."""
    output_dir = Path(output_dir)

    # Create comprehensive figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Cumulative Accuracy Evolution - 8.5 Hour Test', fontsize=14, fontweight='bold')

    # 1. Cumulative Mean Absolute Error (MAE)
    ax = axes[0, 0]
    ax.plot(df_ntp['elapsed_hours'], df_ntp['chronotick_cumulative_mae'],
            'b-', linewidth=2, label='ChronoTick Cumulative MAE')
    ax.plot(df_ntp['elapsed_hours'], df_ntp['system_cumulative_mae'],
            'r-', linewidth=2, label='System Clock Cumulative MAE')
    ax.set_xlabel('Elapsed Time (hours)', fontsize=11)
    ax.set_ylabel('Cumulative Mean Absolute Error (ms)', fontsize=11)
    ax.set_title('Cumulative MAE: Average Error from [0, x]', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)

    # Add shaded regions for performance phases
    if len(df_ntp) > 0:
        # Mark 3-hour boundary
        ax.axvline(x=3, color='orange', linestyle='--', alpha=0.5, linewidth=2)
        ax.text(3.1, ax.get_ylim()[1] * 0.95, 'Hour 3\n(Performance Change?)',
                fontsize=9, color='orange')

    # 2. Cumulative RMS Error
    ax = axes[0, 1]
    ax.plot(df_ntp['elapsed_hours'], df_ntp['chronotick_cumulative_rms'],
            'b-', linewidth=2, label='ChronoTick Cumulative RMS')
    ax.plot(df_ntp['elapsed_hours'], df_ntp['system_cumulative_rms'],
            'r-', linewidth=2, label='System Clock Cumulative RMS')
    ax.set_xlabel('Elapsed Time (hours)', fontsize=11)
    ax.set_ylabel('Cumulative RMS Error (ms)', fontsize=11)
    ax.set_title('Cumulative RMS: Root Mean Square Error from [0, x]', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.axvline(x=3, color='orange', linestyle='--', alpha=0.5, linewidth=2)

    # 3. Rolling Window MAE (1-hour window)
    ax = axes[1, 0]
    ax.plot(df_ntp['elapsed_hours'], df_ntp['chronotick_rolling_mae'],
            'b-', linewidth=2, label='ChronoTick Rolling MAE (1hr)')
    ax.plot(df_ntp['elapsed_hours'], df_ntp['system_rolling_mae'],
            'r-', linewidth=2, label='System Clock Rolling MAE (1hr)')
    ax.set_xlabel('Elapsed Time (hours)', fontsize=11)
    ax.set_ylabel('Rolling Mean Absolute Error (ms)', fontsize=11)
    ax.set_title('Rolling MAE: Recent 1-Hour Window Performance', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.axvline(x=3, color='orange', linestyle='--', alpha=0.5, linewidth=2)

    # 4. Improvement over system clock
    ax = axes[1, 1]
    improvement_mae = ((df_ntp['system_cumulative_mae'] - df_ntp['chronotick_cumulative_mae']) /
                       df_ntp['system_cumulative_mae'] * 100)
    improvement_rms = ((df_ntp['system_cumulative_rms'] - df_ntp['chronotick_cumulative_rms']) /
                       df_ntp['system_cumulative_rms'] * 100)

    ax.plot(df_ntp['elapsed_hours'], improvement_mae,
            'g-', linewidth=2, label='MAE Improvement')
    ax.plot(df_ntp['elapsed_hours'], improvement_rms,
            'm-', linewidth=2, label='RMS Improvement')
    ax.axhline(y=0, color='k', linestyle='--', alpha=0.5, linewidth=1.5,
               label='No improvement (0%)')
    ax.set_xlabel('Elapsed Time (hours)', fontsize=11)
    ax.set_ylabel('ChronoTick Improvement (%)', fontsize=11)
    ax.set_title('ChronoTick vs System Clock: % Improvement Over Time', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.axvline(x=3, color='orange', linestyle='--', alpha=0.5, linewidth=2)

    # Shade regions
    ax.fill_between(df_ntp['elapsed_hours'], 0, improvement_mae,
                     where=(improvement_mae >= 0), alpha=0.2, color='green',
                     label='Better than system')
    ax.fill_between(df_ntp['elapsed_hours'], 0, improvement_mae,
                     where=(improvement_mae < 0), alpha=0.2, color='red',
                     label='Worse than system')

    plt.tight_layout()
    plt.savefig(output_dir / 'cumulative_accuracy_evolution.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'cumulative_accuracy_evolution.png'}")

    # Create focused single plot for clarity
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(df_ntp['elapsed_hours'], df_ntp['chronotick_cumulative_mae'],
            'b-', linewidth=3, label='ChronoTick Cumulative MAE', alpha=0.8)
    ax.plot(df_ntp['elapsed_hours'], df_ntp['system_cumulative_mae'],
            'r-', linewidth=3, label='System Clock Cumulative MAE', alpha=0.8)

    # Highlight the 3-hour mark
    ax.axvline(x=3, color='orange', linestyle='--', alpha=0.6, linewidth=2)
    ax.axvspan(0, 3, alpha=0.1, color='green', label='Early phase (0-3hr)')
    ax.axvspan(3, df_ntp['elapsed_hours'].iloc[-1], alpha=0.1, color='red',
               label='Late phase (3-8.5hr)')

    ax.set_xlabel('Elapsed Time (hours)', fontsize=13)
    ax.set_ylabel('Cumulative Mean Absolute Error from [0, x] (ms)', fontsize=13)
    ax.set_title('Evolution of Average Accuracy: How Mean Error Grows Over Time',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='upper left')

    # Add annotations
    if len(df_ntp) > 0:
        # Annotate start
        start_ct = df_ntp['chronotick_cumulative_mae'].iloc[0]
        start_sys = df_ntp['system_cumulative_mae'].iloc[0]
        ax.annotate(f'{start_ct:.1f}ms', xy=(0.1, start_ct),
                   xytext=(0.3, start_ct + 10),
                   arrowprops=dict(arrowstyle='->', color='blue', alpha=0.6),
                   fontsize=10, color='blue')

        # Annotate 3-hour mark
        idx_3hr = (df_ntp['elapsed_hours'] - 3).abs().idxmin()
        mae_3hr_ct = df_ntp['chronotick_cumulative_mae'].iloc[idx_3hr]
        mae_3hr_sys = df_ntp['system_cumulative_mae'].iloc[idx_3hr]
        ax.annotate(f'3hr: {mae_3hr_ct:.1f}ms', xy=(3, mae_3hr_ct),
                   xytext=(3.5, mae_3hr_ct + 5),
                   arrowprops=dict(arrowstyle='->', color='blue', alpha=0.6),
                   fontsize=10, color='blue')

        # Annotate end
        end_ct = df_ntp['chronotick_cumulative_mae'].iloc[-1]
        end_sys = df_ntp['system_cumulative_mae'].iloc[-1]
        end_time = df_ntp['elapsed_hours'].iloc[-1]
        ax.annotate(f'End: {end_ct:.1f}ms', xy=(end_time, end_ct),
                   xytext=(end_time - 1, end_ct + 10),
                   arrowprops=dict(arrowstyle='->', color='blue', alpha=0.6),
                   fontsize=10, color='blue')

    plt.tight_layout()
    plt.savefig(output_dir / 'cumulative_mae_focused.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'cumulative_mae_focused.png'}")

    plt.close('all')

def analyze_phase_differences(df_ntp):
    """Analyze performance in early vs late phases."""
    # Split at 3 hours
    early = df_ntp[df_ntp['elapsed_hours'] <= 3]
    late = df_ntp[df_ntp['elapsed_hours'] > 3]

    print("\n" + "="*60)
    print("PHASE ANALYSIS: Early (0-3hr) vs Late (3-8.5hr)")
    print("="*60)

    if len(early) > 0:
        print(f"\nEARLY PHASE (0-3 hours, {len(early)} NTP measurements):")
        print(f"  ChronoTick MAE: {early['chronotick_abs_error'].mean():.2f} ms")
        print(f"  System Clock MAE: {early['system_abs_error'].mean():.2f} ms")
        print(f"  ChronoTick RMS: {np.sqrt((early['chronotick_error_vs_ntp_ms']**2).mean()):.2f} ms")
        print(f"  System Clock RMS: {np.sqrt((early['system_error_vs_ntp_ms']**2).mean()):.2f} ms")

        early_improvement = ((early['system_abs_error'].mean() - early['chronotick_abs_error'].mean()) /
                            early['system_abs_error'].mean() * 100)
        print(f"  ChronoTick Improvement: {early_improvement:.1f}%")

    if len(late) > 0:
        print(f"\nLATE PHASE (3-8.5 hours, {len(late)} NTP measurements):")
        print(f"  ChronoTick MAE: {late['chronotick_abs_error'].mean():.2f} ms")
        print(f"  System Clock MAE: {late['system_abs_error'].mean():.2f} ms")
        print(f"  ChronoTick RMS: {np.sqrt((late['chronotick_error_vs_ntp_ms']**2).mean()):.2f} ms")
        print(f"  System Clock RMS: {np.sqrt((late['system_error_vs_ntp_ms']**2).mean()):.2f} ms")

        late_improvement = ((late['system_abs_error'].mean() - late['chronotick_abs_error'].mean()) /
                           late['system_abs_error'].mean() * 100)
        print(f"  ChronoTick Improvement: {late_improvement:.1f}%")

    if len(early) > 0 and len(late) > 0:
        print(f"\nPHASE COMPARISON:")
        ct_degradation = late['chronotick_abs_error'].mean() - early['chronotick_abs_error'].mean()
        sys_degradation = late['system_abs_error'].mean() - early['system_abs_error'].mean()
        print(f"  ChronoTick error increase: {ct_degradation:.2f} ms ({ct_degradation/early['chronotick_abs_error'].mean()*100:.1f}%)")
        print(f"  System error increase: {sys_degradation:.2f} ms ({sys_degradation/early['system_abs_error'].mean()*100:.1f}%)")

    print("="*60)

def main():
    # Find the most recent CSV file
    results_dir = Path('results/long_term_stability')
    csv_files = list(results_dir.glob('chronotick_stability_*.csv'))

    if not csv_files:
        print("ERROR: No stability test CSV files found!")
        sys.exit(1)

    csv_path = max(csv_files, key=lambda p: p.stat().st_mtime)
    print(f"Analyzing: {csv_path}")

    # Load data
    df = load_data(csv_path)
    print(f"Loaded {len(df)} total samples")

    # Calculate cumulative metrics
    df_ntp = calculate_cumulative_metrics(df)
    if df_ntp is None:
        sys.exit(1)

    print(f"Found {len(df_ntp)} NTP measurements")
    print(f"Time range: {df_ntp['elapsed_hours'].iloc[0]:.2f}h to {df_ntp['elapsed_hours'].iloc[-1]:.2f}h")

    # Create visualizations
    print("\nGenerating cumulative accuracy visualizations...")
    output_dir = Path('results/long_term_stability')
    create_cumulative_accuracy_plots(df, df_ntp, output_dir)

    # Analyze phase differences
    analyze_phase_differences(df_ntp)

    # Print final cumulative values
    print(f"\nFINAL CUMULATIVE METRICS (entire 8.5 hours):")
    print(f"  ChronoTick Cumulative MAE: {df_ntp['chronotick_cumulative_mae'].iloc[-1]:.2f} ms")
    print(f"  System Clock Cumulative MAE: {df_ntp['system_cumulative_mae'].iloc[-1]:.2f} ms")
    print(f"  ChronoTick Cumulative RMS: {df_ntp['chronotick_cumulative_rms'].iloc[-1]:.2f} ms")
    print(f"  System Clock Cumulative RMS: {df_ntp['system_cumulative_rms'].iloc[-1]:.2f} ms")

    overall_improvement = ((df_ntp['system_cumulative_mae'].iloc[-1] -
                          df_ntp['chronotick_cumulative_mae'].iloc[-1]) /
                          df_ntp['system_cumulative_mae'].iloc[-1] * 100)
    print(f"  Overall MAE Improvement: {overall_improvement:.1f}%")

    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()
