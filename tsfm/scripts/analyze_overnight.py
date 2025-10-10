#!/usr/bin/env python3
"""
ChronoTick Overnight Validation Analysis

Analyzes overnight validation data to determine:
1. Average error: System clock vs NTP vs ChronoTick
2. Time series visualization with NTP measurement points
3. Whether ChronoTick is closer to real time than system clock
"""

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

def load_data(csv_path: str) -> pd.DataFrame:
    """Load CSV data and parse timestamps"""
    df = pd.read_csv(csv_path)

    # Convert timestamp to datetime for plotting
    df['dt'] = pd.to_datetime(df['timestamp'], unit='s')

    # Calculate elapsed time in minutes from start
    df['elapsed_minutes'] = (df['timestamp'] - df['timestamp'].min()) / 60

    return df

def calculate_errors(df: pd.DataFrame):
    """Calculate errors when NTP measurements are available"""
    # Filter to only NTP measurement samples
    ntp_samples = df[df['sample_type'] == 'ntp'].copy()

    if len(ntp_samples) == 0:
        print("⚠️  No NTP measurements found in data")
        return None

    # For NTP samples, calculate errors
    # NTP offset is the correction needed, so:
    # - System clock error = ntp_offset_ms (how far off system clock is)
    # - ChronoTick error = chronotick_offset_ms - ntp_offset_ms (difference between prediction and truth)

    ntp_samples['system_error_ms'] = ntp_samples['ntp_offset_ms'].abs()
    ntp_samples['chronotick_error_ms'] = (ntp_samples['chronotick_offset_ms'] - ntp_samples['ntp_offset_ms']).abs()

    return ntp_samples

def print_summary(ntp_samples: pd.DataFrame):
    """Print error analysis summary"""
    print("=" * 80)
    print("CHRONOTICK OVERNIGHT VALIDATION ANALYSIS")
    print("=" * 80)
    print()

    # Overall statistics
    total_samples = len(ntp_samples)
    system_errors = ntp_samples['system_error_ms']
    chronotick_errors = ntp_samples['chronotick_error_ms']

    print(f"📊 NTP Ground Truth Measurements: {total_samples}")
    print()

    # System clock statistics
    print("🕐 SYSTEM CLOCK ERROR (vs NTP Ground Truth)")
    print(f"   Mean:    {system_errors.mean():.3f}ms")
    print(f"   Median:  {system_errors.median():.3f}ms")
    print(f"   Std Dev: {system_errors.std():.3f}ms")
    print(f"   Min:     {system_errors.min():.3f}ms")
    print(f"   Max:     {system_errors.max():.3f}ms")
    print()

    # ChronoTick statistics
    print("🎯 CHRONOTICK ERROR (vs NTP Ground Truth)")
    print(f"   Mean:    {chronotick_errors.mean():.3f}ms")
    print(f"   Median:  {chronotick_errors.median():.3f}ms")
    print(f"   Std Dev: {chronotick_errors.std():.3f}ms")
    print(f"   Min:     {chronotick_errors.min():.3f}ms")
    print(f"   Max:     {chronotick_errors.max():.3f}ms")
    print()

    # Comparison
    improvement_mean = system_errors.mean() - chronotick_errors.mean()
    improvement_pct = (improvement_mean / system_errors.mean()) * 100

    print("📈 ACCURACY COMPARISON")
    print(f"   System Clock Mean Error:    {system_errors.mean():.3f}ms")
    print(f"   ChronoTick Mean Error:      {chronotick_errors.mean():.3f}ms")
    print(f"   Improvement:                {improvement_mean:.3f}ms ({improvement_pct:.1f}%)")
    print()

    # Determine winner
    chronotick_wins = (chronotick_errors < system_errors).sum()
    system_wins = (system_errors < chronotick_errors).sum()
    ties = (chronotick_errors == system_errors).sum()

    print(f"🏆 HEAD-TO-HEAD ACCURACY")
    print(f"   ChronoTick more accurate:  {chronotick_wins}/{total_samples} ({chronotick_wins/total_samples*100:.1f}%)")
    print(f"   System clock more accurate: {system_wins}/{total_samples} ({system_wins/total_samples*100:.1f}%)")
    print(f"   Ties:                      {ties}/{total_samples} ({ties/total_samples*100:.1f}%)")
    print()

    if chronotick_wins > system_wins:
        print("✅ RESULT: ChronoTick is MORE accurate than system clock")
    elif system_wins > chronotick_wins:
        print("❌ RESULT: System clock is MORE accurate than ChronoTick")
    else:
        print("⚖️  RESULT: ChronoTick and system clock have equal accuracy")
    print()

def plot_results(df: pd.DataFrame, ntp_samples: pd.DataFrame, output_dir: str = "/tmp"):
    """Generate visualization plots"""

    # Create figure with multiple subplots
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    # Plot 1: Offset over time
    ax1 = axes[0]
    ax1.plot(df['elapsed_minutes'], df['chronotick_offset_ms'],
             'b-', linewidth=1, alpha=0.6, label='ChronoTick Offset (Predicted)')
    ax1.scatter(ntp_samples['elapsed_minutes'], ntp_samples['ntp_offset_ms'],
                c='red', s=100, marker='o', zorder=5, label='NTP Ground Truth')
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Clock Offset (ms)')
    ax1.set_title('Clock Offset Over Time: ChronoTick Predictions vs NTP Ground Truth')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Prediction error over time
    ax2 = axes[1]
    if len(ntp_samples) > 0:
        ax2.plot(ntp_samples['elapsed_minutes'], ntp_samples['chronotick_error_ms'],
                'g-', linewidth=2, marker='o', label='ChronoTick Error')
        ax2.plot(ntp_samples['elapsed_minutes'], ntp_samples['system_error_ms'],
                'r-', linewidth=2, marker='s', label='System Clock Error')
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Absolute Error (ms)')
        ax2.set_title('Prediction Error vs Ground Truth (NTP measurements only)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    # Plot 3: Uncertainty bounds
    ax3 = axes[2]
    if 'chronotick_uncertainty_ms' in df.columns:
        # Plot offset with uncertainty bands
        ax3.plot(df['elapsed_minutes'], df['chronotick_offset_ms'],
                'b-', linewidth=1, label='ChronoTick Offset')

        # Uncertainty bounds
        upper_bound = df['chronotick_offset_ms'] + df['chronotick_uncertainty_ms']
        lower_bound = df['chronotick_offset_ms'] - df['chronotick_uncertainty_ms']
        ax3.fill_between(df['elapsed_minutes'], lower_bound, upper_bound,
                         alpha=0.3, color='blue', label='±1σ Uncertainty')

        # NTP measurements
        ax3.scatter(ntp_samples['elapsed_minutes'], ntp_samples['ntp_offset_ms'],
                   c='red', s=100, marker='o', zorder=5, label='NTP Truth')

        ax3.set_xlabel('Time (minutes)')
        ax3.set_ylabel('Clock Offset (ms)')
        ax3.set_title('ChronoTick Predictions with Uncertainty Bounds')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = f"{output_dir}/chronotick_overnight_analysis_{timestamp}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"📊 Plots saved to: {output_path}")

    # Also save to a standard location
    standard_path = f"{output_dir}/chronotick_overnight_analysis_latest.png"
    plt.savefig(standard_path, dpi=150, bbox_inches='tight')
    print(f"📊 Also saved to: {standard_path}")

    return output_path

def main():
    if len(sys.argv) < 2:
        print("Usage: uv run python analyze_overnight.py <csv_file>")
        print("Example: uv run python analyze_overnight.py /tmp/chronotick_overnight_validation.csv")
        return 1

    csv_path = sys.argv[1]

    if not Path(csv_path).exists():
        print(f"❌ Error: File not found: {csv_path}")
        return 1

    print(f"📁 Loading data from: {csv_path}")
    df = load_data(csv_path)
    print(f"✅ Loaded {len(df)} samples")
    print()

    # Calculate errors
    ntp_samples = calculate_errors(df)

    if ntp_samples is None or len(ntp_samples) == 0:
        print("❌ No NTP measurements available for analysis")
        return 1

    # Print summary
    print_summary(ntp_samples)

    # Generate plots
    output_path = plot_results(df, ntp_samples)

    print("=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    print()
    print(f"📊 Visualization: {output_path}")
    print(f"📁 Data: {csv_path}")
    print()

    return 0

if __name__ == "__main__":
    sys.exit(main())
