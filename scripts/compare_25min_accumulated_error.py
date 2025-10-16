#!/usr/bin/env python3
"""
Compare accumulated absolute error for 25-minute windows across different configurations.
Shows System Clock vs ChronoTick side-by-side for each configuration.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define configurations to compare
CONFIGS = {
    "Short-Only": {
        "file": "tsfm/results/ntp_correction_experiment/8hour_tests/test2_short_advanced/summary_advanced_20251012_005722.csv",
        "label": "Short-Only"
    },
    "Dual + None": {
        "file": "tsfm/results/ntp_correction_experiment/overnight_8hr_FIXED_20251014/summary_backtracking_20251014_155930.csv",
        "label": "Dual + None"
    },
    "Dual + Linear": {
        "file": "~/.local/share/Trash/files/tsfm/results/ntp_correction_experiment/experiment_2_25min_linear/summary_linear_20251011_164706.csv",
        "label": "Dual + Linear"
    },
    "Dual + Drift-Aware": {
        "file": "~/.local/share/Trash/files/tsfm/results/ntp_correction_experiment/experiment_3_25min_drift_aware/summary_drift_aware_20251011_171506.csv",
        "label": "Dual + Drift-Aware"
    },
    "Dual + Temporal": {
        "file": "~/.local/share/Trash/files/tsfm/results/ntp_correction_experiment/experiment_4_25min_advanced/summary_advanced_20251011_174307.csv",
        "label": "Dual + Temporal"
    },
}

def load_and_process(filepath, max_minutes=25):
    """Load CSV and extract first 25 minutes with accumulated errors."""
    filepath = Path(filepath).expanduser()
    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return None, None

    df = pd.read_csv(filepath)

    # Filter to only rows with ground truth
    df = df[df['has_ntp'] == True].copy()

    if len(df) == 0:
        print(f"Warning: No ground truth in {filepath.name}")
        return None, None

    # Filter to first 25 minutes (1500 seconds)
    max_seconds = max_minutes * 60
    df = df[df['elapsed_seconds'] <= max_seconds].copy()

    if len(df) == 0:
        print(f"Warning: No data within {max_minutes} minutes in {filepath.name}")
        return None, None

    # Calculate absolute errors
    df['chronotick_abs_error'] = df['chronotick_error_ms'].abs()
    df['system_abs_error'] = df['system_error_ms'].abs()

    # Calculate accumulated errors
    chronotick_accumulated = df['chronotick_abs_error'].sum()
    system_accumulated = df['system_abs_error'].sum()

    return chronotick_accumulated, system_accumulated

def create_comparison_plot(results, output_path):
    """Create vertical grouped bar chart comparing system vs chronotick."""

    # Prepare data
    labels = []
    system_errors = []
    chronotick_errors = []

    for config_name, (chrono, system) in results.items():
        if chrono is not None and system is not None:
            labels.append(config_name)
            system_errors.append(system)
            chronotick_errors.append(chrono)

    # Sort by chronotick performance (worst to best)
    sorted_indices = np.argsort(chronotick_errors)[::-1]
    labels = [labels[i] for i in sorted_indices]
    system_errors = [system_errors[i] for i in sorted_indices]
    chronotick_errors = [chronotick_errors[i] for i in sorted_indices]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))

    x = np.arange(len(labels))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, system_errors, width, label='System Clock',
                   color='#ff7f0e', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, chronotick_errors, width, label='ChronoTick',
                   color='#2ca02c', alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on top of bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.0f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Customize plot
    ax.set_ylabel('Total Accumulated Absolute Error (ms)', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=12, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add subtle background colors to distinguish configurations
    for i in range(len(labels)):
        if i % 2 == 0:
            ax.axvspan(i - 0.5, i + 0.5, alpha=0.05, color='gray', zorder=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    return fig

def print_summary(results):
    """Print summary table."""
    print("\n" + "="*90)
    print("25-MINUTE ACCUMULATED ERROR COMPARISON")
    print("="*90)
    print(f"\n{'Configuration':<25} | {'System Clock (ms)':<20} | {'ChronoTick (ms)':<20} | {'Improvement':<12}")
    print("-" * 90)

    # Sort by chronotick performance
    sorted_results = sorted(results.items(),
                          key=lambda x: x[1][0] if x[1][0] else float('inf'),
                          reverse=True)

    for config_name, (chrono, system) in sorted_results:
        if chrono is not None and system is not None:
            improvement = ((system - chrono) / system) * 100
            print(f"{config_name:<25} | {system:>18.1f} | {chrono:>18.1f} | {improvement:>10.1f}%")
        else:
            print(f"{config_name:<25} | {'N/A':<20} | {'N/A':<20} | {'N/A':<12}")

    print("="*90 + "\n")

def main():
    """Main function."""
    print("Loading 25-minute experiment data...\n")

    results = {}
    for config_name, config_info in CONFIGS.items():
        print(f"  Processing {config_name}...")
        chrono, system = load_and_process(config_info['file'])
        results[config_info['label']] = (chrono, system)

    # Print summary
    print_summary(results)

    # Create plot
    output_path = Path(__file__).parent.parent / "tsfm/results/ntp_correction_experiment/25min_system_vs_chronotick_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    create_comparison_plot(results, output_path)

    print("Analysis complete!")

if __name__ == "__main__":
    main()
