#!/usr/bin/env python3
"""
Analyze 25-minute NTP correction experiments.
Calculate accumulated absolute error for each configuration and create comparison plots.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define experiment configurations
EXPERIMENTS = {
    "DUAL + NONE": "~/.local/share/Trash/files/tsfm/results/ntp_correction_experiment/experiment_1_25min_none/summary_none_20251011_161905.csv",
    "DUAL + LINEAR": "~/.local/share/Trash/files/tsfm/results/ntp_correction_experiment/experiment_2_25min_linear/summary_linear_20251011_164706.csv",
    "DUAL + DRIFT_AWARE": "~/.local/share/Trash/files/tsfm/results/ntp_correction_experiment/experiment_3_25min_drift_aware/summary_drift_aware_20251011_171506.csv",
    "DUAL + ADVANCED": "~/.local/share/Trash/files/tsfm/results/ntp_correction_experiment/experiment_4_25min_advanced/summary_advanced_20251011_174307.csv",
    "DUAL + ADVANCE_ABSOLUTE": "~/.local/share/Trash/files/tsfm/results/ntp_correction_experiment/experiment_5_25min_advance_absolute/summary_advance_absolute_20251011_192804.csv",
}

def load_experiment_data(filepath):
    """Load experiment CSV and process it."""
    filepath = Path(filepath).expanduser()
    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return None

    df = pd.read_csv(filepath)

    # Filter to only rows where we have ground truth (has_ntp=True)
    df_with_truth = df[df['has_ntp'] == True].copy()

    if len(df_with_truth) == 0:
        print(f"Warning: No ground truth data in {filepath.name}")
        return None

    # Calculate absolute errors
    df_with_truth['chronotick_abs_error'] = df_with_truth['chronotick_error_ms'].abs()
    df_with_truth['system_abs_error'] = df_with_truth['system_error_ms'].abs()

    # Calculate accumulated absolute error (cumulative sum)
    df_with_truth['chronotick_accumulated_error'] = df_with_truth['chronotick_abs_error'].cumsum()
    df_with_truth['system_accumulated_error'] = df_with_truth['system_abs_error'].cumsum()

    return df_with_truth

def calculate_statistics(df):
    """Calculate key statistics for an experiment."""
    if df is None or len(df) == 0:
        return None

    stats = {
        'final_accumulated_error': df['chronotick_accumulated_error'].iloc[-1],
        'mean_abs_error': df['chronotick_abs_error'].mean(),
        'max_abs_error': df['chronotick_abs_error'].max(),
        'std_error': df['chronotick_abs_error'].std(),
        'num_measurements': len(df),
        'duration_seconds': df['elapsed_seconds'].iloc[-1],
    }

    return stats

def plot_comparison(experiments_data, output_path):
    """Create simple bar chart comparing total accumulated error including system clock."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    # Prepare data including system clock baseline
    names = []
    chronotick_errors = []
    system_errors = []

    for name, data in experiments_data.items():
        if data['df'] is not None and data['stats']:
            names.append(name)
            chronotick_errors.append(data['stats']['final_accumulated_error'])
            # System clock accumulated error
            system_accumulated = data['df']['system_abs_error'].cumsum().iloc[-1]
            system_errors.append(system_accumulated)

    # Add system clock as a separate entry
    names.append("System Clock")
    # Use the average system clock accumulated error from the experiments
    avg_system_error = np.mean(system_errors)
    chronotick_errors.append(avg_system_error)

    # Sort by accumulated error (worst to best)
    sorted_indices = np.argsort(chronotick_errors)[::-1]
    names = [names[i] for i in sorted_indices]
    chronotick_errors = [chronotick_errors[i] for i in sorted_indices]

    # Create colors (worst = red, best = green)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(names)))

    # Create horizontal bar chart
    bars = ax.barh(names, chronotick_errors, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, error in zip(bars, chronotick_errors):
        ax.text(error + max(chronotick_errors) * 0.02, bar.get_y() + bar.get_height()/2,
                f'{error:.1f} ms', va='center', fontsize=11, fontweight='bold')

    ax.set_xlabel('Total Accumulated Absolute Error (ms)', fontsize=13)
    ax.set_title('25-Minute Experiment: Total Accumulated Error Comparison\n(Sorted Worst to Best)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()  # Best at bottom (reversed for horizontal)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    return fig

def print_summary_table(experiments_data):
    """Print a summary table of all experiments."""
    print("\n" + "="*100)
    print("ACCUMULATED ERROR ANALYSIS - 25-MINUTE EXPERIMENTS")
    print("="*100)

    # Sort by final accumulated error (worst to best)
    sorted_experiments = sorted(
        experiments_data.items(),
        key=lambda x: x[1]['stats']['final_accumulated_error'] if x[1]['stats'] else float('inf'),
        reverse=True
    )

    print(f"\n{'Configuration':<30} | {'Final Accum.':<15} | {'Mean Error':<12} | {'Max Error':<12} | {'Std Dev':<12} | {'# Meas.':<8}")
    print(f"{'(Worst to Best)':<30} | {'Error (ms)':<15} | {'(ms)':<12} | {'(ms)':<12} | {'(ms)':<12} | {'':<8}")
    print("-" * 100)

    for name, data in sorted_experiments:
        stats = data['stats']
        if stats:
            print(f"{name:<30} | {stats['final_accumulated_error']:>13.2f} | "
                  f"{stats['mean_abs_error']:>10.2f} | "
                  f"{stats['max_abs_error']:>10.2f} | "
                  f"{stats['std_error']:>10.2f} | "
                  f"{stats['num_measurements']:>6}")
        else:
            print(f"{name:<30} | {'N/A':<15} | {'N/A':<12} | {'N/A':<12} | {'N/A':<12} | {'N/A':<8}")

    print("="*100)

    # Print improvement percentages
    if len(sorted_experiments) >= 2:
        best_name, best_data = sorted_experiments[-1]
        worst_name, worst_data = sorted_experiments[0]

        if best_data['stats'] and worst_data['stats']:
            best_error = best_data['stats']['final_accumulated_error']
            worst_error = worst_data['stats']['final_accumulated_error']
            improvement = ((worst_error - best_error) / worst_error) * 100

            print(f"\nBEST: {best_name} with {best_error:.2f} ms accumulated error")
            print(f"WORST: {worst_name} with {worst_error:.2f} ms accumulated error")
            print(f"IMPROVEMENT: {improvement:.1f}% reduction in accumulated error")

    print()

def main():
    """Main analysis function."""
    print("Loading experiment data...")

    experiments_data = {}
    for name, filepath in EXPERIMENTS.items():
        print(f"  - {name}...")
        df = load_experiment_data(filepath)
        stats = calculate_statistics(df)
        experiments_data[name] = {'df': df, 'stats': stats}

    # Print summary table
    print_summary_table(experiments_data)

    # Create comparison plot
    output_path = Path(__file__).parent.parent / "tsfm/results/ntp_correction_experiment/25min_accumulated_error_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(experiments_data, output_path)

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
