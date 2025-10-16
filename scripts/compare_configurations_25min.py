#!/usr/bin/env python3
"""
Compare configurations: System Clock vs ChronoTick for first 25 minutes.
Includes both 25-minute tests and first 25 minutes of 8-hour tests.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define experiment configurations
EXPERIMENTS = {
    "Single Mode": {
        "file": "tsfm/results/ntp_correction_experiment/8hour_tests/test2_short_advanced/summary_advanced_20251012_005722.csv",
        "format": "standard",
        "duration_limit": 1500  # 25 minutes = 1500 seconds
    },
    "Dual + None": {
        "file": "tsfm/results/ntp_correction_experiment/8hour_tests/test3_dual_none/chronotick_stability_20251011_023912.csv",
        "format": "alternative",
        "duration_limit": 1500
    },
    "Dual + Backtracking": {
        "file": "tsfm/results/ntp_correction_experiment/8hour_tests/test1_dual_advanced/summary_advanced_20251012_115734.csv",
        "format": "standard",
        "duration_limit": 1500
    },
    "Dual + Interpolation": {
        "file": "~/.local/share/Trash/files/tsfm/results/ntp_correction_experiment/experiment_4_25min_advanced/summary_advanced_20251011_174307.csv",
        "format": "standard",
        "duration_limit": None
    },
    "Dual + Linear": {
        "file": "~/.local/share/Trash/files/tsfm/results/ntp_correction_experiment/experiment_2_25min_linear/summary_linear_20251011_164706.csv",
        "format": "standard",
        "duration_limit": None
    },
    "Dual + Drift Aware": {
        "file": "~/.local/share/Trash/files/tsfm/results/ntp_correction_experiment/experiment_3_25min_drift_aware/summary_drift_aware_20251011_171506.csv",
        "format": "standard",
        "duration_limit": None
    },
}

def load_experiment_data(config):
    """Load experiment CSV and process it."""
    filepath = Path(config["file"]).expanduser()
    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return None

    df = pd.read_csv(filepath)

    # Apply duration limit if specified
    if config["duration_limit"]:
        df = df[df['elapsed_seconds'] <= config["duration_limit"]].copy()

    # Handle different formats
    if config["format"] == "standard":
        # Standard format: chronotick_error_ms, system_error_ms, has_ntp
        df_with_truth = df[df['has_ntp'] == True].copy()
        if len(df_with_truth) == 0:
            print(f"Warning: No ground truth data in {filepath.name}")
            return None

        df_with_truth['chronotick_abs_error'] = df_with_truth['chronotick_error_ms'].abs()
        df_with_truth['system_abs_error'] = df_with_truth['system_error_ms'].abs()

    elif config["format"] == "alternative":
        # Alternative format: chronotick_error_vs_ntp_ms, system_error_vs_ntp_ms
        df_with_truth = df[df['chronotick_error_vs_ntp_ms'].notna()].copy()
        if len(df_with_truth) == 0:
            print(f"Warning: No ground truth data in {filepath.name}")
            return None

        df_with_truth['chronotick_abs_error'] = df_with_truth['chronotick_error_vs_ntp_ms'].abs()
        df_with_truth['system_abs_error'] = df_with_truth['system_error_vs_ntp_ms'].abs()

    return df_with_truth

def calculate_statistics(df):
    """Calculate accumulated error statistics."""
    if df is None or len(df) == 0:
        return None

    chronotick_accumulated = df['chronotick_abs_error'].sum()
    system_accumulated = df['system_abs_error'].sum()

    # Normalize by number of measurements for fair comparison
    chronotick_mean = df['chronotick_abs_error'].mean()
    system_mean = df['system_abs_error'].mean()

    stats = {
        'chronotick_accumulated': chronotick_accumulated,
        'system_accumulated': system_accumulated,
        'chronotick_mean': chronotick_mean,
        'system_mean': system_mean,
        'num_measurements': len(df),
    }

    return stats

def plot_comparison(experiments_data, output_path):
    """Create vertical bar chart comparing System Clock vs ChronoTick using MEAN error."""
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    # Prepare data for paired bars - use MEAN error for fair comparison
    configs = []
    system_errors = []
    chronotick_errors = []

    for name, data in experiments_data.items():
        if data['stats']:
            configs.append(name)
            system_errors.append(data['stats']['system_mean'])
            chronotick_errors.append(data['stats']['chronotick_mean'])

    # Sort by ChronoTick performance (worst to best)
    sorted_indices = np.argsort(chronotick_errors)[::-1]
    configs = [configs[i] for i in sorted_indices]
    system_errors = [system_errors[i] for i in sorted_indices]
    chronotick_errors = [chronotick_errors[i] for i in sorted_indices]

    # Create positions for grouped bars with more spacing
    x = np.arange(len(configs)) * 2.5  # More spacing between groups
    width = 0.8

    # Create bars
    bars1 = ax.bar(x - width/2, system_errors, width,
                   color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, chronotick_errors, width,
                   color='#27AE60', alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars with decimals
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add "System Clock" and "ChronoTick" labels below each pair
    for i, pos in enumerate(x):
        ax.text(pos - width/2, -max(system_errors) * 0.08, 'System\nClock',
               ha='center', va='top', fontsize=10, fontweight='bold')
        ax.text(pos + width/2, -max(system_errors) * 0.08, 'Chrono\nTick',
               ha='center', va='top', fontsize=10, fontweight='bold')

    # Add configuration labels below the pairs
    for i, (pos, config) in enumerate(zip(x, configs)):
        ax.text(pos, -max(system_errors) * 0.18, config,
               ha='center', va='top', fontsize=12, fontweight='bold')

    # Customize plot
    ax.set_ylabel('Mean Absolute Error (ms)', fontsize=14, fontweight='bold')
    ax.set_xticks([])  # Remove x-tick marks
    ax.set_xlim(x[0] - 1.5, x[-1] + 1.5)
    ax.set_ylim(-max(system_errors) * 0.25, max(system_errors) * 1.15)  # Extra space for labels below
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")

    return fig

def plot_unified_comparison(experiments_data, output_path):
    """Create bar chart with single averaged system clock vs all ChronoTick configs."""
    from matplotlib.patches import Rectangle

    fig, ax = plt.subplots(1, 1, figsize=(14, 8))

    # Prepare data
    configs = []
    chronotick_errors = []
    system_errors_list = []

    for name, data in experiments_data.items():
        if data['stats']:
            configs.append(name)
            chronotick_errors.append(data['stats']['chronotick_mean'])
            system_errors_list.append(data['stats']['system_mean'])

    # Calculate average system clock error
    avg_system_error = np.mean(system_errors_list)

    # Sort by ChronoTick performance (worst to best)
    sorted_indices = np.argsort(chronotick_errors)[::-1]
    configs = [configs[i] for i in sorted_indices]
    chronotick_errors = [chronotick_errors[i] for i in sorted_indices]

    # Create positions for bars
    x = np.arange(len(configs) + 1)  # +1 for system clock
    width = 0.6

    # Create bars - System Clock first, then all ChronoTick configs
    colors = ['#E74C3C'] + ['#27AE60'] * len(configs)
    values = [avg_system_error] + chronotick_errors

    bars = ax.bar(x, values, width, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, value in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{value:.1f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Tier 1 x-axis labels (configuration names)
    tier1_labels = [''] + configs  # Empty label for system clock to avoid duplication
    ax.set_xticks(x)
    ax.set_xticklabels(tier1_labels, fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', which='major', pad=15)

    # Tier 2 x-axis labels (group labels) using secondary x-axis
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())

    # Position tier 2 ticks
    tier2_ticks = [0, (1 + len(configs)) / 2]
    tier2_labels = ['System Clock', 'ChronoTick']

    ax2.set_xticks(tier2_ticks)
    ax2.set_xticklabels(tier2_labels, fontsize=15, fontweight='bold')
    ax2.tick_params(axis='x', which='major', pad=5, length=0)

    # Move tier 2 axis to bottom
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 60))
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Draw horizontal line segments on the tier 2 axis to show grouping
    # Calculate the y-position for tier 2 axis
    trans = ax2.get_xaxis_transform()

    # Line under "System Clock" (from -0.3 to 0.3)
    ax2.plot([-0.3, 0.3], [0, 0], color='black', linewidth=3, transform=trans, clip_on=False)

    # Line under "ChronoTick" (from 0.7 to len(configs))
    chronotick_start = 0.7
    chronotick_end = len(configs) + 0.3
    ax2.plot([chronotick_start, chronotick_end], [0, 0], color='black', linewidth=3, transform=trans, clip_on=False)

    # Customize main plot
    ax.set_ylabel('Mean Absolute Error (ms)', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(values) * 1.12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Unified plot saved to: {output_path}")

    return fig

def print_summary_table(experiments_data):
    """Print a summary table of all experiments."""
    print("\n" + "="*110)
    print("MEAN ERROR COMPARISON - FIRST 25 MINUTES (Normalized by # of measurements)")
    print("="*110)

    # Sort by ChronoTick mean error (worst to best)
    sorted_experiments = sorted(
        experiments_data.items(),
        key=lambda x: x[1]['stats']['chronotick_mean'] if x[1]['stats'] else float('inf'),
        reverse=True
    )

    print(f"\n{'Configuration':<25} | {'System Mean':<13} | {'Chrono Mean':<13} | {'Improvement':<12} | {'# Meas.':<8}")
    print(f"{'(Worst to Best)':<25} | {'Error (ms)':<13} | {'Error (ms)':<13} | {'(%)':<12} | {'':<8}")
    print("-" * 110)

    for name, data in sorted_experiments:
        stats = data['stats']
        if stats:
            improvement = ((stats['system_mean'] - stats['chronotick_mean']) /
                          stats['system_mean']) * 100
            print(f"{name:<25} | {stats['system_mean']:>11.2f} | "
                  f"{stats['chronotick_mean']:>11.2f} | "
                  f"{improvement:>10.1f}% | "
                  f"{stats['num_measurements']:>6}")
        else:
            print(f"{name:<25} | {'N/A':<13} | {'N/A':<13} | {'N/A':<12} | {'N/A':<8}")

    print("="*110)
    print()

def main():
    """Main analysis function."""
    print("Loading experiment data...")

    experiments_data = {}
    for name, config in EXPERIMENTS.items():
        print(f"  - {name}...")
        df = load_experiment_data(config)
        stats = calculate_statistics(df)
        experiments_data[name] = {'df': df, 'stats': stats}

    # Print summary table
    print_summary_table(experiments_data)

    # Create paired comparison plot
    output_path = Path(__file__).parent.parent / "tsfm/results/ntp_correction_experiment/25min_config_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_comparison(experiments_data, output_path)

    # Create unified comparison plot with single system clock baseline
    unified_output_path = Path(__file__).parent.parent / "tsfm/results/ntp_correction_experiment/25min_unified_comparison.png"
    plot_unified_comparison(experiments_data, unified_output_path)

    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
