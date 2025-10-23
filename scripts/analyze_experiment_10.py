#!/usr/bin/env python3
"""
Analyze Experiment-10 Results: Parallel NTP Testing

Generates three analysis plots for each platform:
1. Offset comparison: ChronoTick vs NTP offset over time
2. Cumulative error: Accumulated absolute error comparison
3. Sigma coverage: % of NTP points within 1σ, 2σ, 3σ bounds
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def analyze_platform(csv_path: Path, platform_name: str, output_dir: Path):
    """Analyze data from one platform and generate plots."""

    # Load data
    print(f"\n{'='*60}")
    print(f"Analyzing: {platform_name}")
    print(f"{'='*60}")

    df = pd.read_csv(csv_path)
    print(f"Total samples: {len(df)}")

    # Filter to rows with NTP measurements
    ntp_df = df[df['has_ntp'] == True].copy()
    print(f"NTP samples: {len(ntp_df)}")

    if len(ntp_df) == 0:
        print(f"⚠️  No NTP measurements found for {platform_name}")
        return

    # Convert elapsed_seconds to hours for readability
    df['elapsed_hours'] = df['elapsed_seconds'] / 3600
    ntp_df['elapsed_hours'] = ntp_df['elapsed_seconds'] / 3600

    # Calculate errors (difference from NTP ground truth)
    ntp_df['chronotick_error'] = abs(ntp_df['chronotick_offset_ms'] - ntp_df['ntp_offset_ms'])
    ntp_df['system_error'] = abs(0 - ntp_df['ntp_offset_ms'])  # system clock assumes 0 offset

    # Calculate cumulative errors
    ntp_df['chronotick_cumulative_error'] = ntp_df['chronotick_error'].cumsum()
    ntp_df['system_cumulative_error'] = ntp_df['system_error'].cumsum()

    # Calculate sigma coverage
    ntp_df['within_1sigma'] = ntp_df['chronotick_error'] <= ntp_df['chronotick_uncertainty_ms']
    ntp_df['within_2sigma'] = ntp_df['chronotick_error'] <= (2 * ntp_df['chronotick_uncertainty_ms'])
    ntp_df['within_3sigma'] = ntp_df['chronotick_error'] <= (3 * ntp_df['chronotick_uncertainty_ms'])

    sigma_1 = (ntp_df['within_1sigma'].sum() / len(ntp_df)) * 100
    sigma_2 = (ntp_df['within_2sigma'].sum() / len(ntp_df)) * 100
    sigma_3 = (ntp_df['within_3sigma'].sum() / len(ntp_df)) * 100

    print(f"\nPerformance Metrics:")
    print(f"  ChronoTick Mean Error: {ntp_df['chronotick_error'].mean():.4f} ms")
    print(f"  ChronoTick Median Error: {ntp_df['chronotick_error'].median():.4f} ms")
    print(f"  System Clock Mean Error: {ntp_df['system_error'].mean():.4f} ms")
    print(f"  System Clock Median Error: {ntp_df['system_error'].median():.4f} ms")
    print(f"\nSigma Coverage:")
    print(f"  Within 1σ: {sigma_1:.2f}%")
    print(f"  Within 2σ: {sigma_2:.2f}%")
    print(f"  Within 3σ: {sigma_3:.2f}%")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Offset Comparison
    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot all ChronoTick predictions (including non-NTP intervals)
    ax.plot(df['elapsed_hours'], df['chronotick_offset_ms'],
            'b-', alpha=0.3, linewidth=0.5, label='ChronoTick Predictions')

    # Highlight NTP measurements
    ax.scatter(ntp_df['elapsed_hours'], ntp_df['ntp_offset_ms'],
               c='red', marker='x', s=20, alpha=0.8, label='NTP Ground Truth', zorder=3)

    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Clock Offset (ms)', fontsize=12)
    ax.set_title(f'{platform_name}: Clock Offset Comparison\nChronoTick vs NTP Ground Truth',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{platform_name}_offset_comparison.png', dpi=150)
    print(f"✓ Saved: {output_dir / f'{platform_name}_offset_comparison.png'}")
    plt.close()

    # Plot 2: Cumulative Error
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(ntp_df['elapsed_hours'], ntp_df['chronotick_cumulative_error'],
            'b-', linewidth=2, label='ChronoTick Cumulative Error')
    ax.plot(ntp_df['elapsed_hours'], ntp_df['system_cumulative_error'],
            'r--', linewidth=2, label='System Clock Cumulative Error')

    ax.set_xlabel('Time (hours)', fontsize=12)
    ax.set_ylabel('Cumulative Absolute Error (ms)', fontsize=12)
    ax.set_title(f'{platform_name}: Accumulated Error Over Time',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f'{platform_name}_cumulative_error.png', dpi=150)
    print(f"✓ Saved: {output_dir / f'{platform_name}_cumulative_error.png'}")
    plt.close()

    # Plot 3: Sigma Coverage
    fig, ax = plt.subplots(figsize=(10, 6))

    sigma_levels = ['1σ', '2σ', '3σ']
    coverage_values = [sigma_1, sigma_2, sigma_3]
    colors = ['#2E86AB', '#A23B72', '#F18F01']

    bars = ax.bar(sigma_levels, coverage_values, color=colors, alpha=0.8, edgecolor='black')

    # Add percentage labels on bars
    for bar, value in zip(bars, coverage_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{value:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Add reference line at 68.3% (expected for 1σ)
    ax.axhline(y=68.3, color='gray', linestyle='--', alpha=0.5, label='Expected 1σ (68.3%)')
    ax.axhline(y=95.4, color='gray', linestyle=':', alpha=0.5, label='Expected 2σ (95.4%)')

    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.set_title(f'{platform_name}: Prediction Confidence Bounds Coverage\n% of NTP Measurements Within σ Bounds',
                 fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / f'{platform_name}_sigma_coverage.png', dpi=150)
    print(f"✓ Saved: {output_dir / f'{platform_name}_sigma_coverage.png'}")
    plt.close()

    return {
        'platform': platform_name,
        'total_samples': len(df),
        'ntp_samples': len(ntp_df),
        'chronotick_mean_error': ntp_df['chronotick_error'].mean(),
        'chronotick_median_error': ntp_df['chronotick_error'].median(),
        'system_mean_error': ntp_df['system_error'].mean(),
        'system_median_error': ntp_df['system_error'].median(),
        'sigma_1': sigma_1,
        'sigma_2': sigma_2,
        'sigma_3': sigma_3,
    }

def main():
    """Analyze all three platforms."""

    base_dir = Path(__file__).parent.parent / "results" / "experiment-10"

    platforms = [
        ("homelab", "Homelab"),
        ("ares-11", "ARES comp-11"),
        ("ares-12", "ARES comp-12"),
    ]

    results = []

    for dir_name, platform_name in platforms:
        csv_path = base_dir / dir_name / "chronotick_client_validation_20251022_192238.csv"
        if dir_name == "ares-11":
            csv_path = base_dir / dir_name / "chronotick_client_validation_20251022_192420.csv"
        elif dir_name == "ares-12":
            csv_path = base_dir / dir_name / "chronotick_client_validation_20251022_192443.csv"

        if not csv_path.exists():
            print(f"⚠️  CSV not found: {csv_path}")
            continue

        output_dir = base_dir / dir_name
        result = analyze_platform(csv_path, platform_name, output_dir)
        if result:
            results.append(result)

    # Print summary table
    print(f"\n{'='*80}")
    print("EXPERIMENT-10 SUMMARY")
    print(f"{'='*80}")
    print(f"{'Platform':<15} {'Samples':<10} {'NTP':<8} {'CT Error (ms)':<15} {'Sys Error (ms)':<15} {'1σ%':<8} {'2σ%':<8} {'3σ%':<8}")
    print("-" * 80)

    for r in results:
        print(f"{r['platform']:<15} {r['total_samples']:<10} {r['ntp_samples']:<8} "
              f"{r['chronotick_mean_error']:<15.4f} {r['system_mean_error']:<15.4f} "
              f"{r['sigma_1']:<8.1f} {r['sigma_2']:<8.1f} {r['sigma_3']:<8.1f}")

    print(f"\n✓ Analysis complete. Plots saved in: {base_dir}")

if __name__ == "__main__":
    main()
