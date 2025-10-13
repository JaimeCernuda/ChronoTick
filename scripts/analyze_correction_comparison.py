#!/usr/bin/env python3
"""
Analyze and compare results from all 4 NTP correction methods.

Generates comparison plots showing:
1. Accuracy vs NTP ground truth over time
2. Method performance summary
3. Statistical analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import glob

# Find all test result CSVs
results_dir = Path('results/ntp_correction_experiment')
csv_files = {
    'none': None,
    'linear': None,
    'drift_aware': None,
    'advanced': None
}

# Find most recent CSV for each method
for method in csv_files.keys():
    pattern = f"ntp_correction_{method}_test_*.csv"
    files = sorted(results_dir.glob(pattern))
    if files:
        csv_files[method] = files[-1]  # Most recent
        print(f"Found {method}: {csv_files[method].name}")

# Load data for each method
data = {}
for method, csv_file in csv_files.items():
    if csv_file and csv_file.exists():
        df = pd.read_csv(csv_file)
        # Filter rows with NTP ground truth
        df_with_ntp = df[df['has_ntp'] == True].copy()
        data[method] = {
            'full': df,
            'with_ntp': df_with_ntp
        }
        print(f"  {method}: {len(df)} total samples, {len(df_with_ntp)} with NTP ground truth")

if not data:
    print("ERROR: No test result files found!")
    print(f"Looking in: {results_dir}")
    exit(1)

print(f"\n{'='*80}")
print("ACCURACY COMPARISON VS NTP GROUND TRUTH")
print(f"{'='*80}\n")

# Calculate MAE for each method
results_summary = []
for method in ['none', 'linear', 'drift_aware', 'advanced']:
    if method not in data:
        continue

    df_ntp = data[method]['with_ntp']
    if len(df_ntp) == 0:
        print(f"{method.upper()}: No NTP ground truth data")
        continue

    chronotick_mae = df_ntp['chronotick_error_ms'].mean()
    system_mae = df_ntp['system_error_ms'].mean()
    improvement = ((system_mae - chronotick_mae) / system_mae * 100)

    results_summary.append({
        'method': method,
        'chronotick_mae': chronotick_mae,
        'system_mae': system_mae,
        'improvement': improvement,
        'samples': len(df_ntp)
    })

    print(f"{method.upper()}:")
    print(f"  ChronoTick MAE: {chronotick_mae:.3f}ms")
    print(f"  System MAE: {system_mae:.3f}ms")
    print(f"  Improvement: {improvement:+.1f}%")
    print(f"  Samples with NTP: {len(df_ntp)}")
    print()

# Create comparison visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('NTP Correction Methods: Real-World 5-Minute Test Comparison',
             fontsize=14, fontweight='bold')

colors = {'none': 'blue', 'linear': 'green', 'drift_aware': 'orange', 'advanced': 'red'}

# Plot 1: Error vs Time for each method
for idx, method in enumerate(['none', 'linear', 'drift_aware', 'advanced']):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]

    if method not in data:
        ax.text(0.5, 0.5, f'No data for {method}',
               ha='center', va='center', transform=ax.transAxes)
        continue

    df = data[method]['with_ntp']
    if len(df) == 0:
        ax.text(0.5, 0.5, f'{method}: No NTP ground truth',
               ha='center', va='center', transform=ax.transAxes)
        continue

    ax.plot(df['elapsed_seconds'], df['chronotick_error_ms'],
           color=colors[method], linewidth=2, label='ChronoTick error', alpha=0.8)
    ax.plot(df['elapsed_seconds'], df['system_error_ms'],
           color='gray', linewidth=1, linestyle='--', label='System error', alpha=0.5)

    ax.axhline(y=df['chronotick_error_ms'].mean(), color=colors[method],
              linestyle=':', linewidth=1, alpha=0.5)

    ax.set_xlabel('Elapsed Time (seconds)', fontsize=10)
    ax.set_ylabel('Error vs NTP (ms)', fontsize=10)
    ax.set_title(f'{method.upper()}: MAE={df["chronotick_error_ms"].mean():.2f}ms',
                fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()

output_dir = Path('results/ntp_correction_experiment/analysis')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'correction_methods_comparison_5min.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved comparison plot: {output_path}")
plt.close()

# Create summary bar chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('NTP Correction Methods: Performance Summary',
             fontsize=14, fontweight='bold')

if results_summary:
    df_summary = pd.DataFrame(results_summary)

    # Plot 1: MAE comparison
    x = np.arange(len(df_summary))
    width = 0.35

    ax1.bar(x - width/2, df_summary['chronotick_mae'], width,
           label='ChronoTick MAE', color='steelblue', alpha=0.8)
    ax1.bar(x + width/2, df_summary['system_mae'], width,
           label='System MAE', color='gray', alpha=0.5)

    ax1.set_xlabel('Correction Method')
    ax1.set_ylabel('Mean Absolute Error (ms)')
    ax1.set_title('Accuracy vs NTP Ground Truth')
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_summary['method'].str.upper())
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Improvement percentage
    colors_list = [colors[m] for m in df_summary['method']]
    bars = ax2.bar(x, df_summary['improvement'], color=colors_list, alpha=0.8)

    # Color bars: green for positive, red for negative
    for bar, improvement in zip(bars, df_summary['improvement']):
        if improvement < 0:
            bar.set_color('red')
        else:
            bar.set_color('green')

    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Correction Method')
    ax2.set_ylabel('Improvement over System Clock (%)')
    ax2.set_title('Performance Improvement')
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_summary['method'].str.upper())
    ax2.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, df_summary['improvement'])):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:+.1f}%', ha='center', va='bottom' if height > 0 else 'top')

plt.tight_layout()
output_path = output_dir / 'correction_methods_summary_5min.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved summary plot: {output_path}")
plt.close()

print(f"\n{'='*80}")
print("HYPOTHESIS VERIFICATION")
print(f"{'='*80}\n")

# Check if none ≈ advanced
if 'none' in data and 'advanced' in data:
    none_mae = results_summary[0]['chronotick_mae'] if results_summary[0]['method'] == 'none' else None
    advanced_mae = next((r['chronotick_mae'] for r in results_summary if r['method'] == 'advanced'), None)

    if none_mae and advanced_mae:
        diff_pct = abs(none_mae - advanced_mae) / max(none_mae, advanced_mae) * 100
        print(f"NONE vs ADVANCED:")
        print(f"  NONE MAE: {none_mae:.3f}ms")
        print(f"  ADVANCED MAE: {advanced_mae:.3f}ms")
        print(f"  Difference: {diff_pct:.1f}%")
        print(f"  Similar? {'YES' if diff_pct < 10 else 'NO'}")
        print()

# Check if linear ≈ drift_aware
if 'linear' in data and 'drift_aware' in data:
    linear_mae = next((r['chronotick_mae'] for r in results_summary if r['method'] == 'linear'), None)
    drift_mae = next((r['chronotick_mae'] for r in results_summary if r['method'] == 'drift_aware'), None)

    if linear_mae and drift_mae:
        diff_pct = abs(linear_mae - drift_mae) / max(linear_mae, drift_mae) * 100
        print(f"LINEAR vs DRIFT_AWARE:")
        print(f"  LINEAR MAE: {linear_mae:.3f}ms")
        print(f"  DRIFT_AWARE MAE: {drift_mae:.3f}ms")
        print(f"  Difference: {diff_pct:.1f}%")
        print(f"  Similar? {'YES' if diff_pct < 10 else 'NO'}")
        print()

print(f"{'='*80}")
print("Analysis complete!")
print(f"Results saved in: {output_dir}")
