#!/usr/bin/env python3
"""
Re-analyze the 25-minute NTP correction method comparison.
This time focusing on ACCURACY vs NTP instead of STABILITY.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_and_analyze(csv_path, method_name):
    """Load CSV and calculate accuracy metrics vs NTP."""
    df = pd.read_csv(csv_path)

    # Filter to only NTP measurement points
    ntp_valid = df['has_ntp'] == True
    df_ntp = df[ntp_valid].copy()

    if len(df_ntp) == 0:
        return None

    # Calculate absolute errors vs NTP
    df_ntp['chronotick_abs_error'] = df_ntp['chronotick_error_ms'].abs()
    df_ntp['system_abs_error'] = df_ntp['system_error_ms'].abs()

    metrics = {
        'method': method_name,
        'ntp_samples': len(df_ntp),

        # Accuracy metrics (vs NTP ground truth)
        'chronotick_mae': df_ntp['chronotick_abs_error'].mean(),
        'chronotick_rms': np.sqrt((df_ntp['chronotick_error_ms'] ** 2).mean()),
        'chronotick_mean_error': df_ntp['chronotick_error_ms'].mean(),
        'chronotick_std_error': df_ntp['chronotick_error_ms'].std(),

        'system_mae': df_ntp['system_abs_error'].mean(),
        'system_rms': np.sqrt((df_ntp['system_error_ms'] ** 2).mean()),

        # Stability metrics (internal consistency)
        'chronotick_offset_mean': df['chronotick_offset_ms'].mean(),
        'chronotick_offset_std': df['chronotick_offset_ms'].std(),
        'chronotick_offset_range': df['chronotick_offset_ms'].max() - df['chronotick_offset_ms'].min(),
    }

    # Calculate improvement over system clock
    metrics['mae_improvement_pct'] = (
        (metrics['system_mae'] - metrics['chronotick_mae']) / metrics['system_mae'] * 100
    )
    metrics['rms_improvement_pct'] = (
        (metrics['system_rms'] - metrics['chronotick_rms']) / metrics['system_rms'] * 100
    )

    # Calculate stability score (previous metric)
    metrics['stability_score'] = (
        abs(metrics['chronotick_offset_mean']) +
        metrics['chronotick_offset_std'] +
        metrics['chronotick_offset_range'] / 10
    )

    return metrics, df, df_ntp

def create_comparison_visualization(all_metrics, all_data, output_dir):
    """Create comprehensive comparison visualizations."""
    output_dir = Path(output_dir)

    methods = [m['method'] for m in all_metrics]
    colors = {'none': 'blue', 'linear': 'green', 'drift_aware': 'orange', 'advanced': 'red'}

    # Create comprehensive figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('NTP Correction Methods: Accuracy vs NTP Ground Truth (25-minute tests)',
                 fontsize=14, fontweight='bold')

    # 1. Mean Absolute Error (MAE) - PRIMARY ACCURACY METRIC
    ax = axes[0, 0]
    chronotick_mae = [m['chronotick_mae'] for m in all_metrics]
    system_mae = [m['system_mae'] for m in all_metrics]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax.bar(x - width/2, chronotick_mae, width, label='ChronoTick',
                   color=[colors[m] for m in methods], alpha=0.8)
    bars2 = ax.bar(x + width/2, system_mae, width, label='System Clock',
                   color='gray', alpha=0.5)

    ax.set_ylabel('Mean Absolute Error (ms)', fontsize=11)
    ax.set_title('Accuracy: MAE vs NTP (Lower is Better)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)

    # 2. RMS Error
    ax = axes[0, 1]
    chronotick_rms = [m['chronotick_rms'] for m in all_metrics]
    system_rms = [m['system_rms'] for m in all_metrics]

    bars1 = ax.bar(x - width/2, chronotick_rms, width, label='ChronoTick',
                   color=[colors[m] for m in methods], alpha=0.8)
    bars2 = ax.bar(x + width/2, system_rms, width, label='System Clock',
                   color='gray', alpha=0.5)

    ax.set_ylabel('RMS Error (ms)', fontsize=11)
    ax.set_title('Accuracy: RMS Error vs NTP (Lower is Better)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)

    # 3. Improvement over System Clock
    ax = axes[0, 2]
    mae_improvements = [m['mae_improvement_pct'] for m in all_metrics]
    rms_improvements = [m['rms_improvement_pct'] for m in all_metrics]

    bars1 = ax.bar(x - width/2, mae_improvements, width, label='MAE Improvement',
                   color=[colors[m] for m in methods], alpha=0.8)
    bars2 = ax.bar(x + width/2, rms_improvements, width, label='RMS Improvement',
                   color=[colors[m] for m in methods], alpha=0.5)

    ax.axhline(y=0, color='k', linestyle='--', linewidth=1.5, alpha=0.5)
    ax.set_ylabel('Improvement over System Clock (%)', fontsize=11)
    ax.set_title('ChronoTick vs System Clock (Higher is Better)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Shade positive/negative
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                bar.set_facecolor('green')
                bar.set_alpha(0.7)
            else:
                bar.set_facecolor('red')
                bar.set_alpha(0.7)
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

    # 4. Stability Score (previous metric for comparison)
    ax = axes[1, 0]
    stability_scores = [m['stability_score'] for m in all_metrics]

    bars = ax.bar(methods, stability_scores, color=[colors[m] for m in methods], alpha=0.8)
    ax.set_ylabel('Stability Score', fontsize=11)
    ax.set_title('Stability: Internal Consistency (Lower is Better)', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.1f}',
               ha='center', va='bottom', fontsize=9)

    # 5. Error vs time for each method
    ax = axes[1, 1]
    for method, (metrics, df, df_ntp) in all_data.items():
        if df_ntp is not None and len(df_ntp) > 0:
            ax.plot(df_ntp['elapsed_seconds'] / 60, df_ntp['chronotick_error_ms'],
                   'o-', label=method, color=colors[method], alpha=0.7, markersize=4)

    ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('Elapsed Time (minutes)', fontsize=11)
    ax.set_ylabel('ChronoTick Error vs NTP (ms)', fontsize=11)
    ax.set_title('Error Evolution Over 25 Minutes', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Summary table
    ax = axes[1, 2]
    ax.axis('off')

    # Find best method for each metric
    best_mae_idx = np.argmin([m['chronotick_mae'] for m in all_metrics])
    best_rms_idx = np.argmin([m['chronotick_rms'] for m in all_metrics])
    best_stability_idx = np.argmin([m['stability_score'] for m in all_metrics])

    summary_text = "SUMMARY: Best Methods\n"
    summary_text += "=" * 40 + "\n\n"
    summary_text += f"ACCURACY (vs NTP):\n"
    summary_text += f"  Best MAE: {all_metrics[best_mae_idx]['method']}\n"
    summary_text += f"    {all_metrics[best_mae_idx]['chronotick_mae']:.2f} ms\n"
    summary_text += f"    ({all_metrics[best_mae_idx]['mae_improvement_pct']:.1f}% vs system)\n\n"
    summary_text += f"  Best RMS: {all_metrics[best_rms_idx]['method']}\n"
    summary_text += f"    {all_metrics[best_rms_idx]['chronotick_rms']:.2f} ms\n"
    summary_text += f"    ({all_metrics[best_rms_idx]['rms_improvement_pct']:.1f}% vs system)\n\n"
    summary_text += f"STABILITY (internal):\n"
    summary_text += f"  Best: {all_metrics[best_stability_idx]['method']}\n"
    summary_text += f"    {all_metrics[best_stability_idx]['stability_score']:.2f}\n\n"
    summary_text += "=" * 40 + "\n"
    summary_text += "RECOMMENDATION:\n"
    summary_text += f"  Use '{all_metrics[best_mae_idx]['method']}' method\n"
    summary_text += f"  (best accuracy vs ground truth)"

    ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison_25min.png', dpi=150, bbox_inches='tight')
    print(f"Saved: {output_dir / 'accuracy_comparison_25min.png'}")

    plt.close('all')

def main():
    # Check if we're in tsfm/ directory or project root
    if Path('results/ntp_correction_experiment').exists():
        results_dir = Path('results/ntp_correction_experiment')
    elif Path('../results/ntp_correction_experiment').exists():
        results_dir = Path('../results/ntp_correction_experiment')
    else:
        results_dir = Path('results/ntp_correction_experiment')

    methods = {
        'none': results_dir / 'ntp_correction_none.csv',
        'linear': results_dir / 'ntp_correction_linear.csv',
        'drift_aware': results_dir / 'ntp_correction_drift_aware.csv',
        'advanced': results_dir / 'ntp_correction_advanced.csv',
    }

    print("="*70)
    print("RE-ANALYSIS: NTP Correction Methods - ACCURACY vs NTP Ground Truth")
    print("="*70)
    print("\nLoading and analyzing 25-minute test results...")

    all_metrics = []
    all_data = {}

    for method, csv_path in methods.items():
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found, skipping {method}")
            continue

        result = load_and_analyze(csv_path, method)
        if result is None:
            print(f"WARNING: No NTP data in {method}, skipping")
            continue

        metrics, df, df_ntp = result
        all_metrics.append(metrics)
        all_data[method] = (metrics, df, df_ntp)

        print(f"\n{method.upper()}:")
        print(f"  NTP samples: {metrics['ntp_samples']}")
        print(f"  ChronoTick MAE: {metrics['chronotick_mae']:.2f} ms")
        print(f"  System Clock MAE: {metrics['system_mae']:.2f} ms")
        print(f"  Improvement: {metrics['mae_improvement_pct']:.1f}%")
        print(f"  ChronoTick RMS: {metrics['chronotick_rms']:.2f} ms")
        print(f"  System Clock RMS: {metrics['system_rms']:.2f} ms")
        print(f"  Stability Score: {metrics['stability_score']:.2f} (previous metric)")

    if not all_metrics:
        print("ERROR: No valid data found!")
        return

    # Find best method by accuracy
    best_mae = min(all_metrics, key=lambda m: m['chronotick_mae'])
    best_rms = min(all_metrics, key=lambda m: m['chronotick_rms'])
    best_stability = min(all_metrics, key=lambda m: m['stability_score'])

    print("\n" + "="*70)
    print("COMPARISON: Accuracy vs Stability Ranking")
    print("="*70)

    print("\nACCURACY RANKING (Best to Worst - MAE):")
    sorted_by_mae = sorted(all_metrics, key=lambda m: m['chronotick_mae'])
    for i, m in enumerate(sorted_by_mae, 1):
        print(f"  {i}. {m['method']:12s} - MAE: {m['chronotick_mae']:6.2f} ms "
              f"({m['mae_improvement_pct']:+6.1f}% vs system)")

    print("\nSTABILITY RANKING (Best to Worst):")
    sorted_by_stability = sorted(all_metrics, key=lambda m: m['stability_score'])
    for i, m in enumerate(sorted_by_stability, 1):
        print(f"  {i}. {m['method']:12s} - Score: {m['stability_score']:6.2f}")

    print("\n" + "="*70)
    print("CONCLUSIONS:")
    print("="*70)
    print(f"\nBest for ACCURACY (what matters): {best_mae['method']}")
    print(f"  MAE: {best_mae['chronotick_mae']:.2f} ms ({best_mae['mae_improvement_pct']:+.1f}% vs system)")
    print(f"  RMS: {best_mae['chronotick_rms']:.2f} ms")

    print(f"\nBest for STABILITY (previous metric): {best_stability['method']}")
    print(f"  Stability Score: {best_stability['stability_score']:.2f}")
    print(f"  But MAE: {best_stability['chronotick_mae']:.2f} ms ({best_stability['mae_improvement_pct']:+.1f}% vs system)")

    if best_mae['method'] != best_stability['method']:
        print(f"\n⚠️  IMPORTANT: Best accuracy method ('{best_mae['method']}') is DIFFERENT")
        print(f"             from best stability method ('{best_stability['method']}')")
        print(f"\n   RECOMMENDATION: Use '{best_mae['method']}' for long-term tests")
        print(f"                   (optimizing for accuracy vs ground truth)")

    # Create visualizations
    print("\nGenerating comparison visualizations...")
    create_comparison_visualization(all_metrics, all_data, results_dir)

    # Save metrics to file
    metrics_file = results_dir / 'accuracy_comparison_metrics.txt'
    with open(metrics_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("NTP Correction Methods - ACCURACY ANALYSIS\n")
        f.write("="*70 + "\n\n")

        f.write("ACCURACY RANKING (MAE):\n")
        for i, m in enumerate(sorted_by_mae, 1):
            f.write(f"  {i}. {m['method']:12s} - {m['chronotick_mae']:6.2f} ms "
                   f"({m['mae_improvement_pct']:+6.1f}% vs system)\n")

        f.write(f"\nRECOMMENDATION: Use '{best_mae['method']}' method\n")
        f.write(f"  (best accuracy: {best_mae['chronotick_mae']:.2f} ms MAE)\n")

    print(f"Saved metrics: {metrics_file}")
    print("\nAnalysis complete!")

if __name__ == '__main__':
    main()
