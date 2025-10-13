#!/usr/bin/env python3
"""
Compare Chronos-Bolt vs TimesFM 2.5 accuracy results
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def main():
    # Load both datasets
    chronos_df = pd.read_csv('/tmp/chronotick_chronos_baseline.csv')
    timesfm_df = pd.read_csv('/tmp/chronotick_client_validation.csv')

    # Filter to NTP measurements only
    chronos_ntp = chronos_df[chronos_df['has_ntp'] == True].copy()
    timesfm_ntp = timesfm_df[timesfm_df['has_ntp'] == True].copy()

    # Calculate errors vs NTP ground truth
    chronos_ntp['system_error'] = abs(chronos_ntp['system_time'] - chronos_ntp['ntp_time']) * 1000  # to ms
    chronos_ntp['chronotick_error'] = abs(chronos_ntp['chronotick_time'] - chronos_ntp['ntp_time']) * 1000

    timesfm_ntp['system_error'] = abs(timesfm_ntp['system_time'] - timesfm_ntp['ntp_time']) * 1000
    timesfm_ntp['chronotick_error'] = abs(timesfm_ntp['chronotick_time'] - timesfm_ntp['ntp_time']) * 1000

    # Calculate statistics
    chronos_stats = {
        'system_mean': chronos_ntp['system_error'].mean(),
        'system_std': chronos_ntp['system_error'].std(),
        'chronotick_mean': chronos_ntp['chronotick_error'].mean(),
        'chronotick_std': chronos_ntp['chronotick_error'].std(),
        'win_rate': (chronos_ntp['chronotick_error'] < chronos_ntp['system_error']).sum() / len(chronos_ntp) * 100,
        'improvement': ((chronos_ntp['system_error'].mean() - chronos_ntp['chronotick_error'].mean()) / chronos_ntp['system_error'].mean() * 100)
    }

    timesfm_stats = {
        'system_mean': timesfm_ntp['system_error'].mean(),
        'system_std': timesfm_ntp['system_error'].std(),
        'chronotick_mean': timesfm_ntp['chronotick_error'].mean(),
        'chronotick_std': timesfm_ntp['chronotick_error'].std(),
        'win_rate': (timesfm_ntp['chronotick_error'] < timesfm_ntp['system_error']).sum() / len(timesfm_ntp) * 100,
        'improvement': ((timesfm_ntp['system_error'].mean() - timesfm_ntp['chronotick_error'].mean()) / timesfm_ntp['system_error'].mean() * 100)
    }

    print("=" * 80)
    print("MODEL COMPARISON: Chronos-Bolt vs TimesFM 2.5")
    print("=" * 80)
    print()
    print(f"CHRONOS-BOLT BASELINE:")
    print(f"  System Clock Error: {chronos_stats['system_mean']:.2f} ms ± {chronos_stats['system_std']:.2f} ms")
    print(f"  ChronoTick Error:   {chronos_stats['chronotick_mean']:.2f} ms ± {chronos_stats['chronotick_std']:.2f} ms")
    print(f"  Win Rate:           {chronos_stats['win_rate']:.1f}% ({(chronos_ntp['chronotick_error'] < chronos_ntp['system_error']).sum()}/{len(chronos_ntp)} samples)")
    print(f"  Improvement:        {chronos_stats['improvement']:.1f}%")
    print()
    print(f"TIMESFM 2.5 (Short + Long + Fusion):")
    print(f"  System Clock Error: {timesfm_stats['system_mean']:.2f} ms ± {timesfm_stats['system_std']:.2f} ms")
    print(f"  ChronoTick Error:   {timesfm_stats['chronotick_mean']:.2f} ms ± {timesfm_stats['chronotick_std']:.2f} ms")
    print(f"  Win Rate:           {timesfm_stats['win_rate']:.1f}% ({(timesfm_ntp['chronotick_error'] < timesfm_ntp['system_error']).sum()}/{len(timesfm_ntp)} samples)")
    print(f"  Improvement:        {timesfm_stats['improvement']:.1f}%")
    print()
    print("=" * 80)
    print("COMPARATIVE ANALYSIS")
    print("=" * 80)
    print()

    # Calculate relative improvements
    error_reduction = ((chronos_stats['chronotick_mean'] - timesfm_stats['chronotick_mean']) / chronos_stats['chronotick_mean'] * 100)
    win_rate_gain = timesfm_stats['win_rate'] - chronos_stats['win_rate']

    print(f"TimesFM 2.5 vs Chronos-Bolt:")
    print(f"  ✅ ChronoTick Error Reduction: {error_reduction:.1f}%")
    print(f"     ({chronos_stats['chronotick_mean']:.2f} ms → {timesfm_stats['chronotick_mean']:.2f} ms)")
    print(f"  ✅ Win Rate Gain: +{win_rate_gain:.1f} percentage points")
    print(f"     ({chronos_stats['win_rate']:.1f}% → {timesfm_stats['win_rate']:.1f}%)")
    print(f"  ✅ Accuracy Swing: {chronos_stats['improvement']:.1f}% → {timesfm_stats['improvement']:.1f}%")
    print(f"     (From {abs(chronos_stats['improvement']):.1f}% WORSE to {timesfm_stats['improvement']:.1f}% BETTER)")
    print()

    # Create comparison visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Chronos-Bolt vs TimesFM 2.5: Model Comparison', fontsize=16, fontweight='bold')

    # Plot 1: Mean Error Comparison
    ax = axes[0, 0]
    models = ['Chronos-Bolt', 'TimesFM 2.5']
    system_errors = [chronos_stats['system_mean'], timesfm_stats['system_mean']]
    chronotick_errors = [chronos_stats['chronotick_mean'], timesfm_stats['chronotick_mean']]

    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, system_errors, width, label='System Clock', color='orange', alpha=0.8)
    bars2 = ax.bar(x + width/2, chronotick_errors, width, label='ChronoTick', color='green', alpha=0.8)

    ax.set_ylabel('Mean Absolute Error (ms)', fontweight='bold')
    ax.set_title('Mean Error Comparison (Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom', fontsize=9)

    # Plot 2: Win Rate Comparison
    ax = axes[0, 1]
    win_rates = [chronos_stats['win_rate'], timesfm_stats['win_rate']]
    colors = ['#ff6b6b', '#51cf66']

    bars = ax.bar(models, win_rates, color=colors, alpha=0.8)
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='50% baseline')
    ax.set_ylabel('Win Rate (%)', fontweight='bold')
    ax.set_title('ChronoTick Win Rate vs System Clock')
    ax.set_ylim(0, 110)
    ax.grid(axis='y', alpha=0.3)
    ax.legend()

    # Add value labels
    for i, (bar, rate) in enumerate(zip(bars, win_rates)):
        ax.text(bar.get_x() + bar.get_width()/2., rate + 2,
               f'{rate:.1f}%',
               ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Plot 3: Improvement Percentage
    ax = axes[1, 0]
    improvements = [chronos_stats['improvement'], timesfm_stats['improvement']]
    colors_imp = ['#ff6b6b' if x < 0 else '#51cf66' for x in improvements]

    bars = ax.bar(models, improvements, color=colors_imp, alpha=0.8)
    ax.axhline(y=0, color='black', linewidth=1)
    ax.set_ylabel('Improvement over System Clock (%)', fontweight='bold')
    ax.set_title('ChronoTick Improvement (Positive = Better)')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, imp in zip(bars, improvements):
        y_pos = imp + (2 if imp > 0 else -5)
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
               f'{imp:+.1f}%',
               ha='center', va='bottom' if imp > 0 else 'top',
               fontsize=10, fontweight='bold')

    # Plot 4: Error Evolution Over Time
    ax = axes[1, 1]
    ax.plot(chronos_ntp['elapsed_seconds'], chronos_ntp['chronotick_error'],
            marker='o', label='Chronos-Bolt', color='#ff6b6b', linewidth=2, markersize=6)
    ax.plot(timesfm_ntp['elapsed_seconds'], timesfm_ntp['chronotick_error'],
            marker='s', label='TimesFM 2.5', color='#51cf66', linewidth=2, markersize=6)

    ax.set_xlabel('Time (seconds)', fontweight='bold')
    ax.set_ylabel('ChronoTick Error (ms)', fontweight='bold')
    ax.set_title('ChronoTick Error Over Time (at NTP Measurement Points)')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = '/tmp/chronotick_model_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Comparison visualization saved to: {output_path}")
    print()

    # Generate summary CSV
    summary_data = {
        'Model': ['Chronos-Bolt', 'TimesFM 2.5'],
        'System_Clock_Error_ms': [chronos_stats['system_mean'], timesfm_stats['system_mean']],
        'ChronoTick_Error_ms': [chronos_stats['chronotick_mean'], timesfm_stats['chronotick_mean']],
        'ChronoTick_Error_StdDev_ms': [chronos_stats['chronotick_std'], timesfm_stats['chronotick_std']],
        'Win_Rate_Percent': [chronos_stats['win_rate'], timesfm_stats['win_rate']],
        'Improvement_Percent': [chronos_stats['improvement'], timesfm_stats['improvement']],
    }
    summary_df = pd.DataFrame(summary_data)
    summary_path = '/tmp/chronotick_model_comparison.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"✓ Summary CSV saved to: {summary_path}")
    print()

    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    if timesfm_stats['chronotick_mean'] < chronos_stats['chronotick_mean']:
        print(f"✅ TimesFM 2.5 is the CLEAR WINNER!")
        print(f"   - {error_reduction:.1f}% lower error than Chronos-Bolt")
        print(f"   - {timesfm_stats['win_rate']:.1f}% win rate vs {chronos_stats['win_rate']:.1f}% for Chronos")
        print(f"   - Provides {timesfm_stats['improvement']:.1f}% better accuracy than system clock")
    else:
        print(f"❌ Chronos-Bolt performed better in this comparison")
    print()

if __name__ == '__main__':
    main()
