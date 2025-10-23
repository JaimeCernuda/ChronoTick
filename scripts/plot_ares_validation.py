#!/usr/bin/env python3
"""
Plot ChronoTick vs System Clock accuracy for ARES nodes from validation data
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

print("=" * 80)
print("ARES CHRONOTICK FIX VALIDATION ANALYSIS")
print("=" * 80)
print()

# Load data for both ARES nodes
csv_path_11 = Path("results/fix_partial/ares_comp11_validation.csv")
csv_path_12 = Path("results/fix_partial/ares_comp12_validation.csv")

df11 = pd.read_csv(csv_path_11)
df12 = pd.read_csv(csv_path_12)

print(f"ARES-COMP-11:")
print(f"  Total samples: {len(df11)}")
print(f"  Duration: {df11['elapsed_seconds'].max() / 3600:.2f} hours")

print(f"\nARES-COMP-12:")
print(f"  Total samples: {len(df12)}")
print(f"  Duration: {df12['elapsed_seconds'].max() / 3600:.2f} hours")
print()

# Process both datasets
results = {}
for name, df in [("ares-comp-11", df11), ("ares-comp-12", df12)]:
    print(f"\n{'=' * 80}")
    print(f"ANALYSIS: {name.upper()}")
    print(f"{'=' * 80}\n")

    # Filter out error rows
    df_valid = df[df['chronotick_source'] != 'error'].copy()
    df_ntp = df_valid[df_valid['has_ntp'] == True].copy()

    print(f"Valid samples: {len(df_valid)}")
    print(f"NTP ground truth samples: {len(df_ntp)}")

    if len(df_ntp) > 0:
        # Calculate errors against NTP
        df_ntp['chronotick_error'] = abs(df_ntp['chronotick_offset_ms'] - df_ntp['ntp_offset_ms'])
        df_ntp['system_error'] = abs(df_ntp['ntp_offset_ms'])

        # Statistics
        chronotick_mae = df_ntp['chronotick_error'].mean()
        system_mae = df_ntp['system_error'].mean()
        improvement_factor = system_mae / chronotick_mae if chronotick_mae > 0 else float('inf')

        print(f"\nACCURACY METRICS (vs NTP ground truth)")
        print("-" * 80)
        print(f"ChronoTick Mean Absolute Error: {chronotick_mae:.3f} ms")
        print(f"System Clock Mean Absolute Error: {system_mae:.3f} ms")
        print(f"Improvement Factor: {improvement_factor:.2f}x")
        print()
        print(f"ChronoTick Median Error: {df_ntp['chronotick_error'].median():.3f} ms")
        print(f"System Clock Median Error: {df_ntp['system_error'].median():.3f} ms")
        print()
        print(f"ChronoTick 95th Percentile: {df_ntp['chronotick_error'].quantile(0.95):.3f} ms")
        print(f"System Clock 95th Percentile: {df_ntp['system_error'].quantile(0.95):.3f} ms")

        # Time-segmented analysis
        early = df_ntp[df_ntp['elapsed_seconds'] < 600]
        middle = df_ntp[(df_ntp['elapsed_seconds'] >= 3600) & (df_ntp['elapsed_seconds'] < 7200)]
        late = df_ntp[df_ntp['elapsed_seconds'] >= 14400]

        print(f"\nTIME-SEGMENTED ANALYSIS:")
        print("-" * 80)
        if len(early) > 0:
            early_ct_mae = abs(early['chronotick_offset_ms'] - early['ntp_offset_ms']).mean()
            early_sys_mae = abs(early['ntp_offset_ms']).mean()
            print(f"Early (<10min): ChronoTick={early_ct_mae:.3f}ms, System={early_sys_mae:.3f}ms, Improvement={early_sys_mae/early_ct_mae:.2f}x")
        if len(middle) > 0:
            mid_ct_mae = abs(middle['chronotick_offset_ms'] - middle['ntp_offset_ms']).mean()
            mid_sys_mae = abs(middle['ntp_offset_ms']).mean()
            print(f"Middle (1-2hr): ChronoTick={mid_ct_mae:.3f}ms, System={mid_sys_mae:.3f}ms, Improvement={mid_sys_mae/mid_ct_mae:.2f}x")
        if len(late) > 0:
            late_ct_mae = abs(late['chronotick_offset_ms'] - late['ntp_offset_ms']).mean()
            late_sys_mae = abs(late['ntp_offset_ms']).mean()
            print(f"Late (>4hr): ChronoTick={late_ct_mae:.3f}ms, System={late_sys_mae:.3f}ms, Improvement={late_sys_mae/late_ct_mae:.2f}x")

        # NTP acceptance
        ntp_count = len(df_ntp)
        expected_ntp = int(df['elapsed_seconds'].max() / 120)
        warmup_ntp = 27
        post_warmup_ntp = max(0, ntp_count - warmup_ntp)
        post_warmup_expected = expected_ntp
        acceptance_rate = (post_warmup_ntp / post_warmup_expected) * 100 if post_warmup_expected > 0 else 0

        print(f"\nNTP ACCEPTANCE:")
        print("-" * 80)
        print(f"Total NTP measurements: {ntp_count}")
        print(f"Expected (warmup + operational): ~{warmup_ntp + expected_ntp}")
        print(f"Post-warmup acceptance rate: {post_warmup_ntp}/{post_warmup_expected} ({acceptance_rate:.1f}%)")

        # Offset stability
        ntp_offset_std = df_ntp['ntp_offset_ms'].std()
        chronotick_offset_std = df_valid['chronotick_offset_ms'].std()
        print(f"\nOFFSET STABILITY (std dev):")
        print(f"  NTP measurements: {ntp_offset_std:.3f} ms")
        print(f"  ChronoTick predictions: {chronotick_offset_std:.3f} ms")

        # Store results
        results[name] = {
            'df': df,
            'df_valid': df_valid,
            'df_ntp': df_ntp,
            'chronotick_mae': chronotick_mae,
            'system_mae': system_mae,
            'improvement': improvement_factor,
            'acceptance_rate': acceptance_rate
        }
    else:
        print("\n⚠️ No NTP data available for analysis")
        results[name] = {'df': df, 'df_valid': df_valid, 'df_ntp': df_ntp}

# Create comparison visualization
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

fig.suptitle('ARES Nodes - ChronoTick Fix Validation (NTP Proxy)', fontsize=16, fontweight='bold')

# Row 1: ares-comp-11 analysis
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2])

# Row 2: ares-comp-12 analysis
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[1, 2])

# Row 3: Comparison
ax7 = fig.add_subplot(gs[2, 0])
ax8 = fig.add_subplot(gs[2, 1])
ax9 = fig.add_subplot(gs[2, 2])

# Plot ares-comp-11
if 'ares-comp-11' in results and len(results['ares-comp-11']['df_ntp']) > 0:
    r = results['ares-comp-11']

    # Time series
    ax1.plot(r['df_valid']['elapsed_seconds'] / 60, r['df_valid']['chronotick_offset_ms'],
             'b-', alpha=0.3, linewidth=0.5, label='ChronoTick')
    ax1.scatter(r['df_ntp']['elapsed_seconds'] / 60, r['df_ntp']['ntp_offset_ms'],
                c='red', s=30, alpha=0.7, marker='x', label='NTP', zorder=5)
    ax1.set_xlabel('Time (minutes)')
    ax1.set_ylabel('Offset (ms)')
    ax1.set_title('ares-comp-11: Offsets Over Time', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

    # Error box plot
    box_data = [r['df_ntp']['system_error'], r['df_ntp']['chronotick_error']]
    bp = ax2.boxplot(box_data, tick_labels=['System', 'ChronoTick'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightblue')
    ax2.set_ylabel('Absolute Error (ms)')
    ax2.set_title(f'ares-comp-11: Error (Improvement: {r["improvement"]:.2f}x)', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Error time series
    ax3.plot(r['df_ntp']['elapsed_seconds'] / 60, r['df_ntp']['system_error'],
             'r-', marker='o', markersize=3, label='System', alpha=0.7)
    ax3.plot(r['df_ntp']['elapsed_seconds'] / 60, r['df_ntp']['chronotick_error'],
             'b-', marker='s', markersize=3, label='ChronoTick', alpha=0.7)
    ax3.set_xlabel('Time (minutes)')
    ax3.set_ylabel('Absolute Error (ms)')
    ax3.set_title('ares-comp-11: Accuracy Over Time', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

# Plot ares-comp-12
if 'ares-comp-12' in results and len(results['ares-comp-12']['df_ntp']) > 0:
    r = results['ares-comp-12']

    # Time series
    ax4.plot(r['df_valid']['elapsed_seconds'] / 60, r['df_valid']['chronotick_offset_ms'],
             'b-', alpha=0.3, linewidth=0.5, label='ChronoTick')
    ax4.scatter(r['df_ntp']['elapsed_seconds'] / 60, r['df_ntp']['ntp_offset_ms'],
                c='red', s=30, alpha=0.7, marker='x', label='NTP', zorder=5)
    ax4.set_xlabel('Time (minutes)')
    ax4.set_ylabel('Offset (ms)')
    ax4.set_title('ares-comp-12: Offsets Over Time', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.axhline(y=0, color='k', linestyle='--', linewidth=0.5)

    # Error box plot
    box_data = [r['df_ntp']['system_error'], r['df_ntp']['chronotick_error']]
    bp = ax5.boxplot(box_data, tick_labels=['System', 'ChronoTick'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightblue')
    ax5.set_ylabel('Absolute Error (ms)')
    ax5.set_title(f'ares-comp-12: Error (Improvement: {r["improvement"]:.2f}x)', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')

    # Error time series
    ax6.plot(r['df_ntp']['elapsed_seconds'] / 60, r['df_ntp']['system_error'],
             'r-', marker='o', markersize=3, label='System', alpha=0.7)
    ax6.plot(r['df_ntp']['elapsed_seconds'] / 60, r['df_ntp']['chronotick_error'],
             'b-', marker='s', markersize=3, label='ChronoTick', alpha=0.7)
    ax6.set_xlabel('Time (minutes)')
    ax6.set_ylabel('Absolute Error (ms)')
    ax6.set_title('ares-comp-12: Accuracy Over Time', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

# Comparison plots
if len(results) == 2:
    r11 = results['ares-comp-11']
    r12 = results['ares-comp-12']

    # MAE comparison
    categories = ['System Clock', 'ChronoTick']
    comp11_values = [r11['system_mae'], r11['chronotick_mae']]
    comp12_values = [r12['system_mae'], r12['chronotick_mae']]

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax7.bar(x - width/2, comp11_values, width, label='ares-comp-11', color='steelblue')
    bars2 = ax7.bar(x + width/2, comp12_values, width, label='ares-comp-12', color='darkorange')

    ax7.set_ylabel('Mean Absolute Error (ms)')
    ax7.set_title('MAE Comparison Across Nodes', fontweight='bold')
    ax7.set_xticks(x)
    ax7.set_xticklabels(categories)
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax7.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}',
                    ha='center', va='bottom', fontsize=9)

    # Improvement factor comparison
    nodes = ['ares-comp-11', 'ares-comp-12']
    improvements = [r11['improvement'], r12['improvement']]
    acceptances = [r11['acceptance_rate'], r12['acceptance_rate']]

    bars = ax8.bar(nodes, improvements, color=['steelblue', 'darkorange'])
    ax8.set_ylabel('Improvement Factor (x)')
    ax8.set_title('ChronoTick Improvement Factor', fontweight='bold')
    ax8.axhline(y=1, color='red', linestyle='--', linewidth=1, label='Baseline (1x)')
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax8.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}x',
                ha='center', va='bottom', fontweight='bold')

    # NTP acceptance rates
    bars = ax9.bar(nodes, acceptances, color=['steelblue', 'darkorange'])
    ax9.set_ylabel('NTP Acceptance Rate (%)')
    ax9.set_title('Post-Warmup NTP Acceptance', fontweight='bold')
    ax9.axhline(y=100, color='green', linestyle='--', linewidth=1, label='Perfect (100%)')
    ax9.set_ylim([0, 110])
    ax9.legend()
    ax9.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax9.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontweight='bold')

plot_path = Path("results/fix_partial/ares_accuracy_analysis.png")
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print()
print(f"\n✓ Plot saved to: {plot_path}")

print()
print("=" * 80)
print("ARES ANALYSIS COMPLETE")
print("=" * 80)
