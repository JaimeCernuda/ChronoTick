#!/usr/bin/env python3
"""
Compare TimesFM 2.5 with and without NTP correction
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load both test results
print("Loading test data...")
print("=" * 80)

# Previous test: TimesFM 2.5 WITHOUT NTP correction
without_ntp = pd.read_csv('/home/jcernuda/tick_project/ChronoTick/tsfm/results/timesfm_2.5_validation/chronotick_client_validation.csv')

# Current test: TimesFM 2.5 WITH NTP correction (but check if it was actually applied)
with_ntp = pd.read_csv('/tmp/chronotick_client_validation.csv')

print(f"WITHOUT NTP Correction test: {len(without_ntp)} samples")
print(f"WITH NTP Correction test: {len(with_ntp)} samples")
print()

# Check what sources were used
print("Source distribution in WITH NTP test:")
print(with_ntp['chronotick_source'].value_counts())
print()

# Filter to NTP measurements only
without_ntp_measurements = without_ntp[without_ntp['has_ntp'] == True].copy()
with_ntp_measurements = with_ntp[with_ntp['has_ntp'] == True].copy()

print(f"NTP measurements WITHOUT correction: {len(without_ntp_measurements)}")
print(f"NTP measurements WITH correction: {len(with_ntp_measurements)}")
print()

# Calculate errors
without_ntp_measurements['system_error'] = abs(without_ntp_measurements['system_time'] - without_ntp_measurements['ntp_time']) * 1000
without_ntp_measurements['chronotick_error'] = abs(without_ntp_measurements['chronotick_time'] - without_ntp_measurements['ntp_time']) * 1000

with_ntp_measurements['system_error'] = abs(with_ntp_measurements['system_time'] - with_ntp_measurements['ntp_time']) * 1000
with_ntp_measurements['chronotick_error'] = abs(with_ntp_measurements['chronotick_time'] - with_ntp_measurements['ntp_time']) * 1000

# Statistics
print("=" * 80)
print("OVERALL COMPARISON")
print("=" * 80)
print()
print(f"WITHOUT NTP Correction:")
print(f"  ChronoTick Error: {without_ntp_measurements['chronotick_error'].mean():.2f} ms ± {without_ntp_measurements['chronotick_error'].std():.2f} ms")
print(f"  System Clock Error: {without_ntp_measurements['system_error'].mean():.2f} ms ± {without_ntp_measurements['system_error'].std():.2f} ms")
print(f"  Win Rate: {(without_ntp_measurements['chronotick_error'] < without_ntp_measurements['system_error']).sum()}/{len(without_ntp_measurements)} ({100*(without_ntp_measurements['chronotick_error'] < without_ntp_measurements['system_error']).sum()/len(without_ntp_measurements):.1f}%)")
print()
print(f"WITH NTP Correction:")
print(f"  ChronoTick Error: {with_ntp_measurements['chronotick_error'].mean():.2f} ms ± {with_ntp_measurements['chronotick_error'].std():.2f} ms")
print(f"  System Clock Error: {with_ntp_measurements['system_error'].mean():.2f} ms ± {with_ntp_measurements['system_error'].std():.2f} ms")
print(f"  Win Rate: {(with_ntp_measurements['chronotick_error'] < with_ntp_measurements['system_error']).sum()}/{len(with_ntp_measurements)} ({100*(with_ntp_measurements['chronotick_error'] < with_ntp_measurements['system_error']).sum()/len(with_ntp_measurements):.1f}%)")
print()

# CRITICAL: Check when NTP correction was actually active
print("=" * 80)
print("NTP CORRECTION ANALYSIS")
print("=" * 80)
print()

# Find samples where NTP correction was used
with_correction = with_ntp[with_ntp['chronotick_source'].str.contains('ntp', na=False)]
without_correction = with_ntp[with_ntp['chronotick_source'] == 'cpu']

print(f"Samples WITH NTP correction active: {len(with_correction)} ({100*len(with_correction)/len(with_ntp):.1f}%)")
print(f"Samples WITHOUT NTP correction: {len(without_correction)} ({100*len(without_correction)/len(with_ntp):.1f}%)")
print()

if len(with_correction) > 0:
    print(f"NTP correction active from: {with_correction['elapsed_seconds'].min():.1f}s to {with_correction['elapsed_seconds'].max():.1f}s")
    print(f"NTP correction stopped at: {with_correction['elapsed_seconds'].max():.1f}s ({with_correction['elapsed_seconds'].max()/60:.1f} minutes)")
print()

# Calculate accuracy during periods when NTP correction was active
with_correction_at_ntp = with_ntp_measurements[with_ntp_measurements['chronotick_source'].str.contains('ntp', na=False)]
without_correction_at_ntp = with_ntp_measurements[with_ntp_measurements['chronotick_source'] == 'cpu']

if len(with_correction_at_ntp) > 0:
    print("=" * 80)
    print("ACCURACY WHEN NTP CORRECTION WAS ACTIVE")
    print("=" * 80)
    print()
    print(f"NTP measurements with correction active: {len(with_correction_at_ntp)}")
    print(f"  ChronoTick Error: {with_correction_at_ntp['chronotick_error'].mean():.2f} ms ± {with_correction_at_ntp['chronotick_error'].std():.2f} ms")
    print(f"  Win Rate: {(with_correction_at_ntp['chronotick_error'] < with_correction_at_ntp['system_error']).sum()}/{len(with_correction_at_ntp)} ({100*(with_correction_at_ntp['chronotick_error'] < with_correction_at_ntp['system_error']).sum()/len(with_correction_at_ntp):.1f}%)")
    print()

if len(without_correction_at_ntp) > 0:
    print("=" * 80)
    print("ACCURACY WHEN NTP CORRECTION STOPPED WORKING")
    print("=" * 80)
    print()
    print(f"NTP measurements WITHOUT correction active: {len(without_correction_at_ntp)}")
    print(f"  ChronoTick Error: {without_correction_at_ntp['chronotick_error'].mean():.2f} ms ± {without_correction_at_ntp['chronotick_error'].std():.2f} ms")
    print(f"  Win Rate: {(without_correction_at_ntp['chronotick_error'] < without_correction_at_ntp['system_error']).sum()}/{len(without_correction_at_ntp)} ({100*(without_correction_at_ntp['chronotick_error'] < without_correction_at_ntp['system_error']).sum()/len(without_correction_at_ntp):.1f}%)")
    print()

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('TimesFM 2.5: With vs Without NTP Correction', fontsize=16, fontweight='bold')

# Plot 1: Error comparison
ax = axes[0, 0]
x = np.arange(2)
width = 0.35

without_errors = [without_ntp_measurements['chronotick_error'].mean()]
with_errors = [with_ntp_measurements['chronotick_error'].mean()]

if len(with_correction_at_ntp) > 0:
    with_active_errors = [with_correction_at_ntp['chronotick_error'].mean()]

    bars1 = ax.bar([0], without_errors, width, label='WITHOUT NTP Correction', color='#4CAF50', alpha=0.8)
    bars2 = ax.bar([1], with_errors, width, label='WITH NTP (overall)', color='#FF9800', alpha=0.8)
    bars3 = ax.bar([2], with_active_errors, width, label='WITH NTP (when active)', color='#2196F3', alpha=0.8)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['WITHOUT\nNTP', 'WITH NTP\n(overall)', 'WITH NTP\n(active only)'])

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}ms',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
else:
    bars1 = ax.bar([0], without_errors, width, label='WITHOUT NTP', color='#4CAF50', alpha=0.8)
    bars2 = ax.bar([1], with_errors, width, label='WITH NTP', color='#FF9800', alpha=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['WITHOUT NTP', 'WITH NTP'])

ax.set_ylabel('Mean ChronoTick Error (ms)', fontweight='bold')
ax.set_title('ChronoTick Error Comparison')
ax.legend()
ax.grid(axis='y', alpha=0.3)

# Plot 2: Error over time for WITH NTP test
ax = axes[0, 1]
ax.plot(with_ntp_measurements['elapsed_seconds'], with_ntp_measurements['chronotick_error'],
        'o-', label='ChronoTick Error', color='#2196F3', markersize=8, linewidth=2)
ax.plot(with_ntp_measurements['elapsed_seconds'], with_ntp_measurements['system_error'],
        's-', label='System Clock Error', color='#FF9800', markersize=6, linewidth=2, alpha=0.7)

# Mark when NTP correction stopped
if len(with_correction) > 0 and len(without_correction) > 0:
    stop_time = with_correction['elapsed_seconds'].max()
    ax.axvline(x=stop_time, color='red', linestyle='--', linewidth=2, label=f'NTP correction stopped ({stop_time/60:.1f}min)')

ax.set_xlabel('Time (seconds)', fontweight='bold')
ax.set_ylabel('Error (ms)', fontweight='bold')
ax.set_title('Error Evolution (WITH NTP Correction Test)')
ax.legend()
ax.grid(alpha=0.3)

# Plot 3: Source distribution over time
ax = axes[1, 0]
source_colors = {'cpu+ntp': '#4CAF50', 'cpu': '#FF9800', 'fusion+ntp': '#2196F3'}
for source in with_ntp['chronotick_source'].unique():
    mask = with_ntp['chronotick_source'] == source
    ax.scatter(with_ntp[mask]['elapsed_seconds'],
              with_ntp[mask]['chronotick_offset_ms'],
              label=source, alpha=0.6, s=20,
              color=source_colors.get(source, '#999999'))

ax.set_xlabel('Time (seconds)', fontweight='bold')
ax.set_ylabel('ChronoTick Offset (ms)', fontweight='bold')
ax.set_title('ChronoTick Source Over Time')
ax.legend()
ax.grid(alpha=0.3)

# Plot 4: Comparison table
ax = axes[1, 1]
ax.axis('off')

table_data = [
    ['Metric', 'WITHOUT NTP', 'WITH NTP\n(overall)', 'WITH NTP\n(active)'],
    ['ChronoTick Error',
     f'{without_ntp_measurements["chronotick_error"].mean():.1f} ms',
     f'{with_ntp_measurements["chronotick_error"].mean():.1f} ms',
     f'{with_correction_at_ntp["chronotick_error"].mean():.1f} ms' if len(with_correction_at_ntp) > 0 else 'N/A'],
    ['Win Rate',
     f'{100*(without_ntp_measurements["chronotick_error"] < without_ntp_measurements["system_error"]).sum()/len(without_ntp_measurements):.1f}%',
     f'{100*(with_ntp_measurements["chronotick_error"] < with_ntp_measurements["system_error"]).sum()/len(with_ntp_measurements):.1f}%',
     f'{100*(with_correction_at_ntp["chronotick_error"] < with_correction_at_ntp["system_error"]).sum()/len(with_correction_at_ntp):.1f}%' if len(with_correction_at_ntp) > 0 else 'N/A'],
    ['NTP Samples',
     f'{len(without_ntp_measurements)}',
     f'{len(with_ntp_measurements)}',
     f'{len(with_correction_at_ntp)}' if len(with_correction_at_ntp) > 0 else 'N/A'],
]

table = ax.table(cellText=table_data, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Color header
for i in range(4):
    table[(0, i)].set_facecolor('#E0E0E0')
    table[(0, i)].set_text_props(weight='bold')

plt.tight_layout()
plt.savefig('/tmp/chronotick_ntp_correction_comparison.png', dpi=150, bbox_inches='tight')
print(f"✓ Comparison chart saved to: /tmp/chronotick_ntp_correction_comparison.png")
print()

print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()

if len(with_correction_at_ntp) > 0:
    active_error = with_correction_at_ntp['chronotick_error'].mean()
    baseline_error = without_ntp_measurements['chronotick_error'].mean()

    if active_error < baseline_error:
        improvement = ((baseline_error - active_error) / baseline_error) * 100
        print(f"✅ NTP Correction WORKS when active!")
        print(f"   Error reduced from {baseline_error:.2f}ms to {active_error:.2f}ms ({improvement:.1f}% improvement)")
        print()
        print(f"⚠️  BUT it stopped working after {with_correction['elapsed_seconds'].max()/60:.1f} minutes")
        print(f"   This caused overall performance to degrade")
    else:
        print(f"❌ NTP Correction made things worse")
else:
    print(f"❌ NTP Correction was never active during the test")

print()
