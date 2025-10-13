#!/usr/bin/env python3
"""
Analyze Client-Driven Validation Results

Analyzes CSV from client_driven_validation.py to:
1. Calculate accuracy: ChronoTick vs System Clock (using NTP as ground truth)
2. Generate visualizations showing time series and error comparisons
"""

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# Check if CSV path provided
if len(sys.argv) < 2:
    print("Usage: uv run python scripts/analyze_client_validation.py <csv_path>")
    print("Example: uv run python scripts/analyze_client_validation.py /tmp/chronotick_client_validation.csv")
    sys.exit(1)

csv_path = sys.argv[1]

if not Path(csv_path).exists():
    print(f"Error: CSV file not found: {csv_path}")
    sys.exit(1)

print("=" * 80)
print("CLIENT VALIDATION ANALYSIS")
print("=" * 80)
print(f"Analyzing: {csv_path}")
print()

# Read CSV data
samples = []
ntp_samples = []

with open(csv_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        samples.append(row)
        if row['has_ntp'] == 'True':
            ntp_samples.append(row)

print(f"Total samples: {len(samples)}")
print(f"NTP measurements: {len(ntp_samples)}")
print()

if len(ntp_samples) == 0:
    print("Error: No NTP measurements found in CSV. Cannot calculate accuracy.")
    sys.exit(1)

# Parse data
elapsed_times = [float(s['elapsed_seconds']) for s in samples]
system_times = [float(s['system_time']) for s in samples]
chronotick_times = [float(s['chronotick_time']) for s in samples]
chronotick_offsets = [float(s['chronotick_offset_ms']) for s in samples]
chronotick_uncertainties = [float(s['chronotick_uncertainty_ms']) for s in samples]
chronotick_sources = [s['chronotick_source'] for s in samples]

# NTP data
ntp_elapsed = [float(s['elapsed_seconds']) for s in ntp_samples]
ntp_times = [float(s['ntp_time']) for s in ntp_samples]
ntp_offsets = [float(s['ntp_offset_ms']) for s in ntp_samples]

print("=" * 80)
print("ACCURACY ANALYSIS (using NTP as ground truth)")
print("=" * 80)
print()

# Calculate errors at NTP measurement points
system_errors = []
chronotick_errors = []

print("Per-NTP-Sample Error Analysis:")
print(f"{'Time (s)':>10} | {'System Error (ms)':>18} | {'ChronoTick Error (ms)':>22} | {'Winner':>10}")
print("-" * 80)

for i, ntp_sample in enumerate(ntp_samples):
    idx = samples.index(ntp_sample)

    # NTP time is the ground truth
    ntp_time = ntp_times[i]
    system_time = system_times[idx]
    chronotick_time = chronotick_times[idx]

    # Calculate absolute errors
    system_error = abs((system_time - ntp_time) * 1000)  # Convert to ms
    chronotick_error = abs((chronotick_time - ntp_time) * 1000)

    system_errors.append(system_error)
    chronotick_errors.append(chronotick_error)

    winner = "ChronoTick" if chronotick_error < system_error else "System"
    if chronotick_error < system_error:
        improvement = ((system_error - chronotick_error) / system_error) * 100
        winner_str = f"ChronoTick (-{improvement:.1f}%)"
    else:
        degradation = ((chronotick_error - system_error) / system_error) * 100
        winner_str = f"System (+{degradation:.1f}%)"

    print(f"{ntp_elapsed[i]:>10.1f} | {system_error:>18.2f} | {chronotick_error:>22.2f} | {winner_str:>15}")

print()
print("=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)
print()

# Calculate statistics
system_mean = np.mean(system_errors)
system_std = np.std(system_errors)
system_max = np.max(system_errors)
system_min = np.min(system_errors)

chronotick_mean = np.mean(chronotick_errors)
chronotick_std = np.std(chronotick_errors)
chronotick_max = np.max(chronotick_errors)
chronotick_min = np.min(chronotick_errors)

print(f"System Clock Error (vs NTP ground truth):")
print(f"  Mean:   {system_mean:.2f} ms")
print(f"  StdDev: {system_std:.2f} ms")
print(f"  Min:    {system_min:.2f} ms")
print(f"  Max:    {system_max:.2f} ms")
print()

print(f"ChronoTick Error (vs NTP ground truth):")
print(f"  Mean:   {chronotick_mean:.2f} ms")
print(f"  StdDev: {chronotick_std:.2f} ms")
print(f"  Min:    {chronotick_min:.2f} ms")
print(f"  Max:    {chronotick_max:.2f} ms")
print()

# Calculate improvement
improvement = ((system_mean - chronotick_mean) / system_mean) * 100
wins = sum(1 for i in range(len(system_errors)) if chronotick_errors[i] < system_errors[i])
win_rate = (wins / len(system_errors)) * 100

print(f"ChronoTick vs System Clock:")
if improvement > 0:
    print(f"  ✅ Improvement: {improvement:.1f}% (ChronoTick is MORE accurate)")
else:
    print(f"  ❌ Degradation: {abs(improvement):.1f}% (ChronoTick is LESS accurate)")

print(f"  Win rate: {win_rate:.1f}% ({wins}/{len(system_errors)} NTP samples)")
print()

# Generate visualizations
print("=" * 80)
print("GENERATING VISUALIZATIONS")
print("=" * 80)
print()

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('ChronoTick Client-Driven Validation Results', fontsize=16, fontweight='bold')

# Plot 1: Time series comparison
ax1 = axes[0, 0]
ax1.plot(elapsed_times, chronotick_offsets, 'b-', label='ChronoTick Offset', linewidth=1, alpha=0.7)
ax1.scatter(ntp_elapsed, ntp_offsets, c='red', s=100, marker='x', label='NTP Measurements', zorder=5)
ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.3)
ax1.set_xlabel('Time (seconds)')
ax1.set_ylabel('Offset from System Clock (ms)')
ax1.set_title('Time Offset Over Test Duration')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Uncertainty bounds
ax2 = axes[0, 1]
ax2.plot(elapsed_times, chronotick_offsets, 'b-', label='ChronoTick Offset', linewidth=1)
uncertainties = np.array(chronotick_uncertainties)
offsets = np.array(chronotick_offsets)
ax2.fill_between(elapsed_times,
                  offsets - uncertainties,
                  offsets + uncertainties,
                  alpha=0.3, label='Uncertainty Bounds')
ax2.scatter(ntp_elapsed, ntp_offsets, c='red', s=100, marker='x', label='NTP Ground Truth', zorder=5)
ax2.set_xlabel('Time (seconds)')
ax2.set_ylabel('Offset (ms)')
ax2.set_title('ChronoTick Predictions with Uncertainty Bounds')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Error comparison bar chart
ax3 = axes[1, 0]
x_pos = np.arange(2)
means = [system_mean, chronotick_mean]
stds = [system_std, chronotick_std]
colors = ['orange', 'green']
ax3.bar(x_pos, means, yerr=stds, color=colors, alpha=0.7, capsize=10)
ax3.set_xticks(x_pos)
ax3.set_xticklabels(['System Clock', 'ChronoTick'])
ax3.set_ylabel('Absolute Error (ms)')
ax3.set_title('Mean Absolute Error vs NTP Ground Truth')
ax3.grid(True, alpha=0.3, axis='y')

# Add improvement percentage
improvement_text = f"{improvement:+.1f}%" if improvement != 0 else "0.0%"
ax3.text(1, chronotick_mean + chronotick_std + 5, improvement_text,
         ha='center', fontsize=14, fontweight='bold',
         color='green' if improvement > 0 else 'red')

# Plot 4: Error time series
ax4 = axes[1, 1]
ax4.plot(ntp_elapsed, system_errors, 'o-', color='orange', label='System Clock Error',
         linewidth=2, markersize=8, alpha=0.7)
ax4.plot(ntp_elapsed, chronotick_errors, 's-', color='green', label='ChronoTick Error',
         linewidth=2, markersize=8, alpha=0.7)
ax4.set_xlabel('Time (seconds)')
ax4.set_ylabel('Absolute Error (ms)')
ax4.set_title('Error Evolution Over Time (at NTP Measurement Points)')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
output_path = csv_path.replace('.csv', '_analysis.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"✓ Plot saved to: {output_path}")

# Also display
try:
    plt.show()
    print("✓ Plot displayed")
except:
    print("  (Display unavailable in this environment)")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print()

if improvement > 0:
    print(f"✅ ChronoTick provides {improvement:.1f}% better accuracy than system clock!")
else:
    print(f"⚠️  ChronoTick is {abs(improvement):.1f}% less accurate than system clock")
    print("   This may indicate configuration issues or insufficient training data.")
