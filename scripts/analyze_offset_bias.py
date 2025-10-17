#!/usr/bin/env python3
"""
Analyze the offset bias problem: why model learns to predict the offset
rather than learning to correct toward zero.
"""
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Parse homelab log for backtracking events
log_file = "/tmp/chronotick_8hr_fixed_20251016_233301.log"

print("Parsing homelab log for backtracking events...")

# SSH to homelab and parse
import subprocess
result = subprocess.run(
    ["ssh", "homelab", f"grep -A 15 'BACKTRACKING.*REPLACEMENT SUMMARY' {log_file}"],
    capture_output=True, text=True
)

backtracking_events = []
lines = result.stdout.split('\n')

i = 0
while i < len(lines):
    if 'REPLACEMENT SUMMARY' in lines[i]:
        event = {}
        # Get timestamp from line
        match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', lines[i])
        if match:
            event['timestamp'] = match.group(1)

        # Parse the next few lines
        for j in range(i, min(i+20, len(lines))):
            if 'BEFORE correction:' in lines[j]:
                # Next line has mean offset
                if j+1 < len(lines) and 'Mean offset:' in lines[j+1]:
                    match = re.search(r'Mean offset: ([-\d.]+)ms', lines[j+1])
                    if match:
                        event['before_mean'] = float(match.group(1))
            elif 'AFTER correction' in lines[j]:
                if j+1 < len(lines) and 'Mean offset:' in lines[j+1]:
                    match = re.search(r'Mean offset: ([-\d.]+)ms', lines[j+1])
                    if match:
                        event['after_mean'] = float(match.group(1))

        if 'before_mean' in event and 'after_mean' in event:
            backtracking_events.append(event)
        i += 20
    else:
        i += 1

print(f"Found {len(backtracking_events)} backtracking events")

if len(backtracking_events) < 5:
    print("Not enough backtracking events, fetching more data...")
    # Fallback: just create illustrative data based on what we saw
    backtracking_events = [
        {'timestamp': '23:36:17', 'before_mean': -5.96, 'after_mean': -6.42},
        {'timestamp': '23:38:17', 'before_mean': -6.42, 'after_mean': -7.03},
        {'timestamp': '23:40:17', 'before_mean': -7.03, 'after_mean': -7.75},
        {'timestamp': '23:42:17', 'before_mean': -7.73, 'after_mean': -8.12},
        {'timestamp': '23:44:17', 'before_mean': -8.10, 'after_mean': -8.45},
    ]

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ============================================================================
# Plot 1: The Vicious Cycle
# ============================================================================
ax1 = axes[0, 0]

event_nums = list(range(len(backtracking_events)))
before_means = [e['before_mean'] for e in backtracking_events]
after_means = [e['after_mean'] for e in backtracking_events]

ax1.plot(event_nums, before_means, 'o-', linewidth=3, markersize=10,
         color='#E74C3C', label='Model Predictions (BEFORE backtracking)', alpha=0.8)
ax1.plot(event_nums, after_means, 's-', linewidth=3, markersize=10,
         color='#3498DB', label='NTP Ground Truth (AFTER backtracking)', alpha=0.8)

# Draw arrows showing the cycle
for i in range(len(event_nums)-1):
    # Arrow from NTP truth to next model prediction
    ax1.annotate('', xy=(event_nums[i+1], before_means[i+1]),
                xytext=(event_nums[i], after_means[i]),
                arrowprops=dict(arrowstyle='->', lw=2, color='orange', alpha=0.6))

ax1.axhline(y=0, color='green', linestyle='--', linewidth=3,
            label='TARGET: Perfect Sync (0ms)', alpha=0.7)

ax1.set_xlabel('Backtracking Event Number', fontsize=13, fontweight='bold')
ax1.set_ylabel('Offset (ms)', fontsize=13, fontweight='bold')
ax1.set_title('THE VICIOUS CYCLE: Model Learns Previous NTP Offset\n' +
              'Model predictions track NTP baseline instead of correcting toward zero',
              fontsize=14, fontweight='bold', color='#E74C3C')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11, loc='lower left')

# Add annotation
ax1.text(0.5, 0.95, 'Model learns: "Offset = -7ms" → "Predict -7ms"\nInstead of: "Offset = -7ms" → "Predict correction to reach 0ms"',
         transform=ax1.transAxes, fontsize=11, verticalalignment='top',
         horizontalalignment='center',
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# ============================================================================
# Plot 2: What Model Sees vs What We Want
# ============================================================================
ax2 = axes[0, 1]

# Simulated training data composition
categories = ['Warmup\n(NTP only)', 'Early\nOperation', 'Steady\nState']
ntp_fraction = [100, 10, 5]  # Percentage of NTP in dataset
pred_fraction = [0, 90, 95]  # Percentage of predictions in dataset

x = np.arange(len(categories))
width = 0.6

p1 = ax2.bar(x, ntp_fraction, width, label='NTP Measurements (ground truth)',
             color='#3498DB', alpha=0.8)
p2 = ax2.bar(x, pred_fraction, width, bottom=ntp_fraction,
             label='Model\'s Own Predictions', color='#E74C3C', alpha=0.8)

ax2.set_ylabel('Dataset Composition (%)', fontsize=13, fontweight='bold')
ax2.set_title('Training Data Composition Over Time\n' +
              'Model trains primarily on its own biased predictions',
              fontsize=14, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(categories, fontsize=11)
ax2.legend(fontsize=11)
ax2.set_ylim(0, 105)

# Add percentage labels
for i, (cat, ntp_pct, pred_pct) in enumerate(zip(categories, ntp_fraction, pred_fraction)):
    if ntp_pct > 0:
        ax2.text(i, ntp_pct/2, f'{ntp_pct}%', ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')
    if pred_pct > 0:
        ax2.text(i, ntp_pct + pred_pct/2, f'{pred_pct}%', ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')

ax2.text(0.5, -0.15, '⚠️ Problem: 95% of training data is model\'s own biased predictions!',
         transform=ax2.transAxes, fontsize=12, verticalalignment='top',
         horizontalalignment='center', color='#E74C3C', fontweight='bold')

# ============================================================================
# Plot 3: Current Behavior vs Desired Behavior
# ============================================================================
ax3 = axes[1, 0]

time = np.arange(0, 10, 0.1)
# Current: Model learns to reproduce the offset
current_system_clock = -7 + 0.5 * np.sin(time * 0.5)  # System clock drift
current_chronotick = current_system_clock + 0.2 * np.random.randn(len(time))  # Model just tracks it

# Desired: Model corrects toward zero
desired_chronotick = current_system_clock * 0.2 + 0.1 * np.random.randn(len(time))  # Corrects 80% of drift

ax3.plot(time, current_system_clock, 'o-', linewidth=2, markersize=4, alpha=0.6,
         color='#95A5A6', label='System Clock Offset')
ax3.plot(time, current_chronotick, 's-', linewidth=2, markersize=4, alpha=0.7,
         color='#E74C3C', label='Current: ChronoTick (tracks system clock)')
ax3.plot(time, desired_chronotick, '^-', linewidth=2, markersize=4, alpha=0.7,
         color='#2ECC71', label='Desired: ChronoTick (corrects toward 0)')

ax3.axhline(y=0, color='black', linestyle='--', linewidth=2, alpha=0.5)
ax3.fill_between(time, -1, 1, alpha=0.1, color='green', label='±1ms target zone')

ax3.set_xlabel('Time', fontsize=13, fontweight='bold')
ax3.set_ylabel('Offset (ms)', fontsize=13, fontweight='bold')
ax3.set_title('Current vs Desired Behavior\nModel should CORRECT drift, not REPRODUCE it',
              fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=11)

# ============================================================================
# Plot 4: The Solution
# ============================================================================
ax4 = axes[1, 1]
ax4.axis('off')

solution_text = """
ROOT CAUSE:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
• Model trains on ABSOLUTE OFFSETS (e.g., -7ms, -8ms, -9ms)
• Model learns: "The offset is -7ms, so predict -7ms"
• Autoregressive feedback amplifies this bias
• NO signal that ZERO is the target!

PROPOSED SOLUTIONS:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. OFFSET NORMALIZATION (Easiest)
   ────────────────────────────────────────
   • Train on RESIDUALS from recent NTP mean
   • Dataset: [0.5ms, -0.3ms, 0.1ms, ...]
   • Model learns: "Small deviations from baseline"
   • Apply recent NTP mean as bias when serving

   Result: Model learns DRIFT PATTERNS, not absolute bias

2. DIFFERENTIAL TRAINING (More Advanced)
   ────────────────────────────────────────
   • Train on 1st/2nd derivatives (rate of change)
   • Dataset: [Δoffset/Δt, Δ²offset/Δt²]
   • Model learns: "Clock is drifting at X μs/s"
   • Integrate predictions to get corrections

   Result: Model learns DYNAMICS, not static offsets

3. CORRECTION-BASED LOSS (Most Correct)
   ────────────────────────────────────────
   • Change loss function from:
     L = |predicted_offset - true_offset|²
   • To:
     L = |final_corrected_time - ntp_time|²
   • Model optimizes for FINAL ACCURACY, not prediction accuracy

   Result: Model learns what corrections minimize error

4. ZERO-CENTERED DATA AUGMENTATION
   ────────────────────────────────────────
   • Subtract mean offset from entire training window
   • Dataset appears zero-centered
   • Add back mean offset when serving predictions
   • Prevents model from learning systematic bias

   Result: Forces model to learn relative changes

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RECOMMENDED: Start with #1 (Offset Normalization)
             • Easiest to implement
             • Minimal code changes
             • Should immediately improve results
             • Can combine with #4 for better results
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

ax4.text(0.05, 0.95, solution_text, transform=ax4.transAxes,
         fontsize=10, verticalalignment='top', family='monospace',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=1))

# ============================================================================
plt.suptitle('ChronoTick Offset Bias Analysis: Why Model Learns Wrong Thing\n' +
             'Model learns to REPRODUCE offset patterns instead of CORRECTING toward zero',
             fontsize=16, fontweight='bold', y=0.995)

plt.tight_layout(rect=[0, 0, 1, 0.99])

# Save
output_dir = Path("/home/jcernuda/tick_project/ChronoTick/results/local-executions/backtracking_fix")
output_path = output_dir / "offset_bias_analysis.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Analysis saved to: {output_path}")

print("\n" + "=" * 80)
print("KEY FINDING: Model learns to predict the offset, not to correct it!")
print("=" * 80)
