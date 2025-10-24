#!/usr/bin/env python3
"""Create summary comparison figure across all experiments."""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'figure.dpi': 300,
    'savefig.dpi': 300,
})

# Load results
with open('results/figures/crazy_ideas_CORRECT/summary_results.json', 'r') as f:
    results = json.load(f)

# Extract data
experiments = ['experiment-5', 'experiment-7', 'experiment-10']
exp_labels = ['Exp-5\n(Best)', 'Exp-7\n(Medium)', 'Exp-10\n(Worst)']

# Test 1: Bidirectional Alignment
test1_rates = [results[exp]['test1']['overall_rate'] for exp in experiments]
test1_samples = [results[exp]['test1']['test1_samples'] + results[exp]['test1']['test2_samples']
                 for exp in experiments]

# Test 2: Consensus Windows
test2_rates = [results[exp]['test2']['rate'] for exp in experiments]
test2_samples = [results[exp]['test2']['samples'] for exp in experiments]

# Test 3: Calibration (average both nodes)
test3_rates = [(results[exp]['test3']['node1_3sigma'] + results[exp]['test3']['node2_3sigma']) / 2
               for exp in experiments]
test3_samples = [results[exp]['test3']['samples'] for exp in experiments]

# Test 4: Distributed Lock
test4_rates = [results[exp]['test4']['rate'] for exp in experiments]
test4_samples = [results[exp]['test4']['samples'] for exp in experiments]

# Create figure
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

# Test 1: Bidirectional Alignment
ax1 = fig.add_subplot(gs[0, 0])
bars1 = ax1.bar(exp_labels, test1_rates, color=['#009E73', '#E69F00', '#D55E00'], alpha=0.8, edgecolor='black', linewidth=1.5)
ax1.set_ylabel('Agreement Rate (%)', fontweight='bold')
ax1.set_title('(a) Bidirectional Timeline Alignment\n(Cross-node prediction accuracy)', fontweight='bold', fontsize=12)
ax1.axhline(90, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='90% target')
ax1.set_ylim(0, 105)
ax1.grid(axis='y', alpha=0.3)
ax1.legend(loc='upper right')

# Add percentage labels
for i, (bar, rate, samples) in enumerate(zip(bars1, test1_rates, test1_samples)):
    ax1.text(i, rate + 2, f'{rate:.1f}%\n({samples} samples)', ha='center', fontweight='bold', fontsize=10)

# Calculate and show mean
mean1 = np.mean(test1_rates)
std1 = np.std(test1_rates)
ax1.axhline(mean1, color='blue', linestyle=':', linewidth=2, alpha=0.7, label=f'Mean: {mean1:.1f}±{std1:.1f}%')
ax1.legend(loc='upper right')

# Test 2: Consensus Windows
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(exp_labels, test2_rates, color=['#009E73', '#009E73', '#009E73'], alpha=0.8, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('Overlap Rate (%)', fontweight='bold')
ax2.set_title('(b) Consensus Windows\n(ChronoTick ±3σ overlap)', fontweight='bold', fontsize=12)
ax2.axhline(99.7, color='green', linestyle=':', linewidth=1, alpha=0.6, label='Expected (99.7%)')
ax2.set_ylim(95, 105)
ax2.grid(axis='y', alpha=0.3)
ax2.legend(loc='upper right')

# Add percentage labels
for i, (bar, rate, samples) in enumerate(zip(bars2, test2_rates, test2_samples)):
    ax2.text(i, rate + 0.5, f'{rate:.1f}%\n({samples} samples)', ha='center', fontweight='bold', fontsize=10)

# Add warning annotation
ax2.text(0.5, 0.15, '⚠️ 100% overlap suggests\noverly conservative bounds',
         transform=ax2.transAxes, ha='center', fontsize=10,
         bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

# Test 3: Uncertainty Calibration
ax3 = fig.add_subplot(gs[1, 0])
bars3 = ax3.bar(exp_labels, test3_rates, color=['#E69F00', '#E69F00', '#CC79A7'], alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('±3σ Coverage (%)', fontweight='bold')
ax3.set_title('(c) Uncertainty Calibration\n(Should be 99.7% for well-calibrated)', fontweight='bold', fontsize=12)
ax3.axhline(99.7, color='green', linestyle=':', linewidth=2, alpha=0.7, label='Expected (99.7%)')
ax3.axhline(95, color='orange', linestyle='--', linewidth=1, alpha=0.6, label='Acceptable (95%)')
ax3.set_ylim(0, 110)
ax3.grid(axis='y', alpha=0.3)
ax3.legend(loc='upper right')

# Add percentage labels
for i, (bar, rate, samples) in enumerate(zip(bars3, test3_rates, test3_samples)):
    ax3.text(i, rate + 3, f'{rate:.1f}%\n({samples} samples)', ha='center', fontweight='bold', fontsize=10)

# Add deficit annotation
for i, rate in enumerate(test3_rates):
    deficit = 99.7 - rate
    ax3.text(i, rate / 2, f'-{deficit:.1f}%\ndeficit', ha='center', fontsize=9, color='red', fontweight='bold')

# Test 4: Distributed Lock
ax4 = fig.add_subplot(gs[1, 1])
bars4 = ax4.bar(exp_labels, test4_rates, color=['#CC79A7', '#CC79A7', '#E69F00'], alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_ylabel('Agreement Rate (%)', fontweight='bold')
ax4.set_title('(d) Distributed Lock Agreement\n(Both nodes use ChronoTick pessimistic)', fontweight='bold', fontsize=12)
ax4.axhline(90, color='gray', linestyle='--', linewidth=1, alpha=0.6, label='90% target')
ax4.set_ylim(0, 105)
ax4.grid(axis='y', alpha=0.3)
ax4.legend(loc='upper right')

# Add percentage labels
for i, (bar, rate, samples) in enumerate(zip(bars4, test4_rates, test4_samples)):
    ax4.text(i, rate + 2, f'{rate:.1f}%\n({samples} samples)', ha='center', fontweight='bold', fontsize=10)

# Calculate and show mean
mean4 = np.mean(test4_rates)
std4 = np.std(test4_rates)
ax4.axhline(mean4, color='blue', linestyle=':', linewidth=2, alpha=0.7, label=f'Mean: {mean4:.1f}±{std4:.1f}%')
ax4.legend(loc='upper right')

# Overall title
fig.suptitle('Uncertainty-Aware Coordination: Cross-Experiment Comparison\n(Corrected Evaluation)',
             fontsize=14, fontweight='bold', y=0.98)

# Save
output_dir = Path('results/figures/crazy_ideas_CORRECT')
output_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(output_dir / 'SUMMARY_cross_experiment_comparison.pdf', bbox_inches='tight')
plt.savefig(output_dir / 'SUMMARY_cross_experiment_comparison.png', dpi=300, bbox_inches='tight')

print(f"\n✓ Saved: {output_dir / 'SUMMARY_cross_experiment_comparison.pdf'}")
print(f"✓ Saved: {output_dir / 'SUMMARY_cross_experiment_comparison.png'}")

# Print summary statistics
print("\n" + "="*80)
print("CROSS-EXPERIMENT SUMMARY STATISTICS")
print("="*80)

print(f"\nTest 1 (Bidirectional Alignment):")
print(f"  Best: {max(test1_rates):.1f}% (Exp-5)")
print(f"  Worst: {min(test1_rates):.1f}% (Exp-10)")
print(f"  Mean ± Std: {mean1:.1f} ± {std1:.1f}%")

print(f"\nTest 2 (Consensus Windows):")
print(f"  All experiments: 100.0%")
print(f"  ⚠️ Indicates overly conservative uncertainty bounds")

print(f"\nTest 3 (Uncertainty Calibration):")
print(f"  Best: {max(test3_rates):.1f}% (Exp-5, target: 99.7%)")
print(f"  Worst: {min(test3_rates):.1f}% (Exp-10, target: 99.7%)")
print(f"  Mean deficit: {99.7 - np.mean(test3_rates):.1f}%")

print(f"\nTest 4 (Distributed Lock):")
print(f"  Best: {max(test4_rates):.1f}% (Exp-10)")
print(f"  Worst: {min(test4_rates):.1f}% (Exp-7)")
print(f"  Mean ± Std: {mean4:.1f} ± {std4:.1f}%")

print("\n" + "="*80)
print("OVERALL VERDICT")
print("="*80)
print(f"✅ Bidirectional alignment: {mean1:.1f}% (acceptable for distributed coordination)")
print(f"⚠️ Consensus windows: 100% (overly conservative)")
print(f"❌ Calibration: {np.mean(test3_rates):.1f}% (poor, needs improvement)")
print(f"⚠️ Distributed lock: {mean4:.1f}% (marginal, pessimistic strategy problematic)")

plt.close()
