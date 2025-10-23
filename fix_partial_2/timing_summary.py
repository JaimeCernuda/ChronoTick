#!/usr/bin/env python3
"""Visual comparison of Sequential vs Parallel NTP queries"""

import matplotlib.pyplot as plt
import numpy as np

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('NTP Query Performance: Sequential vs Parallel', fontsize=16, fontweight='bold')

# ============================================================================
# Plot 1: Timing Comparison - 4 Servers
# ============================================================================
ax = axes[0, 0]

scenarios = ['Best\nCase', 'Mixed\nCase', 'Worst\nCase']
sequential_4 = [1160, 4580, 8000]  # ms
parallel_4 = [35, 300, 2000]  # ms

x = np.arange(len(scenarios))
width = 0.35

bars1 = ax.bar(x - width/2, sequential_4, width, label='Sequential', alpha=0.7, color='#d62728')
bars2 = ax.bar(x + width/2, parallel_4, width, label='Parallel', alpha=0.7, color='#2ca02c')

# Add speedup labels
for i in range(len(scenarios)):
    speedup = sequential_4[i] / parallel_4[i]
    ax.text(i, max(sequential_4[i], parallel_4[i]) + 500,
           f'{speedup:.0f}x\nfaster', ha='center', fontsize=10, fontweight='bold')

ax.set_ylabel('Time (ms)', fontsize=12)
ax.set_title('4 Servers: Sequential vs Parallel', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(scenarios)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 9000])

# Add horizontal line at 2 seconds (acceptable threshold)
ax.axhline(y=2000, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='2s threshold')

# ============================================================================
# Plot 2: Timing Comparison - 10 Servers
# ============================================================================
ax = axes[0, 1]

sequential_10 = [2900, 9740, 20000]  # ms
parallel_10 = [50, 350, 2000]  # ms

bars1 = ax.bar(x - width/2, sequential_10, width, label='Sequential', alpha=0.7, color='#d62728')
bars2 = ax.bar(x + width/2, parallel_10, width, label='Parallel', alpha=0.7, color='#2ca02c')

# Add speedup labels
for i in range(len(scenarios)):
    speedup = sequential_10[i] / parallel_10[i]
    ax.text(i, max(sequential_10[i], parallel_10[i]) + 1000,
           f'{speedup:.0f}x\nfaster', ha='center', fontsize=10, fontweight='bold')

ax.set_ylabel('Time (ms)', fontsize=12)
ax.set_title('10 Servers: Sequential vs Parallel', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(scenarios)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0, 22000])
ax.axhline(y=2000, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='2s threshold')

# ============================================================================
# Plot 3: Scaling with Server Count
# ============================================================================
ax = axes[1, 0]

server_counts = [1, 2, 4, 6, 8, 10]
sequential_times = [290, 580, 1160, 1740, 2320, 2900]  # Best case: 290ms per server
parallel_times = [30, 35, 35, 40, 45, 50]  # Best case: max(all servers)

ax.plot(server_counts, sequential_times, marker='o', linewidth=3, markersize=10,
        label='Sequential (Best Case)', color='#d62728')
ax.plot(server_counts, parallel_times, marker='s', linewidth=3, markersize=10,
        label='Parallel (Best Case)', color='#2ca02c')

ax.fill_between(server_counts, parallel_times,
                alpha=0.2, color='#2ca02c', label='Parallel advantage')

ax.set_xlabel('Number of Servers', fontsize=12)
ax.set_ylabel('Time (ms)', fontsize=12)
ax.set_title('Scaling: How Time Grows with Server Count', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 3200])

# Add annotations
ax.annotate('Linear growth\n(O(n))', xy=(10, 2900), xytext=(8, 2400),
           fontsize=10, ha='center',
           arrowprops=dict(arrowstyle='->', color='#d62728', lw=2))

ax.annotate('Constant time\n(O(1))', xy=(10, 50), xytext=(7, 400),
           fontsize=10, ha='center',
           arrowprops=dict(arrowstyle='->', color='#2ca02c', lw=2))

# ============================================================================
# Plot 4: Timeline Visualization
# ============================================================================
ax = axes[1, 1]
ax.set_xlim([0, 3000])
ax.set_ylim([0, 6])

# Sequential timeline
servers = ['Server 1', 'Server 2', 'Server 3', 'Server 4']
start_times = [0, 290, 580, 870]
durations = [290, 290, 290, 290]

for i, (server, start, duration) in enumerate(zip(servers, start_times, durations)):
    ax.barh(4.5, duration, left=start, height=0.3, alpha=0.7, color='#d62728')
    ax.text(start + duration/2, 4.5, f'S{i+1}', ha='center', va='center', fontsize=9, fontweight='bold')

ax.text(-150, 4.5, 'Sequential:', ha='right', va='center', fontsize=11, fontweight='bold')
ax.text(1160, 4.5, '1160ms total', va='center', fontsize=10, fontweight='bold')

# Parallel timeline
for i, (server, duration) in enumerate(zip(servers, [30, 35, 28, 32])):
    ax.barh(3.5 - i*0.35, duration, left=0, height=0.25, alpha=0.7, color='#2ca02c')
    ax.text(duration + 5, 3.5 - i*0.35, f'S{i+1}: {duration}ms', va='center', fontsize=9)

ax.text(-150, 3.3, 'Parallel:', ha='right', va='center', fontsize=11, fontweight='bold')

# Draw max line
ax.plot([35, 35], [2, 4], color='#2ca02c', linewidth=3, linestyle='--')
ax.text(35, 1.8, '35ms\n(max)', ha='center', fontsize=10, fontweight='bold', color='#2ca02c')

# Cleanup
ax.set_xlabel('Time (ms)', fontsize=12)
ax.set_title('Timeline Visualization: Sequential vs Parallel', fontsize=14, fontweight='bold')
ax.set_yticks([])
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim([0, 1300])

plt.tight_layout()
plt.savefig('ntp_parallel_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Timing comparison plot saved to: ntp_parallel_comparison.png")

# ============================================================================
# Summary table
# ============================================================================
print("\n" + "="*80)
print("NTP QUERY TIMING SUMMARY")
print("="*80)

print("\n4 SERVERS:")
print(f"  Sequential - Best: {sequential_4[0]}ms, Mixed: {sequential_4[1]}ms, Worst: {sequential_4[2]}ms")
print(f"  Parallel   - Best: {parallel_4[0]}ms, Mixed: {parallel_4[1]}ms, Worst: {parallel_4[2]}ms")
print(f"  Speedup    - Best: {sequential_4[0]/parallel_4[0]:.0f}x, Mixed: {sequential_4[1]/parallel_4[1]:.0f}x, Worst: {sequential_4[2]/parallel_4[2]:.0f}x")

print("\n10 SERVERS:")
print(f"  Sequential - Best: {sequential_10[0]}ms, Mixed: {sequential_10[1]}ms, Worst: {sequential_10[2]}ms")
print(f"  Parallel   - Best: {parallel_10[0]}ms, Mixed: {parallel_10[1]}ms, Worst: {parallel_10[2]}ms")
print(f"  Speedup    - Best: {sequential_10[0]/parallel_10[0]:.0f}x, Mixed: {sequential_10[1]/parallel_10[1]:.0f}x, Worst: {sequential_10[2]/parallel_10[2]:.0f}x")

print("\n" + "="*80)
print("RECOMMENDATION: Use PARALLEL queries + 10 servers")
print("="*80)
print("  ✓ Query time: 50-350ms (vs 2.9-20 seconds sequential)")
print("  ✓ No performance penalty for more servers")
print("  ✓ Better resilience (more servers, no slowdown)")
print("  ✓ Simple implementation (~30 lines of code)")
print("="*80)
