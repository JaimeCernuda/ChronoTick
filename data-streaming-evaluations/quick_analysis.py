#!/usr/bin/env python3
"""Quick analysis of production-run-1 data to show the ground truth problem"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
coord = pd.read_csv('results/production-run-1/coordinator.csv')
worker_b = pd.read_csv('results/production-run-1/worker_comp11.csv')
worker_c = pd.read_csv('results/production-run-1/worker_comp12.csv')

# Merge on event_id
merged_b = coord.merge(worker_b, on='event_id', suffixes=('_coord', '_worker'))
merged_c = coord.merge(worker_c, on='event_id', suffixes=('_coord', '_worker'))

print("="*60)
print("CURRENT DATA ANALYSIS - production-run-1")
print("="*60)
print()

print(f"Total events: {len(coord)}")
print(f"Worker B received: {len(worker_b)}")
print(f"Worker C received: {len(worker_c)}")
print()

print("Available timestamps:")
print("-" * 60)
print("Coordinator:")
print(f"  ✓ send_time_ns (system clock)")
print(f"  ✗ ntp_timestamp (NOT RECORDED - coordinator didn't use --ntp-server)")
print(f"  ✗ ct_timestamp (NOT RECORDED)")
print()
print("Workers:")
print(f"  ✓ receive_time_ns (system clock)")
print(f"  ✓ ntp_timestamp_ns (NTP corrected)")
print(f"  ✗ ct_timestamp_ns (recorded but using fallback offset=0)")
print()

# Calculate latencies for Worker B
merged_b['latency_system_ns'] = merged_b['receive_time_ns'] - merged_b['send_time_ns']
merged_b['latency_system_ms'] = merged_b['latency_system_ns'] / 1_000_000

# What was actually compared in the "100% violations" result
merged_b['coord_system_to_worker_ntp_ns'] = merged_b['ntp_timestamp_ns'] - merged_b['send_time_ns']
merged_b['coord_system_to_worker_ntp_ms'] = merged_b['coord_system_to_worker_ntp_ns'] / 1_000_000

violations = (merged_b['coord_system_to_worker_ntp_ms'] < 0).sum()
total = len(merged_b)

print("THE PROBLEM:")
print("-" * 60)
print(f"When we compared:")
print(f"  Coordinator system clock  vs  Worker NTP timestamp")
print(f"  (uncorrected)                 (NTP-corrected)")
print()
print(f"Violations: {violations}/{total} ({100*violations/total:.1f}%)")
print()
print("This is UNFAIR because:")
print("  - Coordinator system clock is ~4.5ms AHEAD of workers")
print("  - Worker NTP pulls timestamps BACK by -4.57ms")
print("  - Result: Worker NTP timestamp appears BEFORE coordinator send time")
print("  - This violates causality artificially!")
print()

print("WHAT WE NEED:")
print("-" * 60)
print("Ground truth: Both query NTP to establish 'true' ordering")
print("  coord_ntp_send < worker_ntp_recv  (physics truth)")
print()
print("Then compare:")
print("  1. System clock: coord_system vs worker_system")
print("     Does raw system clock respect ground truth ordering?")
print()
print("  2. ChronoTick: coord_ct_bounds vs worker_ct_bounds  ")
print("     Do uncertainty bounds respect ground truth ordering?")
print()

# Show sample data
print("SAMPLE DATA (first 3 events, Worker B):")
print("-" * 60)
for i in range(min(3, len(merged_b))):
    row = merged_b.iloc[i]
    print(f"\nEvent {row['event_id']}:")
    print(f"  Coordinator send (system):  {row['send_time_ns']} ns")
    print(f"  Worker receive (system):    {row['receive_time_ns']} ns")
    print(f"  Worker receive (NTP):       {row['ntp_timestamp_ns']} ns")
    print(f"  System latency:             {row['latency_system_ms']:.3f} ms")
    print(f"  Unfair comparison delta:    {row['coord_system_to_worker_ntp_ms']:.3f} ms")
    print(f"    → {'VIOLATION!' if row['coord_system_to_worker_ntp_ms'] < 0 else 'OK'}")

# Create visualization
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: System clock latencies (should be positive)
axes[0].hist(merged_b['latency_system_ms'], bins=50, alpha=0.7, edgecolor='black')
axes[0].axvline(0, color='red', linestyle='--', linewidth=2, label='Causality boundary')
axes[0].set_xlabel('Latency (ms)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('System Clock: recv_time - send_time (CORRECT comparison)', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Plot 2: Unfair comparison (coord system vs worker NTP)
axes[1].hist(merged_b['coord_system_to_worker_ntp_ms'], bins=50, alpha=0.7, color='orange', edgecolor='black')
axes[1].axvline(0, color='red', linestyle='--', linewidth=2, label='Causality boundary')
axes[1].set_xlabel('Delta (ms)', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('UNFAIR: worker_ntp - coord_system (what analysis did)', fontsize=14, fontweight='bold', color='red')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/production-run-1/diagnosis.png', dpi=150, bbox_inches='tight')
print(f"\n\nVisualization saved to: results/production-run-1/diagnosis.png")
print()
