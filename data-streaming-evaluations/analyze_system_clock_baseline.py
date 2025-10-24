#!/usr/bin/env python3
"""
Analysis: System Clock vs NTP Ground Truth
Compares raw system clock ordering with NTP-established ground truth
"""

import pandas as pd
import sys

def analyze_baseline(experiment_dir):
    # Load data
    coord = pd.read_csv(f"{experiment_dir}/coordinator.csv")
    worker_b = pd.read_csv(f"{experiment_dir}/worker_comp11.csv")
    worker_c = pd.read_csv(f"{experiment_dir}/worker_comp12.csv")

    print("=" * 80)
    print("SYSTEM CLOCK BASELINE: Raw System Clock vs NTP Ground Truth")
    print("=" * 80)
    print()

    # Basic stats
    print(f"Events sent: {len(coord)}")
    print(f"Worker B received: {len(worker_b)}")
    print(f"Worker C received: {len(worker_c)}")
    print()

    # Analyze Worker B
    print("=" * 80)
    print("WORKER B (comp11) ANALYSIS")
    print("=" * 80)

    # Merge coordinator and worker data
    merged_b = worker_b.merge(coord, on='event_id', suffixes=('_worker', '_coord'))

    # Calculate latencies
    merged_b['system_latency_ns'] = merged_b['receive_time_ns'] - merged_b['send_time_ns']
    merged_b['system_latency_ms'] = merged_b['system_latency_ns'] / 1e6

    merged_b['ntp_latency_ns'] = merged_b['ntp_timestamp_ns_worker'] - merged_b['ntp_timestamp_ns_coord']
    merged_b['ntp_latency_ms'] = merged_b['ntp_latency_ns'] / 1e6

    print(f"\nSystem Clock Latency (recv_system - send_system):")
    print(f"  Mean: {merged_b['system_latency_ms'].mean():.3f} ms")
    print(f"  Std:  {merged_b['system_latency_ms'].std():.3f} ms")
    print(f"  Min:  {merged_b['system_latency_ms'].min():.3f} ms")
    print(f"  Max:  {merged_b['system_latency_ms'].max():.3f} ms")

    print(f"\nNTP Ground Truth Latency (recv_ntp - send_ntp):")
    print(f"  Mean: {merged_b['ntp_latency_ms'].mean():.3f} ms")
    print(f"  Std:  {merged_b['ntp_latency_ms'].std():.3f} ms")
    print(f"  Min:  {merged_b['ntp_latency_ms'].min():.3f} ms")
    print(f"  Max:  {merged_b['ntp_latency_ms'].max():.3f} ms")

    # Causality violations
    system_violations_b = (merged_b['system_latency_ns'] < 0).sum()
    ntp_violations_b = (merged_b['ntp_latency_ns'] < 0).sum()

    print(f"\nCausality Violations (negative latency):")
    print(f"  System Clock: {system_violations_b}/{len(merged_b)} ({100*system_violations_b/len(merged_b):.1f}%)")
    print(f"  NTP Ground Truth: {ntp_violations_b}/{len(merged_b)} ({100*ntp_violations_b/len(merged_b):.1f}%)")

    # Analyze Worker C
    print()
    print("=" * 80)
    print("WORKER C (comp12) ANALYSIS")
    print("=" * 80)

    merged_c = worker_c.merge(coord, on='event_id', suffixes=('_worker', '_coord'))

    merged_c['system_latency_ns'] = merged_c['receive_time_ns'] - merged_c['send_time_ns']
    merged_c['system_latency_ms'] = merged_c['system_latency_ns'] / 1e6

    merged_c['ntp_latency_ns'] = merged_c['ntp_timestamp_ns_worker'] - merged_c['ntp_timestamp_ns_coord']
    merged_c['ntp_latency_ms'] = merged_c['ntp_latency_ns'] / 1e6

    print(f"\nSystem Clock Latency (recv_system - send_system):")
    print(f"  Mean: {merged_c['system_latency_ms'].mean():.3f} ms")
    print(f"  Std:  {merged_c['system_latency_ms'].std():.3f} ms")
    print(f"  Min:  {merged_c['system_latency_ms'].min():.3f} ms")
    print(f"  Max:  {merged_c['system_latency_ms'].max():.3f} ms")

    print(f"\nNTP Ground Truth Latency (recv_ntp - send_ntp):")
    print(f"  Mean: {merged_c['ntp_latency_ms'].mean():.3f} ms")
    print(f"  Std:  {merged_c['ntp_latency_ms'].std():.3f} ms")
    print(f"  Min:  {merged_c['ntp_latency_ms'].min():.3f} ms")
    print(f"  Max:  {merged_c['ntp_latency_ms'].max():.3f} ms")

    system_violations_c = (merged_c['system_latency_ns'] < 0).sum()
    ntp_violations_c = (merged_c['ntp_latency_ns'] < 0).sum()

    print(f"\nCausality Violations (negative latency):")
    print(f"  System Clock: {system_violations_c}/{len(merged_c)} ({100*system_violations_c/len(merged_c):.1f}%)")
    print(f"  NTP Ground Truth: {ntp_violations_c}/{len(merged_c)} ({100*ntp_violations_c/len(merged_c):.1f}%)")

    # Summary
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_events = len(merged_b) + len(merged_c)
    total_system_violations = system_violations_b + system_violations_c
    total_ntp_violations = ntp_violations_b + ntp_violations_c

    print(f"\nTotal events analyzed: {total_events}")
    print(f"\nSystem Clock violations: {total_system_violations}/{total_events} ({100*total_system_violations/total_events:.2f}%)")
    print(f"NTP Ground Truth violations: {total_ntp_violations}/{total_events} ({100*total_ntp_violations/total_events:.2f}%)")

    print()
    print("INTERPRETATION:")
    print("- NTP establishes ground truth: coordinator sent BEFORE worker received")
    print("- System clock violations: times when raw system clock disagrees with NTP")
    print("- NTP should have 0% violations (it's the reference!)")
    print("- System clock violations show when unsynchronized clocks cause errors")
    print()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python analyze_system_clock_baseline.py <experiment_dir>")
        sys.exit(1)

    analyze_baseline(sys.argv[1])
