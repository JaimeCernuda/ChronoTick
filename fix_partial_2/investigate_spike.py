#!/usr/bin/env python3
"""
ULTRATHINK: What caused the homelab 58-minute spike?
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime, timedelta

data_dir = Path(__file__).parent

print("="*80)
print("ULTRATHINKING: HOMELAB 58-MINUTE SPIKE ROOT CAUSE ANALYSIS")
print("="*80)

# Load data
homelab_csv = pd.read_csv(data_dir / 'homelab' / 'chronotick_client_validation_20251022_094657.csv')
homelab_log = data_dir / 'homelab' / 'client_validation_20251022_094656.log'

# Test started at 09:46, so spike at 58min = 10:44
test_start = "09:46:00"
spike_time_offset = 58 * 60  # seconds
spike_time_str = "10:44:00"

print(f"\nTest started: {test_start}")
print(f"Spike time: {spike_time_str} (58 minutes after start)")

# ============================================================================
# 1. EXAMINE THE SPIKE REGION IN DETAIL
# ============================================================================
print("\n" + "="*80)
print("1. DETAILED SPIKE REGION ANALYSIS")
print("="*80)

# Get spike region: 56-60 minutes
spike_region = homelab_csv[
    (homelab_csv['elapsed_seconds'] >= 56*60) &
    (homelab_csv['elapsed_seconds'] <= 60*60)
].copy()

print(f"\nSamples in spike region (56-60 min): {len(spike_region)}")
print(f"Sample numbers: {spike_region['sample_number'].min():.0f} to {spike_region['sample_number'].max():.0f}")

# Show the actual spike samples
print("\nTop 20 samples by offset:")
spike_sorted = spike_region.sort_values('chronotick_offset_ms', ascending=False).head(20)
for idx, row in spike_sorted.iterrows():
    print(f"  Sample {int(row['sample_number']):4d} @ {row['elapsed_seconds']/60:6.2f}min: "
          f"{row['chronotick_offset_ms']:7.2f}ms, source={row['chronotick_source']:8s}, "
          f"conf={row['chronotick_confidence']:.2f}")

# ============================================================================
# 2. CHECK NTP GROUND TRUTH DURING SPIKE
# ============================================================================
print("\n" + "="*80)
print("2. NTP GROUND TRUTH DURING SPIKE")
print("="*80)

# Get NTP measurements around spike time (54-62 minutes)
ntp_around_spike = homelab_csv[
    (homelab_csv['has_ntp'] == True) &
    (homelab_csv['elapsed_seconds'] >= 54*60) &
    (homelab_csv['elapsed_seconds'] <= 62*60)
]

print(f"\nNTP measurements around spike (54-62 min): {len(ntp_around_spike)}")
for idx, row in ntp_around_spike.iterrows():
    print(f"  Sample {int(row['sample_number']):4d} @ {row['elapsed_seconds']/60:6.2f}min: "
          f"NTP offset={row['ntp_offset_ms']:7.2f}ms, "
          f"ChronoTick offset={row['chronotick_offset_ms']:7.2f}ms, "
          f"Error={row['chronotick_offset_ms'] - row['ntp_offset_ms']:7.2f}ms")

# Check if NTP baseline shifted
ntp_before = homelab_csv[
    (homelab_csv['has_ntp'] == True) &
    (homelab_csv['elapsed_seconds'] < 56*60)
]['ntp_offset_ms']

ntp_during = homelab_csv[
    (homelab_csv['has_ntp'] == True) &
    (homelab_csv['elapsed_seconds'] >= 56*60) &
    (homelab_csv['elapsed_seconds'] <= 60*60)
]['ntp_offset_ms']

ntp_after = homelab_csv[
    (homelab_csv['has_ntp'] == True) &
    (homelab_csv['elapsed_seconds'] > 60*60) &
    (homelab_csv['elapsed_seconds'] <= 80*60)
]['ntp_offset_ms']

if len(ntp_before) > 0 and len(ntp_during) > 0:
    print(f"\n*** NTP BASELINE SHIFT ANALYSIS ***")
    print(f"NTP offset before spike (0-56 min): mean={ntp_before.mean():.3f}ms, std={ntp_before.std():.3f}ms")
    print(f"NTP offset during spike (56-60 min): mean={ntp_during.mean():.3f}ms, std={ntp_during.std():.3f}ms")
    if len(ntp_after) > 0:
        print(f"NTP offset after spike (60-80 min): mean={ntp_after.mean():.3f}ms, std={ntp_after.std():.3f}ms")

    baseline_shift = ntp_during.mean() - ntp_before.mean()
    print(f"\n*** BASELINE SHIFT: {baseline_shift:.3f}ms ***")

# ============================================================================
# 3. PARSE LOGS AROUND SPIKE TIME
# ============================================================================
print("\n" + "="*80)
print("3. LOG EVENTS AROUND SPIKE TIME (10:42 - 10:46)")
print("="*80)

spike_log_events = []
with open(homelab_log, 'r') as f:
    for line in f:
        # Parse timestamp
        timestamp_match = re.match(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
        if timestamp_match:
            timestamp_str = timestamp_match.group(1)
            # Check if in spike window (10:42 - 10:46)
            if '10:42:' in timestamp_str or '10:43:' in timestamp_str or '10:44:' in timestamp_str or '10:45:' in timestamp_str or '10:46:' in timestamp_str:
                # Look for interesting events
                if any(keyword in line for keyword in [
                    'FUSION HAPPENING',
                    'NTP',
                    'ERROR',
                    'CRITICAL',
                    'WARNING',
                    'Dataset size',
                    'GPU',
                    'CPU',
                    'Fusion weights',
                    'offset='
                ]):
                    spike_log_events.append(line.strip())

print(f"Found {len(spike_log_events)} relevant log events")
if len(spike_log_events) > 0:
    print("\nShowing first 50 events:")
    for event in spike_log_events[:50]:
        print(f"  {event[:150]}")  # Truncate long lines

# ============================================================================
# 4. CHECK IF THIS CORRELATES WITH DATASET/MODEL EVENTS
# ============================================================================
print("\n" + "="*80)
print("4. DATASET AND MODEL EVENTS")
print("="*80)

# Parse for dataset size changes
dataset_events = []
with open(homelab_log, 'r') as f:
    for line in f:
        if 'Dataset size:' in line and ('10:42:' in line or '10:43:' in line or '10:44:' in line or '10:45:' in line):
            dataset_events.append(line.strip())

if len(dataset_events) > 0:
    print(f"Dataset events around spike: {len(dataset_events)}")
    print("Sample events:")
    for event in dataset_events[:10]:
        print(f"  {event}")

# Check for NTP integration events
ntp_integration = []
with open(homelab_log, 'r') as f:
    for line in f:
        if '✓ Dataset now has 2 total measurements' in line and ('10:42:' in line or '10:43:' in line or '10:44:' in line or '10:45:' in line):
            ntp_integration.append(line.strip())

if len(ntp_integration) > 0:
    print(f"\nNTP integration events: {len(ntp_integration)}")
    for event in ntp_integration[:5]:
        print(f"  {event}")

# ============================================================================
# 5. CALCULATE ERROR RELATIVE TO NTP
# ============================================================================
print("\n" + "="*80)
print("5. CHRONOTICK ERROR RELATIVE TO NTP GROUND TRUTH")
print("="*80)

# For all samples with NTP, calculate ChronoTick error
ntp_samples = homelab_csv[homelab_csv['has_ntp'] == True].copy()
ntp_samples['chronotick_error'] = ntp_samples['chronotick_offset_ms'] - ntp_samples['ntp_offset_ms']

# Split by time periods
before_spike = ntp_samples[ntp_samples['elapsed_seconds'] < 56*60]['chronotick_error']
during_spike = ntp_samples[
    (ntp_samples['elapsed_seconds'] >= 56*60) &
    (ntp_samples['elapsed_seconds'] <= 60*60)
]['chronotick_error']
after_spike = ntp_samples[ntp_samples['elapsed_seconds'] > 60*60]['chronotick_error']

print(f"\nChronoTick ERROR (vs NTP ground truth):")
print(f"Before spike (0-56 min): mean={before_spike.mean():.3f}ms, std={before_spike.std():.3f}ms")
if len(during_spike) > 0:
    print(f"During spike (56-60 min): mean={during_spike.mean():.3f}ms, std={during_spike.std():.3f}ms")
print(f"After spike (60+ min): mean={after_spike.mean():.3f}ms, std={after_spike.std():.3f}ms")

# ============================================================================
# 6. ROOT CAUSE DETERMINATION
# ============================================================================
print("\n" + "="*80)
print("6. ROOT CAUSE DETERMINATION")
print("="*80)

# Check what changed
chronotick_before = homelab_csv[homelab_csv['elapsed_seconds'] < 56*60]['chronotick_offset_ms']
chronotick_during = homelab_csv[
    (homelab_csv['elapsed_seconds'] >= 56*60) &
    (homelab_csv['elapsed_seconds'] <= 60*60)
]['chronotick_offset_ms']
chronotick_after = homelab_csv[
    (homelab_csv['elapsed_seconds'] > 60*60) &
    (homelab_csv['elapsed_seconds'] <= 80*60)
]['chronotick_offset_ms']

print(f"\nChronoTick offset statistics:")
print(f"Before spike: mean={chronotick_before.mean():.3f}ms, std={chronotick_before.std():.3f}ms")
print(f"During spike: mean={chronotick_during.mean():.3f}ms, std={chronotick_during.std():.3f}ms")
print(f"After spike:  mean={chronotick_after.mean():.3f}ms, std={chronotick_after.std():.3f}ms")

chronotick_shift = chronotick_during.mean() - chronotick_before.mean()
print(f"\n*** ChronoTick SHIFT: {chronotick_shift:.3f}ms ***")

# HYPOTHESIS TESTING
print("\n" + "="*80)
print("HYPOTHESIS TESTING")
print("="*80)

print("\nHypothesis 1: System clock jumped (thermal throttling, NTP correction)")
if len(ntp_during) > 0:
    if abs(baseline_shift) > 1.0:
        print(f"  ✓ LIKELY: NTP baseline shifted by {baseline_shift:.2f}ms")
        print(f"  → System clock probably had a step change")
    else:
        print(f"  ✗ UNLIKELY: NTP baseline only shifted by {baseline_shift:.2f}ms")

print("\nHypothesis 2: ChronoTick prediction error (bad ML prediction)")
if len(during_spike) > 0:
    if abs(during_spike.mean()) > abs(before_spike.mean()) + 1.0:
        print(f"  ✓ LIKELY: ChronoTick error increased from {before_spike.mean():.2f}ms to {during_spike.mean():.2f}ms")
        print(f"  → ML models made bad predictions")
    else:
        print(f"  ✗ UNLIKELY: ChronoTick error relatively stable")

print("\nHypothesis 3: Both system clock AND ChronoTick shifted together")
if abs(chronotick_shift) > 3.0 and abs(baseline_shift) < 1.0:
    print(f"  ✓ LIKELY: ChronoTick shifted {chronotick_shift:.2f}ms but NTP only {baseline_shift:.2f}ms")
    print(f"  → This suggests ChronoTick CAUSED the spike, not the system clock")
else:
    print(f"  ? INCONCLUSIVE")

# ============================================================================
# FINAL VERDICT
# ============================================================================
print("\n" + "="*80)
print("FINAL VERDICT: ROOT CAUSE IDENTIFIED")
print("="*80)

if len(ntp_during) > 0 and len(during_spike) > 0:
    # The smoking gun: compare what changed
    print(f"\nSUMMARY:")
    print(f"  ChronoTick offset:  {chronotick_before.mean():.2f}ms → {chronotick_during.mean():.2f}ms (Δ={chronotick_shift:.2f}ms)")
    print(f"  NTP ground truth:   {ntp_before.mean():.2f}ms → {ntp_during.mean():.2f}ms (Δ={baseline_shift:.2f}ms)")
    print(f"  ChronoTick error:   {before_spike.mean():.2f}ms → {during_spike.mean():.2f}ms")

    error_increase = during_spike.mean() - before_spike.mean()

    print(f"\n*** ROOT CAUSE ***")
    if abs(baseline_shift) > 3.0:
        print(f"  PRIMARY: System clock jumped by {baseline_shift:.2f}ms (NTP baseline shift)")
        print(f"  → Likely ntpd step adjustment or thermal throttling")
        if abs(error_increase) > 1.0:
            print(f"  SECONDARY: ChronoTick prediction error increased by {error_increase:.2f}ms")
            print(f"  → ML models hadn't adapted to new baseline yet")
    elif abs(error_increase) > 3.0:
        print(f"  PRIMARY: ChronoTick ML prediction error")
        print(f"  → Models made bad predictions (error increased by {error_increase:.2f}ms)")
        print(f"  → Possible causes: bad training data, model update, dataset issue")
    else:
        print(f"  UNCLEAR: Both shifts are small")
        print(f"  → Spike might be measurement noise or transient event")

print("\n" + "="*80)
