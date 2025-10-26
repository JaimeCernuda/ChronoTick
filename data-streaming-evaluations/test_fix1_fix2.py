#!/usr/bin/env python3
"""
Test Fix 1 and Fix 2 approaches from validation_client_v3.

Based on experiment-13 implementation that achieved sub-ms MAE.
"""

import pandas as pd
import numpy as np

# Load ChronoTick data
print("Loading ChronoTick worker data...")
df = pd.read_csv("results/chronotick_30min_20251025-115922/worker_comp11.csv")

print(f"\nLoaded {len(df)} events\n")
print("="*80)

# Current approach (WRONG)
print("CURRENT APPROACH (what we're doing now):")
print("  ct_timestamp = receive_time + ct_offset_ms")
print()

# Calculate current ChronoTick timestamps (reconstructed from CSV)
# Note: CSV has ct_timestamp_ns already, but let's verify
df['current_ct_time_s'] = df['receive_time_ns'] / 1e9 + df['ct_offset_ms'] / 1000.0
df['current_error_ms'] = (df['current_ct_time_s'] - (df['receive_time_ns'] / 1e9 + df['ntp_offset_ms'] / 1000.0)) * 1000

print(f"Current approach MAE: {abs(df['current_error_ms']).mean():.3f} ms")
print(f"Current approach RMSE: {np.sqrt((df['current_error_ms']**2).mean()):.3f} ms")
print()

# FIX 1: Use drift_rate and prediction_time
print("="*80)
print("FIX 1: System-based with drift correction")
print("  time_delta = receive_time - prediction_time")
print("  ct_time = receive_time + offset + drift_rate * time_delta")
print()

# Calculate time_delta (receive_time - prediction_time)
df['time_delta_s'] = df['receive_time_ns'] / 1e9 - df['ct_prediction_time']

# Calculate Fix 1 timestamp
df['fix1_ct_time_s'] = (df['receive_time_ns'] / 1e9 +
                         df['ct_offset_ms'] / 1000.0 +
                         df['ct_drift_rate'] * df['time_delta_s'])

# Calculate NTP reference time
df['ntp_true_time_s'] = df['receive_time_ns'] / 1e9 + df['ntp_offset_ms'] / 1000.0

# Calculate Fix 1 error
df['fix1_error_ms'] = (df['fix1_ct_time_s'] - df['ntp_true_time_s']) * 1000

print(f"Fix 1 MAE: {abs(df['fix1_error_ms']).mean():.3f} ms")
print(f"Fix 1 RMSE: {np.sqrt((df['fix1_error_ms']**2).mean()):.3f} ms")
print()
print(f"Improvement over current: {abs(df['current_error_ms']).mean() - abs(df['fix1_error_ms']).mean():.3f} ms")
print()

# FIX 2: NTP-anchored time walking
print("="*80)
print("FIX 2: NTP-anchored time walking (chrony-inspired)")
print("  elapsed_since_ntp = receive_time - last_ntp_system_time")
print("  ct_time = last_ntp_true_time + elapsed + drift_rate * elapsed")
print()

# We need to simulate the NTP anchor tracking
# For each event, find the most recent NTP measurement

# First, identify events where we have NTP data
df['has_ntp_data'] = df['ntp_offset_ms'].notna()

# Initialize columns for Fix 2
df['last_ntp_true_time_s'] = np.nan
df['last_ntp_system_time_s'] = np.nan
df['elapsed_since_ntp_s'] = np.nan
df['fix2_ct_time_s'] = np.nan

# Track the last NTP anchor as we iterate
last_ntp_true_time = None
last_ntp_system_time = None

for idx in df.index:
    receive_time_s = df.loc[idx, 'receive_time_ns'] / 1e9

    # Update NTP anchor if this event has NTP data
    if df.loc[idx, 'has_ntp_data']:
        ntp_offset_ms = df.loc[idx, 'ntp_offset_ms']
        last_ntp_true_time = receive_time_s + ntp_offset_ms / 1000.0
        last_ntp_system_time = receive_time_s

    # Calculate Fix 2 timestamp if we have an NTP anchor
    if last_ntp_true_time is not None and last_ntp_system_time is not None:
        elapsed_since_ntp = receive_time_s - last_ntp_system_time
        drift_rate = df.loc[idx, 'ct_drift_rate']

        fix2_time = last_ntp_true_time + elapsed_since_ntp + drift_rate * elapsed_since_ntp

        df.loc[idx, 'last_ntp_true_time_s'] = last_ntp_true_time
        df.loc[idx, 'last_ntp_system_time_s'] = last_ntp_system_time
        df.loc[idx, 'elapsed_since_ntp_s'] = elapsed_since_ntp
        df.loc[idx, 'fix2_ct_time_s'] = fix2_time

# Calculate Fix 2 error (only for events where we have both Fix 2 and NTP reference)
df['fix2_error_ms'] = (df['fix2_ct_time_s'] - df['ntp_true_time_s']) * 1000

# Calculate MAE/RMSE (dropping NaN values)
fix2_errors = df['fix2_error_ms'].dropna()

print(f"Fix 2 MAE: {abs(fix2_errors).mean():.3f} ms")
print(f"Fix 2 RMSE: {np.sqrt((fix2_errors**2).mean()):.3f} ms")
print(f"Events with Fix 2: {len(fix2_errors)} / {len(df)}")
print()
print(f"Improvement over current: {abs(df['current_error_ms']).mean() - abs(fix2_errors).mean():.3f} ms")
print()

# Compare all three approaches
print("="*80)
print("SUMMARY - Mean Absolute Error (MAE)")
print("="*80)
print(f"Current approach:  {abs(df['current_error_ms']).mean():>8.3f} ms")
print(f"Fix 1 (drift):     {abs(df['fix1_error_ms']).mean():>8.3f} ms")
print(f"Fix 2 (NTP-walk):  {abs(fix2_errors).mean():>8.3f} ms")
print()

# Show which is best
approaches = {
    'Current': abs(df['current_error_ms']).mean(),
    'Fix 1': abs(df['fix1_error_ms']).mean(),
    'Fix 2': abs(fix2_errors).mean()
}

best_approach = min(approaches, key=approaches.get)
print(f"Best approach: {best_approach} ({approaches[best_approach]:.3f} ms MAE)")

# Show a few sample calculations
print()
print("="*80)
print("SAMPLE EVENTS (first 10 with NTP data)")
print("="*80)
sample = df[df['has_ntp_data']].head(10)

for idx in sample.index:
    print(f"\nEvent {idx}:")
    print(f"  Receive time: {df.loc[idx, 'receive_time_ns'] / 1e9:.6f} s")
    print(f"  NTP offset: {df.loc[idx, 'ntp_offset_ms']:.3f} ms")
    print(f"  NTP true time: {df.loc[idx, 'ntp_true_time_s']:.6f} s")
    print(f"  ChronoTick offset: {df.loc[idx, 'ct_offset_ms']:.3f} ms")
    print(f"  ChronoTick drift: {df.loc[idx, 'ct_drift_rate']:.9f} s/s")
    print(f"  Time delta: {df.loc[idx, 'time_delta_s']:.6f} s")
    print(f"  Current error: {df.loc[idx, 'current_error_ms']:>+8.3f} ms")
    print(f"  Fix 1 error:   {df.loc[idx, 'fix1_error_ms']:>+8.3f} ms")
    if not pd.isna(df.loc[idx, 'fix2_error_ms']):
        print(f"  Fix 2 error:   {df.loc[idx, 'fix2_error_ms']:>+8.3f} ms")
