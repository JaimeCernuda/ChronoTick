#!/usr/bin/env python3
"""
Diagnostic script to understand ChronoTick offset semantics.

Hypothesis: ChronoTick offset has opposite sign from NTP offset.
"""

import pandas as pd
import numpy as np

# Load ChronoTick data
print("Loading ChronoTick worker data...")
df = pd.read_csv("results/chronotick_30min_20251025-115922/worker_comp11.csv")

print(f"\nLoaded {len(df)} events\n")
print("="*80)

# Show first and last 5 rows with key fields
print("FIRST 5 EVENTS:")
print(df[['event_id', 'ntp_offset_ms', 'ct_offset_ms']].head())

print("\nLAST 5 EVENTS:")
print(df[['event_id', 'ntp_offset_ms', 'ct_offset_ms']].tail())

print("\n" + "="*80)
print("HYPOTHESIS TESTING")
print("="*80)

# Calculate errors under different interpretations
print("\n1. CURRENT INTERPRETATION (ct_offset_ms used as-is):")
df['error_current'] = df['ct_offset_ms'] - df['ntp_offset_ms']
print(f"   Mean error: {df['error_current'].mean():.3f} ms")
print(f"   Median error: {df['error_current'].median():.3f} ms")
print(f"   MAE: {abs(df['error_current']).mean():.3f} ms")
print(f"   RMSE: {np.sqrt((df['error_current']**2).mean()):.3f} ms")

print("\n2. SIGN FLIP HYPOTHESIS (ct_offset_ms negated):")
df['ct_offset_negated'] = -df['ct_offset_ms']
df['error_negated'] = df['ct_offset_negated'] - df['ntp_offset_ms']
print(f"   Mean error: {df['error_negated'].mean():.3f} ms")
print(f"   Median error: {df['error_negated'].median():.3f} ms")
print(f"   MAE: {abs(df['error_negated']).mean():.3f} ms")
print(f"   RMSE: {np.sqrt((df['error_negated']**2).mean()):.3f} ms")

print("\n" + "="*80)
print("OFFSET STATISTICS")
print("="*80)

print("\nNTP offset:")
print(f"   Mean: {df['ntp_offset_ms'].mean():.3f} ms")
print(f"   Median: {df['ntp_offset_ms'].median():.3f} ms")
print(f"   Range: [{df['ntp_offset_ms'].min():.3f}, {df['ntp_offset_ms'].max():.3f}] ms")

print("\nChronoTick offset (as-is):")
print(f"   Mean: {df['ct_offset_ms'].mean():.3f} ms")
print(f"   Median: {df['ct_offset_ms'].median():.3f} ms")
print(f"   Range: [{df['ct_offset_ms'].min():.3f}, {df['ct_offset_ms'].max():.3f}] ms")

print("\nChronoTick offset (negated):")
print(f"   Mean: {-df['ct_offset_ms'].mean():.3f} ms")
print(f"   Median: {-df['ct_offset_ms'].median():.3f} ms")
print(f"   Range: [{-df['ct_offset_ms'].max():.3f}, {-df['ct_offset_ms'].min():.3f}] ms")

print("\n" + "="*80)
print("VERDICT")
print("="*80)

current_mae = abs(df['error_current']).mean()
negated_mae = abs(df['error_negated']).mean()

print(f"\nCurrent interpretation MAE: {current_mae:.3f} ms")
print(f"Negated interpretation MAE: {negated_mae:.3f} ms")

if negated_mae < current_mae:
    improvement = ((current_mae - negated_mae) / current_mae) * 100
    print(f"\n✓ SIGN FLIP HYPOTHESIS CONFIRMED!")
    print(f"  Negating ChronoTick offset improves MAE by {improvement:.1f}%")
    print(f"  ({current_mae:.3f}ms → {negated_mae:.3f}ms)")
else:
    print(f"\n✗ SIGN FLIP HYPOTHESIS REJECTED")
    print(f"  Current interpretation is better")

print("\n" + "="*80)
