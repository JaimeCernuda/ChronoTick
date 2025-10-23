#!/usr/bin/env python3
"""
Deep analysis of ChronoTick experiments
Investigates: outliers, spikes, fusion rates, NTP rejection, error metrics
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import re

data_dir = Path(__file__).parent

print("="*80)
print("DEEP ANALYSIS: INVESTIGATING ISSUES")
print("="*80)

# Load all data
homelab_csv = pd.read_csv(data_dir / 'homelab' / 'chronotick_client_validation_20251022_094657.csv')
ares11_csv = pd.read_csv(data_dir / 'ares-11' / 'chronotick_client_validation_20251022_105514.csv')
ares12_csv = pd.read_csv(data_dir / 'ares-12' / 'chronotick_client_validation_20251022_105732.csv')

# ============================================================================
# 1. ARES-12 NTP OUTLIERS
# ============================================================================
print("\n" + "="*80)
print("1. INVESTIGATING ARES-12 NTP OUTLIERS")
print("="*80)

ares12_ntp = ares12_csv[ares12_csv['has_ntp'] == True].copy()
print(f"Total NTP measurements: {len(ares12_ntp)}")
print(f"\nNTP offset statistics:")
print(f"  Mean: {ares12_ntp['ntp_offset_ms'].mean():.3f} ms")
print(f"  Median: {ares12_ntp['ntp_offset_ms'].median():.3f} ms")
print(f"  Std: {ares12_ntp['ntp_offset_ms'].std():.3f} ms")
print(f"  Min: {ares12_ntp['ntp_offset_ms'].min():.3f} ms")
print(f"  Max: {ares12_ntp['ntp_offset_ms'].max():.3f} ms")

# Find outliers (beyond 3 sigma)
mean_ntp = ares12_ntp['ntp_offset_ms'].mean()
std_ntp = ares12_ntp['ntp_offset_ms'].std()
outliers = ares12_ntp[np.abs(ares12_ntp['ntp_offset_ms'] - mean_ntp) > 3 * std_ntp]

print(f"\nOutliers (>3σ from mean): {len(outliers)}")
if len(outliers) > 0:
    print("\nOutlier details:")
    for idx, row in outliers.iterrows():
        print(f"  Sample {int(row['sample_number'])}: {row['ntp_offset_ms']:.2f}ms at time {row['elapsed_seconds']/60:.1f}min")

# Filter for reasonable visualization
ares12_ntp_filtered = ares12_ntp[np.abs(ares12_ntp['ntp_offset_ms']) < 20].copy()
print(f"\nFiltered NTP (|offset| < 20ms): {len(ares12_ntp_filtered)} / {len(ares12_ntp)}")

# ============================================================================
# 2. HOMELAB CHRONOTICK SPIKES
# ============================================================================
print("\n" + "="*80)
print("2. INVESTIGATING HOMELAB CHRONOTICK SPIKES")
print("="*80)

homelab_mean = homelab_csv['chronotick_offset_ms'].mean()
homelab_std = homelab_csv['chronotick_offset_ms'].std()
print(f"ChronoTick offset: mean={homelab_mean:.3f}ms, std={homelab_std:.3f}ms")

# Find spikes (beyond 2 sigma)
spikes = homelab_csv[np.abs(homelab_csv['chronotick_offset_ms'] - homelab_mean) > 2 * homelab_std]
print(f"\nSpikes (>2σ from mean): {len(spikes)} / {len(homelab_csv)} ({100*len(spikes)/len(homelab_csv):.1f}%)")

print("\nTop 10 largest deviations:")
spikes_sorted = homelab_csv.iloc[(homelab_csv['chronotick_offset_ms'] - homelab_mean).abs().argsort()[::-1]]
for idx in range(min(10, len(spikes_sorted))):
    row = spikes_sorted.iloc[idx]
    deviation = row['chronotick_offset_ms'] - homelab_mean
    print(f"  Sample {int(row['sample_number'])}: {row['chronotick_offset_ms']:.2f}ms "
          f"(Δ={deviation:.2f}ms) at {row['elapsed_seconds']/60:.1f}min, source={row['chronotick_source']}")

# Check if spikes correlate with source type
print("\nSpike correlation with source:")
for source in ['cpu', 'fusion', 'error']:
    source_data = homelab_csv[homelab_csv['chronotick_source'] == source]
    if len(source_data) > 0:
        source_mean = source_data['chronotick_offset_ms'].mean()
        source_std = source_data['chronotick_offset_ms'].std()
        print(f"  {source}: mean={source_mean:.3f}ms, std={source_std:.3f}ms, count={len(source_data)}")

# ============================================================================
# 3. FUSION RATE INVESTIGATION
# ============================================================================
print("\n" + "="*80)
print("3. WHY IS FUSION ~80% NOT 100%?")
print("="*80)

for name, df in [('Homelab', homelab_csv), ('ARES-11', ares11_csv), ('ARES-12', ares12_csv)]:
    sources = df['chronotick_source'].value_counts()
    total = len(df)
    print(f"\n{name}:")
    for source, count in sources.items():
        pct = 100 * count / total
        print(f"  {source}: {count} ({pct:.1f}%)")

    # Check if CPU-only predictions happen at specific times
    cpu_only = df[df['chronotick_source'] == 'cpu']
    if len(cpu_only) > 0:
        print(f"  CPU-only samples: {cpu_only['sample_number'].min():.0f} to {cpu_only['sample_number'].max():.0f}")

        # Check if they're clustered or distributed
        cpu_samples = cpu_only['sample_number'].values
        if len(cpu_samples) > 1:
            gaps = np.diff(cpu_samples)
            print(f"  Gap pattern: mean={gaps.mean():.1f}, std={gaps.std():.1f}, max={gaps.max():.0f}")

            # If gaps are small, they're clustered; if large, they're distributed
            if gaps.mean() < 5:
                print(f"  ✓ CPU predictions are CLUSTERED (likely during GPU model updates)")
            else:
                print(f"  ✓ CPU predictions are DISTRIBUTED (likely alternating pattern)")

# ============================================================================
# 4. NTP REJECTION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("4. WHY 15% AND 11% NTP REJECTION?")
print("="*80)

# Parse homelab log for NTP rejections
homelab_log = data_dir / 'homelab' / 'client_validation_20251022_094656.log'
ares11_log = data_dir / 'ares-11' / 'client_validation_20251022_105511.log'

def analyze_ntp_rejections(log_path, name):
    print(f"\n{name}:")
    rejections = []
    acceptances = []

    with open(log_path, 'r') as f:
        for line in f:
            if 'Advanced NTP from' in line:
                # Example: Advanced NTP from time.google.com: offset=1558.2μs, delay=61.6ms, uncertainty=30779.1μs
                match = re.search(r'offset=([-\d.]+)([μm])s.*delay=([\d.]+)ms.*uncertainty=([\d.]+)([μm])s', line)
                if match:
                    offset_val = float(match.group(1))
                    offset_unit = match.group(2)
                    delay = float(match.group(3))
                    unc_val = float(match.group(4))
                    unc_unit = match.group(5)

                    # Convert to ms
                    offset = offset_val / 1000 if offset_unit == 'μ' else offset_val
                    uncertainty = unc_val / 1000 if unc_unit == 'μ' else unc_val

                    acceptances.append({'offset': offset, 'delay': delay, 'uncertainty': uncertainty})

            elif 'Poor NTP measurement' in line or 'NTP measurement rejected' in line:
                # Example: Poor NTP measurement from time.google.com: delay=136.6ms, stratum=2, uncertainty=68.3ms
                match = re.search(r'delay=([\d.]+)ms.*uncertainty=([\d.]+)ms', line)
                if match:
                    delay = float(match.group(1))
                    uncertainty = float(match.group(2))
                    rejections.append({'delay': delay, 'uncertainty': uncertainty})

    print(f"  Accepted: {len(acceptances)}, Rejected: {len(rejections)}")
    if len(rejections) > 0:
        rej_df = pd.DataFrame(rejections)
        print(f"  Rejection reasons:")
        print(f"    Mean delay: {rej_df['delay'].mean():.1f}ms (vs accepted: {pd.DataFrame(acceptances)['delay'].mean():.1f}ms)")
        print(f"    Mean uncertainty: {rej_df['uncertainty'].mean():.1f}ms (vs accepted: {pd.DataFrame(acceptances)['uncertainty'].mean():.1f}ms)")
        print(f"    Max uncertainty: {rej_df['uncertainty'].max():.1f}ms")

analyze_ntp_rejections(homelab_log, 'Homelab')
analyze_ntp_rejections(ares11_log, 'ARES-11')

# ============================================================================
# 5. MAE AND ACCUMULATED ERROR vs SYSTEM CLOCK
# ============================================================================
print("\n" + "="*80)
print("5. COMPARING TO SYSTEM CLOCK - MAE AND ACCUMULATED ERROR")
print("="*80)

def calculate_metrics(df, name):
    print(f"\n{name}:")

    # Filter to samples with NTP measurements (ground truth)
    ntp_samples = df[df['has_ntp'] == True].copy()

    if len(ntp_samples) == 0:
        print("  No NTP samples available")
        return

    # System clock error = NTP offset (how much system clock is off)
    # ChronoTick error = ChronoTick offset - NTP offset (residual error after correction)

    # At NTP measurement points, calculate errors
    system_error = ntp_samples['ntp_offset_ms'].values
    chronotick_error = ntp_samples['chronotick_offset_ms'].values - ntp_samples['ntp_offset_ms'].values

    # MAE (Mean Absolute Error)
    system_mae = np.abs(system_error).mean()
    chronotick_mae = np.abs(chronotick_error).mean()

    # Accumulated error (sum of absolute errors over time)
    system_accumulated = np.abs(system_error).sum()
    chronotick_accumulated = np.abs(chronotick_error).sum()

    # Improvement
    mae_improvement = 100 * (system_mae - chronotick_mae) / system_mae
    acc_improvement = 100 * (system_accumulated - chronotick_accumulated) / system_accumulated

    print(f"  NTP measurements analyzed: {len(ntp_samples)}")
    print(f"\n  System Clock (uncorrected):")
    print(f"    MAE: {system_mae:.3f} ms")
    print(f"    Accumulated error: {system_accumulated:.1f} ms")
    print(f"\n  ChronoTick (corrected):")
    print(f"    MAE: {chronotick_mae:.3f} ms")
    print(f"    Accumulated error: {chronotick_accumulated:.1f} ms")
    print(f"\n  Improvement:")
    print(f"    MAE: {mae_improvement:.1f}% better")
    print(f"    Accumulated: {acc_improvement:.1f}% better")

    # Calculate RMS error
    system_rms = np.sqrt(np.mean(system_error**2))
    chronotick_rms = np.sqrt(np.mean(chronotick_error**2))
    rms_improvement = 100 * (system_rms - chronotick_rms) / system_rms

    print(f"\n  RMS Error:")
    print(f"    System: {system_rms:.3f} ms")
    print(f"    ChronoTick: {chronotick_rms:.3f} ms")
    print(f"    Improvement: {rms_improvement:.1f}% better")

    return {
        'system_mae': system_mae,
        'chronotick_mae': chronotick_mae,
        'system_rms': system_rms,
        'chronotick_rms': chronotick_rms,
        'mae_improvement': mae_improvement,
        'rms_improvement': rms_improvement
    }

homelab_metrics = calculate_metrics(homelab_csv, 'Homelab')
ares11_metrics = calculate_metrics(ares11_csv, 'ARES-11')
ares12_metrics = calculate_metrics(ares12_csv, 'ARES-12')

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
