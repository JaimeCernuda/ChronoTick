#!/usr/bin/env python3
"""Compare LINEAR vs DRIFT_AWARE corrections mathematically."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from validate_ntp_final import create_scenario

# Create scenario
timestamps, measured, true_offsets, ntp_time, ntp_offset, ntp_idx = create_scenario(
    base_offset=-50e-3, drift_rate=100e-6, ntp_error=10e-3
)

error = ntp_offset - measured[ntp_idx]
interval = ntp_time
print(f"Error: {error*1000:.3f}ms, Interval: {interval:.0f}s\n")

# LINEAR formula: correction(t) = (t/T) * error
print("LINEAR:")
print(f"  Formula: correction(t) = (t/{interval}) * {error*1000:.3f}ms")
print(f"  At t=0:  {0.0 * error * 1000:.3f}ms")
print(f"  At t=30: {(30/interval) * error * 1000:.3f}ms")
print(f"  At t=60: {(60/interval) * error * 1000:.3f}ms")
print(f"  Slope: {(error/interval)*1000:.3f} ms/s")

# DRIFT_AWARE formula
sigma_offset, sigma_drift = 0.001, 0.0001
var_offset = sigma_offset**2
var_drift = (sigma_drift * interval)**2
var_total = var_offset + var_drift
w_offset = var_offset / var_total
w_drift = var_drift / var_total

offset_corr = w_offset * error
drift_corr = (w_drift * error) / interval

print(f"\nDRIFT_AWARE:")
print(f"  Offset weight: {w_offset:.4f}, Drift weight: {w_drift:.4f}")
print(f"  Formula: correction(t) = {offset_corr*1000:.3f}ms + {drift_corr*1000:.3f}*(ms/s) * t")
print(f"  At t=0:  {offset_corr * 1000:.3f}ms")
print(f"  At t=30: {(offset_corr + drift_corr*30) * 1000:.3f}ms")
print(f"  At t=60: {(offset_corr + drift_corr*60) * 1000:.3f}ms")
print(f"  Slope: {drift_corr*1000:.3f} ms/s")

print(f"\nDIFFERENCE:")
linear_slope = (error/interval)*1000
drift_slope = drift_corr*1000
print(f"  Linear slope: {linear_slope:.3f} ms/s")
print(f"  Drift slope: {drift_slope:.3f} ms/s")
print(f"  Slope difference: {abs(linear_slope - drift_slope):.3f} ms/s ({abs(linear_slope-drift_slope)/linear_slope*100:.1f}%)")
print(f"\n  Drift_aware adds constant offset of {offset_corr*1000:.3f}ms")
print(f"  This is {offset_corr/error*100:.1f}% of total error")
