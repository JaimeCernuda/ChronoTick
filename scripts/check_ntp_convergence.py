#!/usr/bin/env python3
"""Check if corrected values match NTP at measurement point."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from validate_ntp_final import apply_correction, create_scenario

timestamps, measured, true_offsets, ntp_time, ntp_offset, ntp_idx = create_scenario(
    base_offset=-50e-3, drift_rate=100e-6, ntp_error=10e-3
)

print(f"NTP reports: {ntp_offset*1000:.3f}ms at t={ntp_time:.0f}s\n")

for method in ['none', 'linear', 'drift_aware', 'advanced']:
    corrected = apply_correction(timestamps, measured, ntp_time, ntp_offset, method)

    value_at_ntp = corrected[ntp_idx] * 1000
    error_at_ntp = abs(corrected[ntp_idx] - ntp_offset) * 1000

    print(f"{method.upper()}:")
    print(f"  Corrected value at NTP: {value_at_ntp:.3f}ms")
    print(f"  Error vs NTP: {error_at_ntp:.3f}ms")
    print()
