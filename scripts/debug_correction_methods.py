#!/usr/bin/env python3
"""Debug script to understand what each correction method is doing."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

# Import the correction function from validate_ntp_final
from validate_ntp_final import apply_correction, create_scenario

# Test with Medium positive error
print("="*80)
print("DEBUG: Medium Positive Error (+10ms)")
print("="*80)

timestamps, measured, true_offsets, ntp_time, ntp_offset, ntp_idx = create_scenario(
    base_offset=-50e-3, drift_rate=100e-6, ntp_error=10e-3
)

print(f"\nScenario:")
print(f"  Time range: 0 to {timestamps[-1]:.0f}s")
print(f"  NTP measurement at: t={ntp_time:.0f}s (index {ntp_idx})")
print(f"  True offset at NTP: {true_offsets[ntp_idx]*1000:.2f}ms")
print(f"  Measured offset at NTP: {measured[ntp_idx]*1000:.2f}ms")
print(f"  NTP reports: {ntp_offset*1000:.2f}ms")
print(f"  Prediction error: {(ntp_offset - measured[ntp_idx])*1000:.2f}ms")

methods = ['linear', 'drift_aware', 'advanced']

for method in methods:
    corrected = apply_correction(timestamps, measured, ntp_time, ntp_offset, method, debug=True)

    # Show actual correction amounts
    print(f"\n  Actual corrections applied:")
    print(f"    At t=0:     {(corrected[0] - measured[0])*1000:+.4f}ms")
    print(f"    At t={ntp_time/2:.0f}:    {(corrected[ntp_idx//2] - measured[ntp_idx//2])*1000:+.4f}ms")
    print(f"    At t={ntp_time:.0f} (NTP): {(corrected[ntp_idx] - measured[ntp_idx])*1000:+.4f}ms")

print("\n" + "="*80)
