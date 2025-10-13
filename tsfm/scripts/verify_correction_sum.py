#!/usr/bin/env python3
"""Verify that corrections sum to total error."""

import sys
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from validate_ntp_final import apply_correction, create_scenario

timestamps, measured, true_offsets, ntp_time, ntp_offset, ntp_idx = create_scenario(
    base_offset=-50e-3, drift_rate=100e-6, ntp_error=10e-3
)

error = ntp_offset - measured[ntp_idx]
print(f"Total error to distribute: {error*1000:.3f}ms\n")

for method in ['linear', 'drift_aware', 'advanced']:
    corrected = apply_correction(timestamps, measured, ntp_time, ntp_offset, method)

    # Calculate total correction applied
    corrections = corrected[:ntp_idx+1] - measured[:ntp_idx+1]
    total_correction = np.sum(corrections)

    print(f"{method.upper()}:")
    print(f"  Sum of all corrections: {total_correction*1000:.3f}ms")
    print(f"  Should equal error: {error*1000:.3f}ms")
    print(f"  Ratio: {total_correction/error:.4f}")
    print()
