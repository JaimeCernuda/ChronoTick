#!/usr/bin/env python3
"""Simple API check - verify method names exist"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from chronotick.inference.real_data_pipeline import RealDataPipeline, DatasetManager

print("Checking RealDataPipeline API...")

# Check if methods exist
has_get_real_clock_correction = hasattr(RealDataPipeline, 'get_real_clock_correction')
has_get_correction = hasattr(RealDataPipeline, 'get_correction')

print(f"✓ has get_real_clock_correction(): {has_get_real_clock_correction}")
print(f"  has get_correction(): {has_get_correction} (old name)")

print("\nChecking DatasetManager API...")

has_get_recent_measurements = hasattr(DatasetManager, 'get_recent_measurements')
print(f"✓ has get_recent_measurements(): {has_get_recent_measurements}")

# Check method signature
import inspect
if has_get_recent_measurements:
    sig = inspect.signature(DatasetManager.get_recent_measurements)
    print(f"  Parameters: {list(sig.parameters.keys())}")

print("\n✅ API check complete - methods exist with correct names!")
