#!/usr/bin/env python3
"""Quick test to verify API fixes work"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from chronotick_inference.engine import ChronoTickInferenceEngine
from chronotick_inference.real_data_pipeline import RealDataPipeline
from chronotick_inference.tsfm_model_wrapper import create_model_wrappers

print("=" * 60)
print("Quick API Test")
print("=" * 60)

config_path = "chronotick_inference/config.yaml"

# Initialize
print("\n1. Initializing engine...")
engine = ChronoTickInferenceEngine(config_path)
engine.initialize_models()
print("✓ Engine initialized")

print("\n2. Initializing pipeline...")
pipeline = RealDataPipeline(config_path)
print("✓ Pipeline initialized")

print("\n3. Creating wrappers...")
cpu_wrapper, gpu_wrapper = create_model_wrappers(
    engine, pipeline.dataset_manager, pipeline.system_metrics
)
print("✓ Wrappers created")

print("\n4. Connecting models...")
pipeline.initialize(cpu_wrapper, gpu_wrapper)
pipeline.predictive_scheduler.set_model_interfaces(
    cpu_wrapper, gpu_wrapper, pipeline.fusion_engine
)
print("✓ Models connected")

print("\n5. Waiting 5 seconds for NTP measurements...")
time.sleep(5)

print("\n6. Testing API methods...")

# Test get_real_clock_correction
try:
    correction = pipeline.get_real_clock_correction(time.time())
    print(f"✓ get_real_clock_correction() works!")
    print(f"  - Offset: {correction.offset_correction:.9f}s")
    print(f"  - Uncertainty: {correction.uncertainty:.9f}s")
except Exception as e:
    print(f"✗ get_real_clock_correction() failed: {e}")

# Test get_recent_measurements
try:
    measurements = pipeline.dataset_manager.get_recent_measurements(window_seconds=10)
    print(f"✓ get_recent_measurements() works!")
    print(f"  - Count: {len(measurements)}")
    if measurements:
        ts, offset = measurements[-1]
        print(f"  - Latest offset: {offset:.9f}s ({offset*1e6:.1f}μs)")
except Exception as e:
    print(f"✗ get_recent_measurements() failed: {e}")

print("\n✅ API test complete!")

# Cleanup
engine.shutdown()
print("✓ Cleanup done")
