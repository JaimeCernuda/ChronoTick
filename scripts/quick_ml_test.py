#!/usr/bin/env python3
"""
Quick ML Validation - Confirms ML predictions are working (no fallbacks)
"""

import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Minimal logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)-8s | %(name)-30s | %(message)s'
)

from chronotick.inference.engine import ChronoTickInferenceEngine
from chronotick.inference.real_data_pipeline import RealDataPipeline
from chronotick.inference.tsfm_model_wrapper import create_model_wrappers

print("=" * 80)
print("QUICK ML VALIDATION TEST")
print("=" * 80)
print()

config_path = "configs/config.yaml"

# Initialize
print("Loading TimesFM models...")
engine = ChronoTickInferenceEngine(config_path)
engine.initialize_models()
print("✓ Models loaded\n")

pipeline = RealDataPipeline(config_path)
cpu_wrapper, gpu_wrapper = create_model_wrappers(
    engine, pipeline.dataset_manager, pipeline.system_metrics
)
pipeline.initialize(cpu_model=cpu_wrapper, gpu_model=gpu_wrapper)
print("✓ Pipeline initialized\n")

# Wait for warmup
warmup_duration = pipeline.ntp_collector.warm_up_duration
print(f"Warmup phase ({warmup_duration}s) - Populating dataset...")
for i in range(warmup_duration):
    time.sleep(1)
    try:
        correction = pipeline.get_real_clock_correction(time.time())
        if i % 10 == 0:
            print(f"  [{i}/{warmup_duration}s] source={correction.source}")
    except Exception as e:
        print(f"  [{i}/{warmup_duration}s] ERROR: {e}")
        break

print()

# Wait for scheduler to start
print("Waiting for scheduler to start...")
time.sleep(7)  # Give scheduler time to generate first predictions
print("✓ Scheduler should be running\n")

# Test ML predictions
print("Testing ML predictions (10 samples)...")
print("=" * 80)

ml_count = 0
errors = []

for i in range(10):
    try:
        correction = pipeline.get_real_clock_correction(time.time())
        print(f"[{i+1}/10] source={correction.source:15s} "
              f"offset={correction.offset_correction*1000:8.3f}ms "
              f"confidence={correction.confidence:.2f}")

        if correction.source in ['cpu', 'gpu', 'fusion']:
            ml_count += 1

        time.sleep(0.5)
    except Exception as e:
        errors.append(str(e))
        print(f"[{i+1}/10] ERROR: {e}")
        break

print()
print("=" * 80)
print("RESULTS")
print("=" * 80)

if errors:
    print(f"❌ Test FAILED with errors:")
    for err in errors:
        print(f"   {err}")
elif ml_count == 10:
    print(f"✅ SUCCESS: All 10 predictions from ML (100%)")
    print(f"   NO FALLBACKS - System is using ML predictions exclusively!")
else:
    print(f"⚠️  PARTIAL: {ml_count}/10 from ML ({ml_count*10}%)")

print()

# Cleanup
engine.shutdown()
pipeline.shutdown()
print("✓ Test complete")
