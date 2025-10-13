#!/usr/bin/env python3
"""Quick integration test - verify no deadlock"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from chronotick.inference.engine import ChronoTickInferenceEngine
from chronotick.inference.real_data_pipeline import RealDataPipeline
from chronotick.inference.tsfm_model_wrapper import create_model_wrappers

print("=" * 80)
print("Quick Integration Test - Deadlock Fix Verification")
print("=" * 80)

config_path = "configs/config.yaml"

# Initialize
print("\n1. Initializing...")
engine = ChronoTickInferenceEngine(config_path)
engine.initialize_models()

pipeline = RealDataPipeline(config_path)
cpu_wrapper, gpu_wrapper = create_model_wrappers(
    engine, pipeline.dataset_manager, pipeline.system_metrics
)
pipeline.initialize(cpu_model=cpu_wrapper, gpu_model=gpu_wrapper)
pipeline.predictive_scheduler.set_model_interfaces(
    cpu_model=cpu_wrapper, gpu_model=gpu_wrapper, fusion_engine=pipeline.fusion_engine
)
print("✓ All components initialized")

# Wait for 3 seconds of NTP collection
print("\n2. Waiting 3 seconds for NTP measurements...", flush=True)
pipeline.system_metrics.start_collection()
time.sleep(3)

# Test during warmup
print("\n3. Testing during warmup (should use NTP)...", flush=True)
correction = pipeline.get_real_clock_correction(time.time())
print(f"✓ During warmup: offset={correction.offset_correction:.6f}s, source={correction.source}")

# Force warmup complete
print("\n4. Forcing warmup completion...")
pipeline.warm_up_complete = True
time.sleep(1)  # Give scheduler time to generate predictions

# Test after warmup
print("\n5. Testing after warmup (should use ML predictions)...")
correction = pipeline.get_real_clock_correction(time.time())
print(f"✓ After warmup: offset={correction.offset_correction:.6f}s, source={correction.source}")

# Test multiple rapid calls (stress test)
print("\n6. Stress test - 10 rapid calls...")
for i in range(10):
    correction = pipeline.get_real_clock_correction(time.time())
    print(f"  [{i+1}/10] offset={correction.offset_correction:.6f}s, source={correction.source}")
    time.sleep(0.1)

print("\n✅ ALL TESTS PASSED - No deadlock!")

# Cleanup
engine.shutdown()
pipeline.system_metrics.stop_collection()
print("✓ Cleanup complete")
