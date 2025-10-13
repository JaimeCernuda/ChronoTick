#!/usr/bin/env python3
"""
Run integration test with full DEBUG logging enabled.
This script is designed to verify the deadlock fix works properly.
"""

import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Enable DEBUG logging for ALL components
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(name)-40s | %(funcName)-30s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Also enable debug logger for pipeline traces
debug_logger = logging.getLogger('chronotick.inference.real_data_pipeline.debug')
debug_logger.setLevel(logging.DEBUG)

from chronotick.inference.engine import ChronoTickInferenceEngine
from chronotick.inference.real_data_pipeline import RealDataPipeline
from chronotick.inference.tsfm_model_wrapper import create_model_wrappers

print("=" * 80)
print("INTEGRATION TEST WITH FULL DEBUG LOGGING")
print("=" * 80)
print()

config_path = "configs/config.yaml"

# Step 1: Initialize engine
print("STEP 1: Initializing ChronoTickInferenceEngine...")
engine = ChronoTickInferenceEngine(config_path)
engine.initialize_models()
print("✓ Engine initialized\n")

# Step 2: Initialize pipeline
print("STEP 2: Initializing RealDataPipeline...")
pipeline = RealDataPipeline(config_path)
print(f"✓ Pipeline created: initialized={pipeline.initialized}, warm_up_complete={pipeline.warm_up_complete}\n")

# Step 3: Create wrappers
print("STEP 3: Creating model wrappers...")
cpu_wrapper, gpu_wrapper = create_model_wrappers(
    engine, pipeline.dataset_manager, pipeline.system_metrics
)
print(f"✓ CPU wrapper: {cpu_wrapper.model_type}")
print(f"✓ GPU wrapper: {gpu_wrapper.model_type}\n")

# Step 4: Connect models to pipeline
print("STEP 4: Connecting models to pipeline...")
pipeline.initialize(cpu_model=cpu_wrapper, gpu_model=gpu_wrapper)
print(f"✓ Pipeline initialized with models: initialized={pipeline.initialized}\n")

# Step 5: Set scheduler interfaces
print("STEP 5: Setting scheduler interfaces...")
pipeline.predictive_scheduler.set_model_interfaces(
    cpu_model=cpu_wrapper,
    gpu_model=gpu_wrapper,
    fusion_engine=pipeline.fusion_engine
)
print("✓ Scheduler configured\n")

# Step 6: Wait for warmup (now only 5 seconds)
print("STEP 6: Waiting 6 seconds for warmup to complete...")
pipeline.system_metrics.start_collection()
for i in range(6):
    time.sleep(1)
    print(f"  {i+1}/6 seconds, warm_up_complete={pipeline.warm_up_complete}")

# Step 7: Test during warmup
print("\nSTEP 7: Testing get_real_clock_correction during warmup...")
current_time = time.time()
print(f"  Calling pipeline.get_real_clock_correction({current_time})")
correction = pipeline.get_real_clock_correction(current_time)
print(f"✓ During warmup: offset={correction.offset_correction:.6f}s, source={correction.source}\n")

# Step 8: Force warmup complete
print("STEP 8: Forcing warmup completion...")
pipeline.warm_up_complete = True
print(f"  warm_up_complete={pipeline.warm_up_complete}")
print("  Sleeping 2 seconds to let scheduler generate predictions...")
time.sleep(2)

# Step 9: Test after warmup (THIS IS WHERE IT USED TO HANG)
print("\nSTEP 9: Testing get_real_clock_correction after warmup (critical test)...")
current_time = time.time()
print(f"  Calling pipeline.get_real_clock_correction({current_time})")
print("  >>> If this hangs, the deadlock still exists <<<")
correction = pipeline.get_real_clock_correction(current_time)
print(f"✓ After warmup: offset={correction.offset_correction:.6f}s, source={correction.source}\n")

# Step 10: Stress test
print("STEP 10: Stress test - 10 rapid calls...")
for i in range(10):
    current_time = time.time()
    correction = pipeline.get_real_clock_correction(current_time)
    print(f"  [{i+1}/10] offset={correction.offset_correction:.6f}s, source={correction.source}")
    time.sleep(0.1)

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED - NO DEADLOCK!")
print("=" * 80)

# Cleanup
engine.shutdown()
pipeline.system_metrics.stop_collection()
print("\n✓ Cleanup complete")
