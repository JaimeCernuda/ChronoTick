#!/usr/bin/env python3
"""
3-Minute Validation Test
Verifies that TimesFM is actually being used for predictions, not just fallbacks.
"""

import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Enable DEBUG logging
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
print("3-MINUTE VALIDATION TEST - TimesFM Usage Verification")
print("=" * 80)
print()

config_path = "configs/config_complete.yaml"

# Initialize
print("Initializing ChronoTick system...")
engine = ChronoTickInferenceEngine(config_path)
engine.initialize_models()
print("✓ TimesFM models loaded\n")

pipeline = RealDataPipeline(config_path)
cpu_wrapper, gpu_wrapper = create_model_wrappers(
    engine, pipeline.dataset_manager, pipeline.system_metrics
)
pipeline.initialize(cpu_model=cpu_wrapper, gpu_model=gpu_wrapper)
pipeline.predictive_scheduler.set_model_interfaces(
    cpu_model=cpu_wrapper,
    gpu_model=gpu_wrapper,
    fusion_engine=pipeline.fusion_engine
)
print("✓ All components initialized\n")

# Wait for proper warmup duration and populate dataset
warmup_duration = pipeline.ntp_collector.warm_up_duration
print(f"Waiting {warmup_duration}s for warmup and collecting NTP data...")
print("(Calling get_real_clock_correction() every second to populate dataset)")
for i in range(warmup_duration):
    time.sleep(1)
    # Call get_real_clock_correction() to trigger _check_for_ntp_updates()
    # This populates the dataset with NTP measurements
    try:
        correction = pipeline.get_real_clock_correction(time.time())
        print(f"  [{i+1:2d}/{warmup_duration}s] NTP data: offset={correction.offset_correction*1000:.3f}ms, source={correction.source}")
    except Exception as e:
        print(f"  [{i+1:2d}/{warmup_duration}s] Error: {e}")
print()

# Warmup should be complete automatically via timer
print("Waiting for warmup to complete automatically...")
time.sleep(5)  # Give warmup timer and scheduler time to transition
print("✓ Warmup complete, ML predictions should now be active\n")

# Run for 3 minutes, collecting corrections every 5 seconds
print("=" * 80)
print("RUNNING 3-MINUTE TEST - Collecting predictions every 5 seconds")
print("=" * 80)
print()

test_duration = 180  # 3 minutes
interval = 5  # Every 5 seconds
iterations = test_duration // interval

results = []

for i in range(iterations):
    current_time = time.time()
    correction = pipeline.get_real_clock_correction(current_time)

    result = {
        'iteration': i + 1,
        'time': current_time,
        'offset': correction.offset_correction,
        'uncertainty': correction.offset_uncertainty,
        'drift': correction.drift_rate,
        'source': correction.source,
        'confidence': correction.confidence
    }
    results.append(result)

    print(f"[{i+1:2d}/{iterations}] "
          f"Time: {current_time:.1f}, "
          f"Offset: {correction.offset_correction*1000:>8.3f}ms, "
          f"Uncertainty: {correction.offset_uncertainty*1000:>8.3f}ms, "
          f"Source: {correction.source:15s}, "
          f"Confidence: {correction.confidence:.3f}")

    if i < iterations - 1:
        time.sleep(interval)

print()
print("=" * 80)
print("TEST COMPLETE - Analyzing Results")
print("=" * 80)
print()

# Analyze results
sources = {}
for r in results:
    source = r['source']
    sources[source] = sources.get(source, 0) + 1

print("Prediction Sources:")
for source, count in sorted(sources.items()):
    percentage = (count / len(results)) * 100
    print(f"  {source:20s}: {count:3d} times ({percentage:5.1f}%)")
print()

# Check if TimesFM was actually used
ml_sources = ['cpu', 'gpu', 'fusion']
ml_count = sum(sources.get(s, 0) for s in ml_sources)
fallback_count = len(results) - ml_count

print("Summary:")
print(f"  Total predictions: {len(results)}")
print(f"  ML predictions (cpu/gpu/fusion): {ml_count} ({ml_count/len(results)*100:.1f}%)")
print(f"  Fallback predictions: {fallback_count} ({fallback_count/len(results)*100:.1f}%)")
print()

if ml_count > 0:
    print("✅ SUCCESS: TimesFM ML predictions are being used!")
else:
    print("❌ FAILURE: No ML predictions observed - system using only fallbacks")
print()

# Get scheduler stats
scheduler_stats = pipeline.predictive_scheduler.get_stats()
print("Scheduler Statistics:")
for key, value in scheduler_stats.items():
    print(f"  {key}: {value}")
print()

# Cleanup
print("Cleaning up...")
engine.shutdown()
pipeline.system_metrics.stop_collection()
print("✓ Test complete")
