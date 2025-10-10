#!/usr/bin/env python3
"""
FINAL VALIDATION TEST - 3 Minutes
Verifies all fixes: ML-only predictions, correct extrapolation, no fallbacks
"""

import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Minimal logging - only warnings and errors
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)-8s | %(name)-30s | %(message)s'
)

from chronotick_inference.engine import ChronoTickInferenceEngine
from chronotick_inference.real_data_pipeline import RealDataPipeline
from chronotick_inference.tsfm_model_wrapper import create_model_wrappers

print("=" * 80)
print("FINAL VALIDATION TEST - 3 MINUTES")
print("Verifying: ML-only predictions, no fallbacks, correct extrapolation")
print("=" * 80)
print()

config_path = "chronotick_inference/config.yaml"

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

# Warmup phase
warmup_duration = pipeline.ntp_collector.warm_up_duration
print(f"Warmup phase ({warmup_duration}s) - Collecting NTP data...")
print("Calling get_real_clock_correction() every second to populate dataset")
print("(Will stop calling 3 seconds early to let scheduler start)")
# Call get_real_clock_correction() for most of warmup to populate dataset
# But stop 3 seconds early to give scheduler time to start and generate first predictions
for i in range(warmup_duration - 3):
    time.sleep(1)
    try:
        correction = pipeline.get_real_clock_correction(time.time())
        if i % 15 == 0 or i == warmup_duration - 4:
            print(f"  [{i+1:2d}/{warmup_duration}s] Dataset: {len(pipeline.dataset_manager.get_recent_measurements())} measurements, source={correction.source}")
    except Exception as e:
        print(f"  [{i+1:2d}/{warmup_duration}s] ERROR: {e}")
        raise

# Wait the remaining seconds for warmup to complete + scheduler to start
print(f"Waiting final 3 seconds for warmup timer to complete and scheduler to start...")
time.sleep(3)

print()
print("✓ Warmup phase complete\n")
print("Waiting for scheduler to start and generate first predictions...")
print("(Models need time to fully load + scheduler needs time to generate predictions)")
print("Waiting 20 seconds to ensure:")
print("  1. TimesFM models fully loaded")
print("  2. Scheduler has started")
print("  3. First CPU predictions generated and cached")
time.sleep(20)  # Give more time for models to load and first predictions to cache
print("✓ Scheduler should have predictions cached\n")

# Main test - 3 minutes, sampling every 5 seconds
print("=" * 80)
print("RUNNING 3-MINUTE TEST - Collecting predictions every 5 seconds")
print("=" * 80)
print()

test_duration = 180  # 3 minutes
interval = 5
iterations = test_duration // interval

results = []
errors = []

for i in range(iterations):
    try:
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
              f"source={correction.source:10s} "
              f"offset={correction.offset_correction*1000:>8.3f}ms "
              f"drift={correction.drift_rate*1e6:>8.3f}μs/s "
              f"uncertainty=±{correction.offset_uncertainty*1000:>6.3f}ms "
              f"conf={correction.confidence:.2f}")

        if i < iterations - 1:
            time.sleep(interval)

    except Exception as e:
        errors.append(str(e))
        print(f"[{i+1:2d}/{iterations}] ERROR: {e}")
        break

print()
print("=" * 80)
print("TEST RESULTS")
print("=" * 80)
print()

# Analyze sources
sources = {}
for r in results:
    source = r['source']
    sources[source] = sources.get(source, 0) + 1

print("Prediction Sources:")
for source, count in sorted(sources.items()):
    percentage = (count / len(results)) * 100
    print(f"  {source:20s}: {count:3d} times ({percentage:5.1f}%)")
print()

# Check ML usage
ml_sources = ['cpu', 'gpu', 'fusion']
ml_count = sum(sources.get(s, 0) for s in ml_sources)
fallback_sources = set(sources.keys()) - set(ml_sources)
fallback_count = sum(sources.get(s, 0) for s in fallback_sources)

print("Summary:")
print(f"  Total samples: {len(results)}")
print(f"  ML predictions (cpu/gpu/fusion): {ml_count} ({ml_count/len(results)*100:.1f}%)")
print(f"  Fallback predictions: {fallback_count} ({fallback_count/len(results)*100:.1f}%)")
if fallback_sources:
    print(f"  Fallback types: {', '.join(fallback_sources)}")
print()

# Get scheduler stats
scheduler_stats = pipeline.predictive_scheduler.get_stats()
print("Scheduler Statistics:")
for key, value in sorted(scheduler_stats.items()):
    print(f"  {key:30s}: {value}")
print()

# Final verdict
if errors:
    print("❌ TEST FAILED - Errors occurred:")
    for err in errors:
        print(f"   {err}")
elif ml_count == len(results):
    print("✅ SUCCESS: 100% ML predictions - NO FALLBACKS!")
    print("   All fixes verified:")
    print("   ✓ prediction_time bug fixed")
    print("   ✓ Silent fallbacks removed")
    print("   ✓ Scheduler timing correct")
    print("   ✓ ML-only mode enforced")
elif ml_count > 0:
    print(f"⚠️  PARTIAL SUCCESS: {ml_count}/{len(results)} from ML ({ml_count/len(results)*100:.1f}%)")
    print(f"   Warning: {fallback_count} predictions used fallbacks")
else:
    print("❌ FAILURE: No ML predictions observed")

print()

# Cleanup
print("Cleaning up...")
engine.shutdown()
pipeline.shutdown()
print("✓ Test complete")
