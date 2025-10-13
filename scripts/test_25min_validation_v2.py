#!/usr/bin/env python3
"""
25-Minute Validation Test - NTP Correction v2 (Drift-Aware)
Tests the drift-aware NTP correction algorithm over 25 minutes.
"""

import sys
import time
import csv
import logging
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from chronotick.inference.engine import ChronoTickInferenceEngine
from chronotick.inference.real_data_pipeline import RealDataPipeline
from chronotick.inference.tsfm_model_wrapper import create_model_wrappers

print("=" * 80)
print("25-MINUTE VALIDATION TEST - NTP Correction v2 (Drift-Aware)")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Initialize pipeline
config_path = "configs/config_complete.yaml"

print("Initializing ChronoTick system with drift-aware NTP correction...")
engine = ChronoTickInferenceEngine(config_path)
engine.initialize_models()
print("✓ Models loaded\n")

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
print("✓ Pipeline initialized\n")

# Use pipeline's NTP collector for ground truth measurements
print("Using pipeline's NTP collector for ground truth measurements...")
print("✓ NTP collector initialized\n")

# Wait for first NTP measurement (critical - prevents crash during warmup)
print("Waiting for first NTP measurement...")
while pipeline.ntp_collector.get_latest_offset() is None:
    time.sleep(0.5)
print("✓ First NTP measurement received\n")

# Wait for warmup (60 seconds)
warmup_duration = pipeline.ntp_collector.warm_up_duration
print(f"Warmup period: {warmup_duration}s")
print("Populating dataset with NTP measurements...")

for i in range(warmup_duration):
    try:
        current_time = time.time()
        correction = pipeline.get_real_clock_correction(current_time)
        if i % 10 == 0:
            print(f"  [{i:2d}/{warmup_duration}s] offset={correction.offset_correction*1000:.2f}ms, source={correction.source}")
    except Exception as e:
        if i % 10 == 0:
            print(f"  [{i:2d}/{warmup_duration}s] Error: {e}")
    time.sleep(1)

print()
print("✓ Warmup complete, ML predictions should now be active\n")

# Wait for warmup timer to complete automatically
time.sleep(5)

# Wait for prediction cache to populate (critical - prevents cache miss crash)
print("Waiting for prediction cache to populate...")
wait_count = 0
while len(pipeline.predictive_scheduler.prediction_cache) == 0:
    time.sleep(1)
    wait_count += 1
    if wait_count > 30:
        print("⚠️  Warning: Prediction cache still empty after 30s")
        break
    if wait_count % 5 == 0:
        print(f"  Waiting for cache... ({wait_count}s)")

cache_size = len(pipeline.predictive_scheduler.prediction_cache)
print(f"✓ Prediction cache populated with {cache_size} entries\n")

# Run for 25 minutes (1500 seconds) with 10-second intervals
print("=" * 80)
print("RUNNING 25-MINUTE TEST - Testing Drift-Aware NTP Correction v2")
print("=" * 80)
print()

test_duration = 1500  # 25 minutes
interval = 10  # Every 10 seconds
iterations = test_duration // interval

# CSV output
csv_path = "results/ntp_correction_experiment/ntp_correction_v2_test.csv"
Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
csv_file = open(csv_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    'timestamp', 'elapsed_seconds', 'datetime', 'system_time', 'chronotick_time',
    'chronotick_offset_ms', 'chronotick_drift_us_per_s', 'chronotick_uncertainty_ms',
    'chronotick_confidence', 'chronotick_source',
    'ntp_ground_truth_offset_ms', 'ntp_ground_truth_uncertainty_ms', 'ntp_server',
    'has_ntp', 'chronotick_error_ms', 'system_error_ms'
])

results = []
ntp_ground_truth_count = 0
ntp_correction_active_count = 0
start_time = time.time()

try:
    for i in range(iterations):
        current_time = time.time()
        elapsed_seconds = current_time - start_time

        # Get ChronoTick correction
        correction = pipeline.get_real_clock_correction(current_time)

        # Calculate corrected time
        time_delta = current_time - correction.prediction_time
        corrected_time = current_time + correction.offset_correction + (correction.drift_rate * time_delta)

        # Get NTP ground truth from pipeline's collector
        has_ntp = False
        ntp_offset_ms = ""
        ntp_uncertainty_ms = ""
        ntp_server = ""
        chronotick_error = ""
        system_error = ""

        try:
            # Get recent NTP measurements from pipeline's collector
            # get_recent_measurements returns List[Tuple[timestamp, offset, uncertainty]]
            recent_ntp = pipeline.ntp_collector.get_recent_measurements(window_seconds=15)
            if recent_ntp and abs(current_time - recent_ntp[-1][0]) < 15:
                # Use the most recent measurement (last in list)
                ntp_timestamp, ntp_offset, ntp_uncertainty = recent_ntp[-1]
                has_ntp = True
                ntp_ground_truth_count += 1
                ntp_offset_ms = ntp_offset * 1000
                ntp_uncertainty_ms = ntp_uncertainty * 1000
                ntp_server = "ntp"  # Generic - specific server not stored in tuple

                # Calculate errors (ChronoTick vs NTP ground truth, System vs NTP ground truth)
                chronotick_error = abs(correction.offset_correction - ntp_offset) * 1000
                system_error = abs(0.0 - ntp_offset) * 1000  # System clock has 0 correction
        except Exception as e:
            # Log exceptions for debugging
            if i % 10 == 0:  # Don't spam logs
                logger.debug(f"NTP ground truth fetch failed: {e}")

        # Track if NTP correction was applied (source contains "ntp")
        if 'ntp' in correction.source:
            ntp_correction_active_count += 1

        # Write to CSV
        csv_writer.writerow([
            current_time,
            elapsed_seconds,
            datetime.fromtimestamp(current_time).isoformat(),
            current_time,
            corrected_time,
            correction.offset_correction * 1000,
            correction.drift_rate * 1e6,  # Convert to μs/s
            correction.offset_uncertainty * 1000,
            correction.confidence,
            correction.source,
            ntp_offset_ms,
            ntp_uncertainty_ms,
            ntp_server,
            has_ntp,
            chronotick_error,
            system_error
        ])
        csv_file.flush()

        # Console output with NTP correction indicator
        ntp_marker = "📡 NTP_V2" if 'ntp_v2' in correction.source else ("📡 NTP" if 'ntp' in correction.source else "")
        ground_truth_marker = "✅ GT" if has_ntp else ""

        print(f"[{int(elapsed_seconds):4d}s / 25min] "
              f"Offset: {correction.offset_correction*1000:>8.2f}ms, "
              f"Drift: {correction.drift_rate*1e6:>8.2f}μs/s, "
              f"Uncertainty: {correction.offset_uncertainty*1000:>8.2f}ms, "
              f"Source: {correction.source:15s}, "
              f"Confidence: {correction.confidence:.2f}  "
              f"{ntp_marker}  {ground_truth_marker}")

        results.append({
            'elapsed': elapsed_seconds,
            'offset': correction.offset_correction,
            'drift': correction.drift_rate,
            'uncertainty': correction.offset_uncertainty,
            'source': correction.source,
            'confidence': correction.confidence,
            'has_ntp': has_ntp
        })

        if i < iterations - 1:
            time.sleep(interval)

finally:
    csv_file.close()

print()
print("=" * 80)
print("TEST COMPLETE - 25-Minute Results")
print("=" * 80)
print()

# Analysis
print(f"Total samples: {len(results)}")
print(f"NTP ground truth measurements: {ntp_ground_truth_count}")
print(f"NTP correction applied: {ntp_correction_active_count} times ({ntp_correction_active_count/len(results)*100:.1f}%)")
print()

# Source breakdown
sources = {}
for r in results:
    source = r['source']
    sources[source] = sources.get(source, 0) + 1

print("Prediction Sources:")
for source, count in sorted(sources.items()):
    percentage = (count / len(results)) * 100
    print(f"  {source:20s}: {count:3d} times ({percentage:5.1f}%)")
print()

# Check for v2 correction usage
v2_count = sum(1 for r in results if 'ntp_v2' in r['source'])
if v2_count > 0:
    print(f"✅ SUCCESS: NTP Correction v2 (drift-aware) was used {v2_count} times!")
else:
    print(f"⚠️  WARNING: NTP Correction v2 was never applied - check configuration")
print()

print(f"CSV results saved to: {csv_path}")
print()

# Cleanup
print("Cleaning up...")
engine.shutdown()
pipeline.system_metrics.stop_collection()
print("✓ Test complete")
