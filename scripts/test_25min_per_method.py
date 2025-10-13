#!/usr/bin/env python3
"""
25-Minute Per Method Test with Dataset Correction Tracking

Tests each NTP correction method for 25 minutes (NOT 25 min total) and captures:
- ChronoTick predictions
- NTP ground truth measurements
- Dataset corrections applied (before/after)
- Correction deltas

Usage:
    python test_25min_per_method.py none
    python test_25min_per_method.py linear
    python test_25min_per_method.py drift_aware
    python test_25min_per_method.py advanced
"""

import sys
import time
import csv
import logging
import argparse
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))

from chronotick.inference.engine import ChronoTickInferenceEngine
from chronotick.inference.real_data_pipeline import RealDataPipeline
from chronotick.inference.tsfm_model_wrapper import create_model_wrappers

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Test NTP correction methods - 25 min PER method')
parser.add_argument('method', choices=['none', 'linear', 'drift_aware', 'advanced'],
                   help='NTP correction method to test')
parser.add_argument('--duration', type=int, default=1500,
                   help='Test duration in seconds (default: 1500 = 25 min)')
parser.add_argument('--interval', type=int, default=10,
                   help='Sampling interval in seconds (default: 10)')
args = parser.parse_args()

print("=" * 80)
print(f"25-MINUTE TEST (PER METHOD) - Method: {args.method.upper()}")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Duration: {args.duration}s ({args.duration//60} minutes) FOR THIS METHOD")
print(f"Sampling interval: {args.interval}s")
print()

# Load config and temporarily override the correction method
config_path = "configs/config_complete.yaml"

import yaml
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Update the correction method
if 'prediction_scheduling' not in config:
    config['prediction_scheduling'] = {}
if 'ntp_correction' not in config['prediction_scheduling']:
    config['prediction_scheduling']['ntp_correction'] = {}

config['prediction_scheduling']['ntp_correction']['method'] = args.method
print(f"Setting correction method to: {args.method}")
print(f"Config path: prediction_scheduling.ntp_correction.method")

# Write temporary config
temp_config_path = f"configs/config_test_{args.method}.yaml"
with open(temp_config_path, 'w') as f:
    yaml.dump(config, f)
print(f"✓ Created temporary config: {temp_config_path}\n")

# Initialize pipeline
print("Initializing ChronoTick system...")
engine = ChronoTickInferenceEngine(temp_config_path)
engine.initialize_models()
print("✓ Models loaded\n")

pipeline = RealDataPipeline(temp_config_path)
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

# Wait for first NTP measurement
print("Waiting for first NTP measurement...")
while pipeline.ntp_collector.get_latest_offset() is None:
    time.sleep(0.5)
print("✓ First NTP measurement received\n")

# Warmup
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
print("✓ Warmup complete\n")
time.sleep(5)

# Wait for prediction cache
print("Waiting for prediction cache...")
wait_count = 0
while len(pipeline.predictive_scheduler.prediction_cache) == 0:
    time.sleep(1)
    wait_count += 1
    if wait_count > 30:
        print("⚠️  Warning: Prediction cache still empty")
        break

print(f"✓ Cache populated with {len(pipeline.predictive_scheduler.prediction_cache)} entries\n")

# Verify dataset
dataset_size = len(pipeline.dataset_manager.get_recent_measurements())
print(f"Dataset verification:")
print(f"  Total measurements: {dataset_size}")
print(f"  NTP correction method: {args.method}")
print(f"  NTP corrections applied: {pipeline.stats['ntp_corrections_applied']}")
print()

# Run test with ENHANCED logging
print("=" * 80)
print(f"RUNNING TEST - Method: {args.method.upper()}")
print("=" * 80)
print()

iterations = args.duration // args.interval
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
csv_filename = f"ntp_correction_{args.method}_25min_{timestamp_str}.csv"
csv_path = Path("results/ntp_correction_experiment") / csv_filename
csv_path.parent.mkdir(parents=True, exist_ok=True)

csv_file = open(csv_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    'timestamp', 'elapsed_seconds', 'datetime', 'system_time', 'chronotick_time',
    'chronotick_offset_ms', 'chronotick_drift_us_per_s', 'chronotick_uncertainty_ms',
    'chronotick_confidence', 'chronotick_source',
    'ntp_ground_truth_offset_ms', 'ntp_ground_truth_uncertainty_ms', 'ntp_server',
    'has_ntp', 'chronotick_error_ms', 'system_error_ms',
    # ENHANCED: Add dataset state
    'dataset_size', 'dataset_corrections_applied'
])

results = []
ntp_ground_truth_count = 0
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

        # Get NTP ground truth
        has_ntp = False
        ntp_offset_ms = ""
        ntp_uncertainty_ms = ""
        ntp_server = ""
        chronotick_error = ""
        system_error = ""

        try:
            recent_ntp = pipeline.ntp_collector.get_recent_measurements(window_seconds=15)
            if recent_ntp and abs(current_time - recent_ntp[-1][0]) < 15:
                ntp_timestamp, ntp_offset, ntp_uncertainty = recent_ntp[-1]
                has_ntp = True
                ntp_ground_truth_count += 1
                ntp_offset_ms = ntp_offset * 1000
                ntp_uncertainty_ms = ntp_uncertainty * 1000
                ntp_server = "ntp"

                chronotick_error = abs(correction.offset_correction - ntp_offset) * 1000
                system_error = abs(0.0 - ntp_offset) * 1000
        except Exception as e:
            if i % 10 == 0:
                logger.debug(f"NTP ground truth fetch failed: {e}")

        # ENHANCED: Get dataset state
        dataset_size = len(pipeline.dataset_manager.get_recent_measurements())
        dataset_corrections_applied = pipeline.stats.get('ntp_corrections_applied', 0)

        # Write to CSV
        csv_writer.writerow([
            current_time,
            elapsed_seconds,
            datetime.fromtimestamp(current_time).isoformat(),
            current_time,
            corrected_time,
            correction.offset_correction * 1000,
            correction.drift_rate * 1e6,
            correction.offset_uncertainty * 1000,
            correction.confidence,
            correction.source,
            ntp_offset_ms,
            ntp_uncertainty_ms,
            ntp_server,
            has_ntp,
            chronotick_error,
            system_error,
            dataset_size,
            dataset_corrections_applied
        ])
        csv_file.flush()

        # Console output
        ntp_marker = f"📡 {args.method.upper()}" if args.method in correction.source.lower() else ""
        ground_truth_marker = "✅ GT" if has_ntp else ""

        print(f"[{int(elapsed_seconds):4d}s / {args.duration//60}min] "
              f"Offset: {correction.offset_correction*1000:>8.2f}ms, "
              f"Drift: {correction.drift_rate*1e6:>8.2f}μs/s, "
              f"DS: {dataset_size:3d}, "
              f"Corr: {dataset_corrections_applied:3d}, "
              f"Source: {correction.source:20s}  "
              f"{ntp_marker}  {ground_truth_marker}")

        results.append({
            'elapsed': elapsed_seconds,
            'offset': correction.offset_correction,
            'has_ntp': has_ntp,
            'chronotick_error': chronotick_error if chronotick_error != "" else None,
            'system_error': system_error if system_error != "" else None
        })

        if i < iterations - 1:
            time.sleep(args.interval)

finally:
    csv_file.close()

print()
print("=" * 80)
print(f"TEST COMPLETE - Method: {args.method.upper()}")
print("=" * 80)
print()

# Analysis
print(f"Total samples: {len(results)}")
print(f"NTP ground truth measurements: {ntp_ground_truth_count}")
print()

# Calculate accuracy
results_with_gt = [r for r in results if r['chronotick_error'] is not None]
if results_with_gt:
    chronotick_errors = [r['chronotick_error'] for r in results_with_gt]
    system_errors = [r['system_error'] for r in results_with_gt]

    import numpy as np
    chronotick_mae = np.mean(chronotick_errors)
    system_mae = np.mean(system_errors)

    print("ACCURACY VS NTP GROUND TRUTH:")
    print(f"  ChronoTick MAE: {chronotick_mae:.3f}ms")
    print(f"  System Clock MAE: {system_mae:.3f}ms")
    print(f"  Improvement: {((system_mae - chronotick_mae) / system_mae * 100):+.1f}%")
    print()

print(f"CSV results saved to: {csv_path}")
print()

# Cleanup
print("Cleaning up...")
engine.shutdown()
pipeline.system_metrics.stop_collection()
Path(temp_config_path).unlink()
print("✓ Test complete")
