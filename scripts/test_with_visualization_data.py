#ana!/usr/bin/env python3
"""
Enhanced Test with Full Before/After Correction Visualization Data

Captures:
1. Client predictions (solid line) - what was returned to users
2. Dataset state after corrections (dashed line) - what ML model sees
3. NTP ground truth (red X markers)

Usage:
    python test_with_visualization_data.py drift_aware --duration 1500
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
from chronotick.inference.dataset_correction_logger import DatasetCorrectionLogger, ClientPredictionLogger

# Parse arguments
parser = argparse.ArgumentParser(description='Test with visualization data')
parser.add_argument('method', choices=['none', 'linear', 'drift_aware', 'advanced', 'advance_absolute', 'backtracking'])
parser.add_argument('--duration', type=int, default=1500, help='Test duration in seconds')
parser.add_argument('--interval', type=int, default=10, help='Sampling interval')
parser.add_argument('--config', type=str, default='configs/config_complete.yaml',
                    help='Config file to use')
parser.add_argument('--output-dir', type=str, default='results/ntp_correction_experiment/visualization_data',
                    help='Output directory for results')
args = parser.parse_args()

print("=" * 80)
print(f"TEST WITH VISUALIZATION DATA - Method: {args.method.upper()}")
print("=" * 80)
print(f"Config: {args.config}")
print(f"Output: {args.output_dir}")
print(f"Duration: {args.duration}s ({args.duration//60} min)")
print(f"This will capture:")
print(f"  1. Client predictions (what users received)")
print(f"  2. Dataset corrections (how data was adjusted)")
print(f"  3. NTP ground truth (for comparison)")
print()

# Setup config
import yaml
config_path = args.config
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

if 'prediction_scheduling' not in config:
    config['prediction_scheduling'] = {}
if 'ntp_correction' not in config['prediction_scheduling']:
    config['prediction_scheduling']['ntp_correction'] = {}

config['prediction_scheduling']['ntp_correction']['method'] = args.method

# Print config summary for verification
print(f"Correction method: {args.method}")
print(f"Long-term model: {'ENABLED' if config.get('long_term', {}).get('enabled', False) else 'DISABLED'}")
print(f"Covariates: {'ENABLED' if config.get('covariates', {}).get('enabled', False) else 'DISABLED'}")
if config.get('covariates', {}).get('enabled', False):
    covars = config.get('covariates', {}).get('variables', [])
    print(f"  Variables: {', '.join(covars) if covars else 'None'}")

temp_config_path = f"configs/config_test_{args.method}.yaml"
with open(temp_config_path, 'w') as f:
    yaml.dump(config, f)

# Initialize
print("\nInitializing ChronoTick...")
engine = ChronoTickInferenceEngine(temp_config_path)
engine.initialize_models()

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
print("✓ Initialized\n")

# Setup loggers
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
results_dir = Path(args.output_dir)
results_dir.mkdir(parents=True, exist_ok=True)

correction_logger = DatasetCorrectionLogger(
    results_dir / f"dataset_corrections_{args.method}_{timestamp_str}.csv",
    enabled=True
)

client_logger = ClientPredictionLogger(
    results_dir / f"client_predictions_{args.method}_{timestamp_str}.csv",
    enabled=True
)

# Also create summary CSV
summary_csv = results_dir / f"summary_{args.method}_{timestamp_str}.csv"
summary_file = open(summary_csv, 'w', newline='')
summary_writer = csv.writer(summary_file)
summary_writer.writerow([
    'timestamp', 'elapsed_seconds',
    'client_offset_ms', 'client_drift_us_per_s', 'client_source',
    'ntp_ground_truth_offset_ms', 'ntp_uncertainty_ms', 'has_ntp',
    'chronotick_error_ms', 'system_error_ms',
    'dataset_size', 'corrections_applied'
])

# Wait for NTP
print("Waiting for first NTP...")
while pipeline.ntp_collector.get_latest_offset() is None:
    time.sleep(0.5)
print("✓ First NTP received\n")

# Warmup
warmup_duration = pipeline.ntp_collector.warm_up_duration
print(f"Warmup: {warmup_duration}s")
for i in range(warmup_duration):
    try:
        correction = pipeline.get_real_clock_correction(time.time())
        if i % 10 == 0:
            print(f"  [{i}/{warmup_duration}s] offset={correction.offset_correction*1000:.2f}ms")
    except Exception as e:
        if i % 10 == 0:
            print(f"  [{i}/{warmup_duration}s] Error: {e}")
    time.sleep(1)

print("\n✓ Warmup complete\n")
time.sleep(5)

# HOOK: Monkey-patch dataset manager to log corrections
original_apply_correction = pipeline.dataset_manager.apply_ntp_correction

def logged_apply_correction(ntp_measurement, method, offset_uncertainty, drift_uncertainty):
    """Wrapper that logs corrections"""
    # Apply the correction and get metadata
    metadata = original_apply_correction(ntp_measurement, method, offset_uncertainty, drift_uncertainty)

    # Log the correction event if metadata was returned (meaning correction was applied)
    if metadata is not None:
        correction_logger.log_correction_event(
            pipeline.dataset_manager,
            ntp_measurement,
            metadata['method'],
            metadata['interval_start'],
            metadata['interval_end'],
            metadata['error']
        )

    return metadata

pipeline.dataset_manager.apply_ntp_correction = logged_apply_correction

# Run test
print("=" * 80)
print(f"RUNNING TEST - Method: {args.method.upper()}")
print("=" * 80)
print()

iterations = args.duration // args.interval
start_time = time.time()
ntp_ground_truth_count = 0
last_evaluated_ntp_timestamp = None  # Track last NTP we used for evaluation

try:
    for i in range(iterations):
        current_time = time.time()
        elapsed = current_time - start_time

        # Get correction (what's returned to client)
        correction = pipeline.get_real_clock_correction(current_time)

        # Log client prediction
        client_logger.log_prediction(
            current_time,
            correction.offset_correction,
            correction.drift_rate,
            correction.offset_uncertainty,
            correction.confidence,
            correction.source
        )

        # Get NTP ground truth - only evaluate once per NTP arrival and skip early warmup
        has_ntp = False
        ntp_offset_ms = ""
        ntp_uncertainty_ms = ""
        chronotick_error = ""
        system_error = ""

        try:
            # Skip evaluations during early warmup (first 120 seconds)
            if elapsed >= 120:
                recent_ntp = pipeline.ntp_collector.get_recent_measurements(window_seconds=15)
                if recent_ntp:
                    ntp_timestamp, ntp_offset, ntp_uncertainty = recent_ntp[-1]

                    # Only evaluate if this is a NEW NTP measurement (not one we already used)
                    if last_evaluated_ntp_timestamp is None or abs(ntp_timestamp - last_evaluated_ntp_timestamp) > 1.0:
                        # Check if NTP is recent enough (within 15 seconds)
                        if abs(current_time - ntp_timestamp) < 15:
                            has_ntp = True
                            ntp_ground_truth_count += 1
                            last_evaluated_ntp_timestamp = ntp_timestamp
                            ntp_offset_ms = ntp_offset * 1000
                            ntp_uncertainty_ms = ntp_uncertainty * 1000
                            chronotick_error = abs(correction.offset_correction - ntp_offset) * 1000
                            system_error = abs(0.0 - ntp_offset) * 1000
        except:
            pass

        # Write summary
        dataset_size = len(pipeline.dataset_manager.get_recent_measurements())
        corrections_applied = pipeline.stats.get('ntp_corrections_applied', 0)

        summary_writer.writerow([
            current_time,
            elapsed,
            correction.offset_correction * 1000,
            correction.drift_rate * 1e6,
            correction.source,
            ntp_offset_ms,
            ntp_uncertainty_ms,
            has_ntp,
            chronotick_error,
            system_error,
            dataset_size,
            corrections_applied
        ])
        summary_file.flush()

        # Console
        gt_marker = "✅ GT" if has_ntp else ""
        print(f"[{int(elapsed):4d}s] "
              f"Offset: {correction.offset_correction*1000:>8.2f}ms, "
              f"DS: {dataset_size:3d}, "
              f"Corr: {corrections_applied:3d}  "
              f"{gt_marker}")

        if i < iterations - 1:
            time.sleep(args.interval)

finally:
    summary_file.close()
    client_logger.close()
    correction_logger.close()

print()
print("=" * 80)
print(f"TEST COMPLETE - Method: {args.method.upper()}")
print("=" * 80)
print(f"\nNTP ground truth samples: {ntp_ground_truth_count}")
print(f"\nResults saved:")
print(f"  Summary: {summary_csv}")
print(f"  Client predictions: {results_dir / f'client_predictions_{args.method}_{timestamp_str}.csv'}")
print(f"  Dataset corrections: {results_dir / f'dataset_corrections_{args.method}_{timestamp_str}.csv'}")
print()

# Cleanup
engine.shutdown()
pipeline.system_metrics.stop_collection()
Path(temp_config_path).unlink()
print("✓ Complete")
