#!/usr/bin/env python3
"""
7-Minute Validation Test - Tests past the 5-minute timeout point
Verifies that increased IPC timeout (300ms) fixes GIL contention issues.
"""

import sys
import time
import csv
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from chronotick_inference.engine import ChronoTickInferenceEngine
from chronotick_inference.real_data_pipeline import RealDataPipeline
from chronotick_inference.tsfm_model_wrapper import create_model_wrappers

print("=" * 80)
print("7-MINUTE VALIDATION TEST - IPC Timeout Fix Verification")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Initialize pipeline
config_path = "chronotick_inference/config_complete.yaml"

print("Initializing ChronoTick system...")
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

# Run for 7 minutes (420 seconds) with 10-second intervals
print("=" * 80)
print("RUNNING 7-MINUTE TEST - Monitoring for timeouts past 5-minute mark")
print("=" * 80)
print()

test_duration = 420  # 7 minutes
interval = 10  # Every 10 seconds
iterations = test_duration // interval

# CSV output
csv_path = "/tmp/chronotick_7min_validation.csv"
csv_file = open(csv_path, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    'timestamp', 'datetime', 'system_time', 'chronotick_time',
    'chronotick_offset_ms', 'chronotick_uncertainty_ms', 'chronotick_confidence',
    'ntp_offset_ms', 'ntp_uncertainty_ms', 'ntp_server', 'sample_type'
])

results = []
timeout_count = 0
ml_count = 0

try:
    for i in range(iterations):
        current_time = time.time()
        correction = pipeline.get_real_clock_correction(current_time)

        # Track ML vs fallback usage
        if correction.source in ['cpu', 'gpu', 'fusion']:
            ml_count += 1

        # Track timeouts (offset=0.0 indicates timeout/fallback)
        if correction.offset_correction == 0.0:
            timeout_count += 1

        # Calculate corrected time
        time_delta = current_time - correction.prediction_time
        corrected_time = current_time + correction.offset_correction + (correction.drift_rate * time_delta)

        # Get NTP info if available
        ntp_offset = ""
        ntp_uncertainty = ""
        ntp_server = ""
        sample_type = "regular"

        try:
            recent_ntp = pipeline.ntp_collector.get_recent_measurements(count=1)
            if recent_ntp and abs(current_time - recent_ntp[0]['timestamp']) < 5:
                ntp_offset = recent_ntp[0]['offset'] * 1000
                ntp_uncertainty = recent_ntp[0]['uncertainty'] * 1000
                ntp_server = recent_ntp[0]['server']
                sample_type = "ntp"
        except:
            pass

        # Write to CSV
        csv_writer.writerow([
            current_time,
            datetime.fromtimestamp(current_time).isoformat(),
            current_time,
            corrected_time,
            correction.offset_correction * 1000,
            correction.offset_uncertainty * 1000,
            correction.confidence,
            ntp_offset,
            ntp_uncertainty,
            ntp_server,
            sample_type
        ])
        csv_file.flush()

        # Console output with timeout indicator
        elapsed = (i + 1) * interval
        timeout_marker = "⚠️  TIMEOUT" if correction.offset_correction == 0.0 else ""

        print(f"[{elapsed:3d}s / 7min] "
              f"Offset: {correction.offset_correction*1000:>8.2f}ms, "
              f"Uncertainty: {correction.offset_uncertainty*1000:>8.2f}ms, "
              f"Source: {correction.source:10s}, "
              f"Confidence: {correction.confidence:.2f}  "
              f"{timeout_marker}")

        results.append({
            'elapsed': elapsed,
            'offset': correction.offset_correction,
            'uncertainty': correction.offset_uncertainty,
            'source': correction.source,
            'confidence': correction.confidence
        })

        if i < iterations - 1:
            time.sleep(interval)

finally:
    csv_file.close()

print()
print("=" * 80)
print("TEST COMPLETE - 7-Minute Results")
print("=" * 80)
print()

# Analysis
print(f"Total samples: {len(results)}")
print(f"ML predictions (cpu/gpu/fusion): {ml_count} ({ml_count/len(results)*100:.1f}%)")
print(f"Timeouts (offset=0.0): {timeout_count} ({timeout_count/len(results)*100:.1f}%)")
print()

# Source breakdown
sources = {}
for r in results:
    source = r['source']
    sources[source] = sources.get(source, 0) + 1

print("Prediction Sources:")
for source, count in sorted(sources.items()):
    percentage = (count / len(results)) * 100
    print(f"  {source:15s}: {count:3d} times ({percentage:5.1f}%)")
print()

# Check if we passed the 5-minute mark without timeouts
five_min_mark = 300  # 5 minutes
samples_past_5min = [r for r in results if r['elapsed'] > five_min_mark]
timeouts_past_5min = [r for r in samples_past_5min if r['offset'] == 0.0]

print(f"Samples past 5-minute mark: {len(samples_past_5min)}")
print(f"Timeouts past 5-minute mark: {len(timeouts_past_5min)}")
print()

if timeout_count == 0:
    print("✅ SUCCESS: No timeouts observed - IPC timeout fix is working!")
elif len(timeouts_past_5min) == 0:
    print("✅ SUCCESS: No timeouts past 5-minute mark - fix resolved the issue!")
else:
    print("❌ FAILURE: Timeouts still occurring past 5-minute mark")
print()

print(f"CSV results saved to: {csv_path}")
print()

# Cleanup
print("Cleaning up...")
engine.shutdown()
pipeline.system_metrics.stop_collection()
print("✓ Test complete")
