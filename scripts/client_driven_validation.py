#!/usr/bin/env python3
"""
Client-Driven Long-Term Validation Test

Simulates a real client that:
- Requests time from ChronoTick system regularly
- Gets NTP measurements periodically for ground truth comparison
- Logs all data to CSV for analysis

This validates that ChronoTick provides better time accuracy than system clock alone.
"""

import sys
import time
import csv
import ntplib
import argparse
from pathlib import Path
from datetime import datetime

# Add server/src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "server" / "src"))

from chronotick.inference.engine import ChronoTickInferenceEngine
from chronotick.inference.real_data_pipeline import RealDataPipeline
from chronotick.inference.tsfm_model_wrapper import create_model_wrappers

# Parse command-line arguments
parser = argparse.ArgumentParser(description='ChronoTick Client-Driven Validation Test')
parser.add_argument('--config', default='configs/config_stable_clock.yaml',
                    help='Path to ChronoTick configuration file')
parser.add_argument('--ntp-server', default='time.google.com',
                    help='NTP server for ground truth (supports IP:port format, e.g., 172.20.1.1:8123)')
parser.add_argument('--duration', type=int, default=480,
                    help='Test duration in minutes (default: 480 = 8 hours)')
parser.add_argument('--sample-interval', type=int, default=10,
                    help='Sample interval in seconds (default: 10)')
parser.add_argument('--ntp-interval', type=int, default=120,
                    help='NTP measurement interval in seconds (default: 120 = 2 minutes)')
args = parser.parse_args()

print("=" * 80)
print("CLIENT-DRIVEN LONG-TERM VALIDATION TEST")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Configuration
TEST_DURATION_MINUTES = args.duration
SAMPLE_INTERVAL_SECONDS = args.sample_interval
NTP_INTERVAL_SECONDS = args.ntp_interval
NTP_SERVER = args.ntp_server
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
CSV_PATH = f"/tmp/chronotick_client_validation_{TIMESTAMP}.csv"

# Parse NTP server address and port
if ':' in NTP_SERVER:
    ntp_host, ntp_port_str = NTP_SERVER.rsplit(':', 1)
    ntp_port = int(ntp_port_str)
else:
    ntp_host = NTP_SERVER
    ntp_port = 123  # Default NTP port

print(f"Configuration:")
print(f"  Test duration: {TEST_DURATION_MINUTES} minutes")
print(f"  Sample interval: {SAMPLE_INTERVAL_SECONDS} seconds")
print(f"  NTP interval: {NTP_INTERVAL_SECONDS} seconds ({NTP_INTERVAL_SECONDS//60} minutes)")
print(f"  NTP server: {ntp_host}:{ntp_port}")
print(f"  ChronoTick config: {args.config}")
print(f"  CSV output: {CSV_PATH}")
print()

# Initialize ChronoTick system
config_path = args.config

print("Initializing ChronoTick system...")
engine = ChronoTickInferenceEngine(config_path)
engine.initialize_models()
print("✓ Models loaded")

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
print("✓ Pipeline initialized")
print()

# Wait for warmup
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
print("✓ Warmup complete")
time.sleep(5)  # Wait for warmup timer
print()

# Initialize NTP client for ground truth measurements
ntp_client = ntplib.NTPClient()

# CSV setup
csv_file = open(CSV_PATH, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow([
    'sample_number',
    'elapsed_seconds',
    'datetime',
    'system_time',
    'chronotick_time',
    'chronotick_offset_ms',
    'chronotick_uncertainty_ms',
    'chronotick_confidence',
    'chronotick_source',
    'ntp_time',
    'ntp_offset_ms',
    'ntp_uncertainty_ms',
    'ntp_server',
    'has_ntp'
])

print("=" * 80)
print("STARTING CLIENT-DRIVEN TEST")
print("=" * 80)
print()

test_start = time.time()
test_duration = TEST_DURATION_MINUTES * 60
sample_number = 0
last_ntp_time = 0
ntp_sample_count = 0

try:
    while time.time() - test_start < test_duration:
        sample_number += 1
        elapsed = time.time() - test_start

        # Get system time (client's clock)
        system_time = time.time()

        # Get ChronoTick correction
        try:
            correction = pipeline.get_real_clock_correction(system_time)

            # Calculate corrected time
            time_delta = system_time - correction.prediction_time
            chronotick_time = system_time + correction.offset_correction + (correction.drift_rate * time_delta)

            chronotick_offset_ms = correction.offset_correction * 1000
            chronotick_uncertainty_ms = correction.offset_uncertainty * 1000
            chronotick_confidence = correction.confidence
            chronotick_source = correction.source

        except Exception as e:
            print(f"⚠️  ChronoTick error at {elapsed:.1f}s: {e}")
            chronotick_time = system_time
            chronotick_offset_ms = 0.0
            chronotick_uncertainty_ms = 0.0
            chronotick_confidence = 0.0
            chronotick_source = "error"

        # Get NTP measurement every NTP_INTERVAL_SECONDS
        has_ntp = False
        ntp_time = ""
        ntp_offset_ms = ""
        ntp_uncertainty_ms = ""
        ntp_server_used = ""

        if elapsed - last_ntp_time >= NTP_INTERVAL_SECONDS:
            try:
                # Get NTP response using parsed host and port
                response = ntp_client.request(ntp_host, port=ntp_port, version=3, timeout=2)

                # NTP time is the reference time from the server
                # tx_time is when server sent the response
                ntp_time = response.tx_time

                # Offset is how much our clock differs from NTP
                # Positive = our clock is ahead, Negative = our clock is behind
                ntp_offset_ms = response.offset * 1000
                ntp_uncertainty_ms = response.root_delay * 1000 / 2  # Approximate uncertainty
                ntp_server_used = f"{ntp_host}:{ntp_port}"
                has_ntp = True
                last_ntp_time = elapsed
                ntp_sample_count += 1

                print(f"[{elapsed:6.1f}s] 📡 NTP: offset={ntp_offset_ms:>8.2f}ms, "
                      f"ChronoTick: offset={chronotick_offset_ms:>8.2f}ms ± {chronotick_uncertainty_ms:>6.2f}ms, "
                      f"source={chronotick_source}")

            except Exception as e:
                # NTP failed, but continue test
                if ntp_sample_count == 0:
                    print(f"⚠️  NTP query failed at {elapsed:.1f}s: {e}")

        # Log to CSV
        csv_writer.writerow([
            sample_number,
            elapsed,
            datetime.fromtimestamp(system_time).isoformat(),
            system_time,
            chronotick_time,
            chronotick_offset_ms,
            chronotick_uncertainty_ms,
            chronotick_confidence,
            chronotick_source,
            ntp_time,
            ntp_offset_ms,
            ntp_uncertainty_ms,
            ntp_server_used,
            has_ntp
        ])
        csv_file.flush()

        # Console output (reduced frequency)
        if sample_number % 6 == 0:  # Every minute
            print(f"[{elapsed:6.1f}s] ChronoTick: offset={chronotick_offset_ms:>8.2f}ms, "
                  f"uncertainty=±{chronotick_uncertainty_ms:>6.2f}ms, "
                  f"source={chronotick_source:10s}, "
                  f"confidence={chronotick_confidence:.2f}")

        time.sleep(SAMPLE_INTERVAL_SECONDS)

finally:
    csv_file.close()

print()
print("=" * 80)
print("TEST COMPLETE")
print("=" * 80)
print()
print(f"Total samples: {sample_number}")
print(f"NTP measurements: {ntp_sample_count}")
print(f"CSV saved to: {CSV_PATH}")
print()

# Cleanup
print("Cleaning up...")
engine.shutdown()
pipeline.system_metrics.stop_collection()
print("✓ Done")
print()
print(f"Run analysis with: uv run python scripts/analyze_client_validation.py {CSV_PATH}")
