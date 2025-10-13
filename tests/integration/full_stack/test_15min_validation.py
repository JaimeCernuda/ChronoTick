#!/usr/bin/env python3
"""
Full-Stack 15-Minute Integration Test

Tests the complete ChronoTick system using the Python client library:
1. Starts ChronoTick daemon
2. Waits for warmup
3. Makes continuous requests for 15 minutes
4. Validates time corrections with NTP ground truth
5. Generates summary report

Usage:
    python test_15min_validation.py
"""

import sys
import time
import csv
import logging
from pathlib import Path
from datetime import datetime

# Add clients to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "clients" / "python"))

# Import ChronoTick client
from client import ChronoTickClient
from shm_config import ChronoTickSHMConfig

# Add server to path for NTP client (for ground truth validation)
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "server" / "src"))
from chronotick.inference.ntp_client import NTPClient

logger = logging.getLogger(__name__)

def main():
    print("=" * 80)
    print("CHRONOTICK FULL-STACK 15-MINUTE VALIDATION TEST")
    print("=" * 80)
    print()

    # Configuration
    test_duration = 900  # 15 minutes
    sample_interval = 10  # 10 seconds
    warmup_duration = 180  # 3 minutes

    # Results directory
    results_dir = Path("results/integration/full_stack")
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_csv = results_dir / f"validation_{timestamp_str}.csv"

    print(f"Configuration:")
    print(f"  Duration: {test_duration}s ({test_duration//60} min)")
    print(f"  Sample interval: {sample_interval}s")
    print(f"  Warmup: {warmup_duration}s ({warmup_duration//60} min)")
    print(f"  Results: {results_csv}")
    print()

    # Initialize client
    print("Initializing ChronoTick client...")
    try:
        client = ChronoTickClient()
        print("✓ Client initialized")
    except Exception as e:
        print(f"✗ Failed to initialize client: {e}")
        print()
        print("Please ensure ChronoTick daemon is running:")
        print("  uv run chronotick-daemon --config configs/config_enhanced_features.yaml")
        sys.exit(1)

    # Initialize NTP client for ground truth
    print("Initializing NTP client for ground truth...")
    ntp_client = NTPClient(
        servers=['time.google.com', 'time.cloudflare.com', 'time.nist.gov'],
        poll_interval=60  # Poll every 60 seconds for validation
    )
    ntp_client.start()
    print("✓ NTP client initialized")
    print()

    # Warmup
    print(f"Warmup phase: {warmup_duration}s...")
    for i in range(warmup_duration):
        try:
            offset, drift, uncertainty, confidence, source = client.get_time()
            if i % 30 == 0:
                print(f"  [{i}/{warmup_duration}s] offset={offset*1000:.2f}ms, source={source}")
        except Exception as e:
            if i % 30 == 0:
                print(f"  [{i}/{warmup_duration}s] Error: {e}")
        time.sleep(1)

    print("✓ Warmup complete")
    print()

    # Open results CSV
    csv_file = open(results_csv, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'timestamp', 'elapsed_seconds',
        'chronotick_offset_ms', 'chronotick_drift_us_per_s', 'chronotick_uncertainty_ms',
        'chronotick_confidence', 'chronotick_source',
        'ntp_ground_truth_offset_ms', 'ntp_uncertainty_ms', 'has_ntp',
        'chronotick_error_ms', 'system_error_ms'
    ])

    # Test phase
    print("=" * 80)
    print(f"RUNNING TEST: {test_duration//60} MINUTES")
    print("=" * 80)
    print()

    start_time = time.time()
    iterations = test_duration // sample_interval
    ntp_ground_truth_count = 0
    last_evaluated_ntp_timestamp = None

    total_chronotick_error = 0.0
    total_system_error = 0.0
    error_count = 0

    try:
        for i in range(iterations):
            current_time = time.time()
            elapsed = current_time - start_time

            # Get ChronoTick correction
            try:
                offset, drift, uncertainty, confidence, source = client.get_time()
            except Exception as e:
                print(f"[{int(elapsed):4d}s] Error getting time: {e}")
                time.sleep(sample_interval)
                continue

            # Get NTP ground truth
            has_ntp = False
            ntp_offset_ms = ""
            ntp_uncertainty_ms = ""
            chronotick_error = ""
            system_error = ""

            try:
                # Skip evaluations during early warmup
                if elapsed >= 120:
                    recent_ntp = ntp_client.get_recent_measurements(window_seconds=90)
                    if recent_ntp:
                        ntp_timestamp, ntp_offset, ntp_uncertainty = recent_ntp[-1]

                        # Only evaluate if this is a NEW NTP measurement
                        if last_evaluated_ntp_timestamp is None or abs(ntp_timestamp - last_evaluated_ntp_timestamp) > 1.0:
                            # Check if NTP is recent enough (within 90 seconds)
                            if abs(current_time - ntp_timestamp) < 90:
                                has_ntp = True
                                ntp_ground_truth_count += 1
                                last_evaluated_ntp_timestamp = ntp_timestamp
                                ntp_offset_ms = ntp_offset * 1000
                                ntp_uncertainty_ms = ntp_uncertainty * 1000
                                chronotick_error = abs(offset - ntp_offset) * 1000
                                system_error = abs(0.0 - ntp_offset) * 1000

                                # Accumulate errors
                                total_chronotick_error += chronotick_error
                                total_system_error += system_error
                                error_count += 1
            except Exception as e:
                pass

            # Write to CSV
            csv_writer.writerow([
                current_time,
                elapsed,
                offset * 1000,
                drift * 1e6,
                uncertainty * 1000,
                confidence,
                source,
                ntp_offset_ms,
                ntp_uncertainty_ms,
                has_ntp,
                chronotick_error,
                system_error
            ])
            csv_file.flush()

            # Console
            gt_marker = "✅ GT" if has_ntp else ""
            print(f"[{int(elapsed):4d}s] "
                  f"Offset: {offset*1000:>8.2f}ms, "
                  f"Unc: {uncertainty*1000:>6.2f}ms, "
                  f"Src: {source:>6s}  "
                  f"{gt_marker}")

            if i < iterations - 1:
                time.sleep(sample_interval)

    finally:
        csv_file.close()
        ntp_client.stop()

    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print()
    print(f"NTP ground truth samples: {ntp_ground_truth_count}")

    if error_count > 0:
        mean_chronotick_error = total_chronotick_error / error_count
        mean_system_error = total_system_error / error_count
        improvement = ((mean_system_error - mean_chronotick_error) / mean_system_error) * 100

        print()
        print("Error Analysis:")
        print(f"  ChronoTick mean error: {mean_chronotick_error:.2f}ms")
        print(f"  System clock mean error: {mean_system_error:.2f}ms")
        print(f"  Improvement: {improvement:.1f}%")

    print()
    print(f"Results saved: {results_csv}")
    print("✓ Complete")

if __name__ == "__main__":
    main()
