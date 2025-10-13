#!/usr/bin/env python3
"""
ChronoTick Overnight Validation Test

Runs for extended period collecting:
- System time (every 10 seconds)
- ChronoTick corrected time (every 10 seconds)
- NTP ground truth measurements (every 2 minutes)

Logs all data to CSV for analysis tomorrow.
"""

import sys
import time
import csv
import logging
import signal
from pathlib import Path
from datetime import datetime, timedelta

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import chronotick
from chronotick_inference.ntp_client import NTPClient, NTPConfig


# Configuration
SAMPLE_INTERVAL = 10  # Sample ChronoTick every 10 seconds
NTP_INTERVAL = 120    # Sample NTP every 2 minutes (ground truth)
OUTPUT_CSV = "/tmp/chronotick_overnight_validation.csv"
LOG_FILE = "/tmp/chronotick_overnight_validation.log"

# Global flag for graceful shutdown
running = True


def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global running
    print("\n⚠️  Received interrupt signal - shutting down gracefully...")
    logging.info("Received interrupt signal - shutting down")
    running = False


def setup_logging():
    """Configure logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-7s | %(message)s',
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler()
        ]
    )


def setup_ntp_client():
    """Initialize NTP client for ground truth measurements"""
    config = NTPConfig(
        servers=['pool.ntp.org', 'time.nist.gov', 'time.google.com'],
        timeout_seconds=2.0,
        max_acceptable_uncertainty=0.100,
        min_stratum=1
    )
    return NTPClient(config)


def write_csv_header(csv_file):
    """Write CSV header"""
    writer = csv.writer(csv_file)
    writer.writerow([
        'timestamp',
        'datetime',
        'system_time',
        'chronotick_time',
        'chronotick_offset_ms',
        'chronotick_uncertainty_ms',
        'chronotick_confidence',
        'ntp_offset_ms',
        'ntp_uncertainty_ms',
        'ntp_server',
        'sample_type'
    ])
    csv_file.flush()


def main():
    global running

    # Setup
    setup_logging()
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print("=" * 80)
    print("CHRONOTICK OVERNIGHT VALIDATION TEST")
    print("=" * 80)
    print()
    print(f"📊 Sample interval: {SAMPLE_INTERVAL}s (ChronoTick)")
    print(f"🌐 NTP interval: {NTP_INTERVAL}s (ground truth)")
    print(f"📁 Output CSV: {OUTPUT_CSV}")
    print(f"📝 Log file: {LOG_FILE}")
    print()

    # Start ChronoTick
    print("🚀 Starting ChronoTick...")
    config_path = str(Path(__file__).parent.parent / "chronotick_inference" / "config_complete.yaml")

    if not chronotick.start(config_path=config_path, auto_config=False):
        print("❌ FAILED: Could not start ChronoTick")
        logging.error("Failed to start ChronoTick")
        return 1

    print("✅ ChronoTick started successfully")
    logging.info("ChronoTick started successfully")

    # Wait for warmup
    warmup_duration = 60
    print(f"\n⏳ Waiting {warmup_duration}s for warmup...")
    time.sleep(warmup_duration)
    print("✅ Warmup complete\n")
    logging.info("Warmup complete - starting data collection")

    # Initialize NTP client
    print("🌐 Initializing NTP client...")
    ntp_client = setup_ntp_client()
    print("✅ NTP client ready\n")

    # Open CSV file
    csv_file = open(OUTPUT_CSV, 'w', newline='')
    write_csv_header(csv_file)

    print("=" * 80)
    print("📈 DATA COLLECTION STARTED")
    print("=" * 80)
    print(f"Press Ctrl+C to stop gracefully")
    print()

    # Tracking
    sample_count = 0
    ntp_sample_count = 0
    last_ntp_time = 0
    start_time = time.time()

    try:
        while running:
            current_time = time.time()
            timestamp = current_time
            dt = datetime.fromtimestamp(timestamp)

            # Get system time
            system_time = current_time

            # Get ChronoTick corrected time
            # Note: Public API uses default 100ms timeout internally
            try:
                corrected = chronotick.time_detailed()
                chronotick_time = corrected.timestamp
                chronotick_offset_ms = corrected.offset_correction * 1000
                chronotick_uncertainty_ms = corrected.uncertainty * 1000 if corrected.uncertainty else None
                chronotick_confidence = corrected.confidence if corrected.confidence else None
            except Exception as e:
                logging.error(f"ChronoTick error: {e}")
                chronotick_time = None
                chronotick_offset_ms = None
                chronotick_uncertainty_ms = None
                chronotick_confidence = None

            # Check if we need NTP measurement
            ntp_offset_ms = None
            ntp_uncertainty_ms = None
            ntp_server = None
            sample_type = 'regular'

            if current_time - last_ntp_time >= NTP_INTERVAL:
                print(f"🌐 [{dt.strftime('%H:%M:%S')}] Taking NTP measurement (#{ntp_sample_count + 1})...")
                logging.info(f"Taking NTP measurement #{ntp_sample_count + 1}")

                try:
                    ntp_measurement = ntp_client.get_best_measurement()
                    if ntp_measurement:
                        ntp_offset_ms = ntp_measurement.offset * 1000
                        ntp_uncertainty_ms = ntp_measurement.uncertainty * 1000
                        ntp_server = ntp_measurement.server
                        sample_type = 'ntp'
                        ntp_sample_count += 1
                        last_ntp_time = current_time

                        print(f"   ✓ NTP offset: {ntp_offset_ms:.3f}ms from {ntp_server}")
                        logging.info(f"NTP measurement: offset={ntp_offset_ms:.3f}ms, server={ntp_server}")
                except Exception as e:
                    logging.error(f"NTP measurement failed: {e}")
                    print(f"   ✗ NTP measurement failed: {e}")

            # Write to CSV
            writer = csv.writer(csv_file)
            writer.writerow([
                timestamp,
                dt.isoformat(),
                system_time,
                chronotick_time if chronotick_time else '',
                chronotick_offset_ms if chronotick_offset_ms is not None else '',
                chronotick_uncertainty_ms if chronotick_uncertainty_ms is not None else '',
                chronotick_confidence if chronotick_confidence is not None else '',
                ntp_offset_ms if ntp_offset_ms is not None else '',
                ntp_uncertainty_ms if ntp_uncertainty_ms is not None else '',
                ntp_server if ntp_server else '',
                sample_type
            ])
            csv_file.flush()

            sample_count += 1

            # Progress update every minute
            if sample_count % (60 // SAMPLE_INTERVAL) == 0:
                elapsed = current_time - start_time
                elapsed_str = str(timedelta(seconds=int(elapsed)))
                print(f"⏱️  [{dt.strftime('%H:%M:%S')}] Runtime: {elapsed_str} | "
                      f"Samples: {sample_count} | NTP: {ntp_sample_count}")

                if chronotick_offset_ms is not None and chronotick_uncertainty_ms is not None:
                    print(f"   ChronoTick offset: {chronotick_offset_ms:.3f}ms ± {chronotick_uncertainty_ms:.3f}ms")
                elif chronotick_offset_ms is not None:
                    print(f"   ChronoTick offset: {chronotick_offset_ms:.3f}ms (uncertainty unavailable)")

            # Sleep until next sample
            next_sample_time = timestamp + SAMPLE_INTERVAL
            sleep_duration = max(0, next_sample_time - time.time())
            time.sleep(sleep_duration)

    except Exception as e:
        logging.error(f"Error in collection loop: {e}")
        print(f"\n❌ Error: {e}")

    finally:
        # Cleanup
        print("\n" + "=" * 80)
        print("🛑 STOPPING DATA COLLECTION")
        print("=" * 80)
        print()

        elapsed = time.time() - start_time
        elapsed_str = str(timedelta(seconds=int(elapsed)))

        print(f"📊 Collection Summary:")
        print(f"   Runtime: {elapsed_str}")
        print(f"   Total samples: {sample_count}")
        print(f"   NTP measurements: {ntp_sample_count}")
        print(f"   Output file: {OUTPUT_CSV}")
        print()

        logging.info(f"Data collection complete: {sample_count} samples, {ntp_sample_count} NTP measurements")

        csv_file.close()

        print("🛑 Stopping ChronoTick...")
        chronotick.stop()
        print("✅ ChronoTick stopped")
        logging.info("ChronoTick stopped")

        print()
        print("=" * 80)
        print("✅ OVERNIGHT VALIDATION TEST COMPLETE")
        print("=" * 80)
        print()
        print("📊 To analyze results tomorrow, run:")
        print(f"   uv run python scripts/analyze_overnight.py {OUTPUT_CSV}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
