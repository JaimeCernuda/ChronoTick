#!/usr/bin/env python3
"""
Client-Driven Long-Term Validation Test V3 - Fix 1 + Fix 2 Implementation

IMPROVEMENTS OVER V2:
1. **FIX 1**: Use drift_rate in ChronoTick time calculations
2. **FIX 2**: NTP-anchored time walking (chrony-inspired approach)
3. Enhanced CSV with drift_rate, prediction_time, time_since_ntp columns
4. Improved logging showing Fix 1 vs Fix 2 behavior
5. Dual output: Both fix approaches logged for comparison

V2 FEATURES (preserved):
- Multi-server NTP averaging with MAD outlier rejection
- Parallel NTP queries
- 1Hz ChronoTick/System sampling, 60s NTP interval

Simulates a real client testing TWO approaches:
- **Fix 1**: system_time + offset + drift * time_delta (still tied to system clock)
- **Fix 2**: last_ntp_time + elapsed + drift * elapsed (independent from system clock)
"""

import sys
import time
import csv
import ntplib
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add server/src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "server" / "src"))

from chronotick.inference.engine import ChronoTickInferenceEngine
from chronotick.inference.real_data_pipeline import RealDataPipeline
from chronotick.inference.tsfm_model_wrapper import create_model_wrappers

# =============================================================================
# MULTI-SERVER NTP CLIENT (like ChronoTick internal)
# =============================================================================

class MultiServerNTPClient:
    """
    Query multiple NTP servers in parallel and average with outlier rejection.
    Uses same algorithm as ChronoTick's internal NTP client.
    """

    def __init__(self, servers, timeout=2.0, max_workers=5):
        """
        Args:
            servers: List of (host, port) tuples
            timeout: Query timeout in seconds
            max_workers: Max parallel workers
        """
        self.servers = servers
        self.timeout = timeout
        self.max_workers = max_workers
        self.ntp_client = ntplib.NTPClient()

    def query_single_server(self, host, port):
        """Query a single NTP server"""
        try:
            response = self.ntp_client.request(host, port=port, version=3, timeout=self.timeout)
            return {
                'host': host,
                'port': port,
                'offset_ms': response.offset * 1000,
                'delay_ms': response.root_delay * 1000,
                'uncertainty_ms': response.root_delay * 1000 / 2,
                'success': True
            }
        except Exception as e:
            return {
                'host': host,
                'port': port,
                'error': str(e),
                'success': False
            }

    def query_all_servers(self):
        """Query all servers in parallel"""
        results = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all queries
            futures = {executor.submit(self.query_single_server, host, port): (host, port)
                      for host, port in self.servers}

            # Collect results
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result['success']:
                        results.append(result)
                except Exception as e:
                    pass  # Skip failed queries

        return results

    def average_with_outlier_rejection(self, measurements):
        """
        Average multiple NTP measurements using MAD outlier rejection.

        Algorithm (same as ChronoTick internal):
        1. Calculate median offset
        2. Calculate MAD (Median Absolute Deviation)
        3. Reject measurements > 3×MAD from median
        4. Average remaining measurements
        5. Return avg offset, uncertainty, and stats
        """
        if len(measurements) == 0:
            return None

        if len(measurements) == 1:
            # Single measurement - no averaging needed
            m = measurements[0]
            return {
                'offset_ms': m['offset_ms'],
                'uncertainty_ms': m['uncertainty_ms'],
                'delay_ms': m['delay_ms'],
                'n_combined': 1,
                'n_total': 1,
                'n_rejected': 0,
                'rejected_servers': [],
                'mad_ms': 0.0
            }

        # Extract offsets
        offsets = np.array([m['offset_ms'] for m in measurements])

        # Calculate median offset (robust central tendency)
        median_offset = np.median(offsets)

        # Calculate MAD (Median Absolute Deviation) - robust dispersion measure
        mad = np.median(np.abs(offsets - median_offset))

        # MAD threshold: 3 × MAD ≈ 3σ for normal distribution
        mad_threshold = 3.0 * mad if mad > 0.000001 else 3.0  # Fallback: 3ms

        # Filter outliers
        mask = np.abs(offsets - median_offset) <= mad_threshold
        filtered_measurements = [m for m, keep in zip(measurements, mask) if keep]
        filtered_offsets = offsets[mask]

        # Rejected servers
        rejected = [m for m, keep in zip(measurements, mask) if not keep]
        rejected_servers = [f"{m['host']}:{m['port']}" for m in rejected]

        if len(filtered_measurements) == 0:
            # All rejected? Fall back to median
            return {
                'offset_ms': median_offset,
                'uncertainty_ms': mad,
                'delay_ms': np.mean([m['delay_ms'] for m in measurements]),
                'n_combined': 0,
                'n_total': len(measurements),
                'n_rejected': len(measurements),
                'rejected_servers': rejected_servers,
                'mad_ms': mad
            }

        # Calculate average of filtered measurements
        avg_offset = np.mean(filtered_offsets)

        # Uncertainty: std of filtered measurements
        avg_uncertainty = np.std(filtered_offsets) if len(filtered_offsets) > 1 else filtered_measurements[0]['uncertainty_ms']

        # Average delay
        avg_delay = np.mean([m['delay_ms'] for m in filtered_measurements])

        return {
            'offset_ms': avg_offset,
            'uncertainty_ms': avg_uncertainty,
            'delay_ms': avg_delay,
            'n_combined': len(filtered_measurements),
            'n_total': len(measurements),
            'n_rejected': len(rejected),
            'rejected_servers': rejected_servers,
            'mad_ms': mad
        }

    def get_averaged_measurement(self):
        """
        Get averaged NTP measurement with outlier rejection.
        Returns dict with offset, uncertainty, and stats.
        """
        measurements = self.query_all_servers()

        if len(measurements) == 0:
            return None

        return self.average_with_outlier_rejection(measurements)


# =============================================================================
# SMART SERVER CONFIGURATION
# =============================================================================

def parse_ntp_servers(ntp_server_arg):
    """
    Parse NTP server argument and generate server list.

    Supports:
    1. Single proxy server (e.g., "172.20.1.1:8123") → generates 5 proxy ports
    2. Single domain (e.g., "time.google.com") → uses standard 5 servers
    3. Multiple servers comma-separated → uses as-is
    """

    # Check if it's a proxy server (IP address with port)
    if ':' in ntp_server_arg and ntp_server_arg.split(':')[0].replace('.', '').isdigit():
        # Proxy server - generate 5 ports
        host, base_port_str = ntp_server_arg.rsplit(':', 1)
        base_port = int(base_port_str)

        # ARES proxy mapping (based on config_experiment11_ares.yaml):
        # 8123 -> time.google.com
        # 8127 -> time.cloudflare.com
        # 8128 -> time.nist.gov
        # 8129 -> 0.pool.ntp.org
        # 8130 -> 1.pool.ntp.org

        if base_port == 8123:
            # User provided base port, generate the 5 ports
            ports = [8123, 8127, 8128, 8129, 8130]
        else:
            # Unknown base port, generate sequential
            ports = [base_port + i for i in range(5)]

        servers = [(host, port) for port in ports]
        print(f"  Detected PROXY configuration: {host}")
        print(f"  Using 5 proxy ports: {[p for p in ports]}")

    elif ',' in ntp_server_arg:
        # Multiple servers comma-separated
        server_strs = [s.strip() for s in ntp_server_arg.split(',')]
        servers = []
        for s in server_strs:
            if ':' in s:
                host, port_str = s.rsplit(':', 1)
                servers.append((host, int(port_str)))
            else:
                servers.append((s, 123))
        print(f"  Using {len(servers)} servers from argument")

    else:
        # Single domain name - use standard 5 servers
        servers = [
            ('time.google.com', 123),
            ('time.cloudflare.com', 123),
            ('time.nist.gov', 123),
            ('0.pool.ntp.org', 123),
            ('1.pool.ntp.org', 123)
        ]
        print(f"  Using standard 5 NTP servers (google, cloudflare, nist, pool)")

    return servers


# =============================================================================
# MAIN VALIDATION LOOP
# =============================================================================

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='ChronoTick Client-Driven Validation Test V2 (Multi-Server NTP)')
    parser.add_argument('--config', default='configs/config_stable_clock.yaml',
                        help='Path to ChronoTick configuration file')
    parser.add_argument('--ntp-server', default='time.google.com',
                        help='NTP server(s): single proxy (172.20.1.1:8123), domain (time.google.com), or comma-separated list')
    parser.add_argument('--duration', type=int, default=480,
                        help='Test duration in minutes (default: 480 = 8 hours)')
    parser.add_argument('--sample-interval', type=int, default=1,
                        help='Sample interval in seconds (default: 1 = every second)')
    parser.add_argument('--ntp-interval', type=int, default=60,
                        help='NTP measurement interval in seconds (default: 60 = every minute)')
    args = parser.parse_args()

    print("=" * 80)
    print("CLIENT-DRIVEN LONG-TERM VALIDATION TEST V2")
    print("Multi-Server NTP Averaging with MAD Outlier Rejection")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Configuration
    TEST_DURATION_MINUTES = args.duration
    SAMPLE_INTERVAL_SECONDS = args.sample_interval
    NTP_INTERVAL_SECONDS = args.ntp_interval
    TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
    CSV_PATH = f"/tmp/chronotick_client_validation_v2_{TIMESTAMP}.csv"

    # Parse NTP servers
    print("NTP Server Configuration:")
    ntp_servers = parse_ntp_servers(args.ntp_server)
    print()

    print(f"Configuration:")
    print(f"  Test duration: {TEST_DURATION_MINUTES} minutes")
    print(f"  Sample interval: {SAMPLE_INTERVAL_SECONDS} seconds (ChronoTick + System clock)")
    print(f"  NTP interval: {NTP_INTERVAL_SECONDS} seconds ({NTP_INTERVAL_SECONDS//60} minute)")
    print(f"  NTP servers: {len(ntp_servers)} servers (parallel queries)")
    print(f"  ChronoTick config: {args.config}")
    print(f"  CSV output: {CSV_PATH}")
    print()

    # Initialize Multi-Server NTP Client
    ntp_client = MultiServerNTPClient(ntp_servers, timeout=2.0, max_workers=5)

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
                print(f"  Warmup: {i}/{warmup_duration}s")
        except Exception as e:
            pass
        time.sleep(1)

    print("✓ Warmup complete")
    print()

    # Start validation test
    print("=" * 80)
    print(f"Started validation test at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test will run for {TEST_DURATION_MINUTES} minutes ({TEST_DURATION_MINUTES / 60:.1f} hours)")
    print("=" * 80)
    print()

    # CSV header (V3: Added new columns for Fix 1 & Fix 2 analysis + system clock drift)
    csv_file = open(CSV_PATH, 'w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        'sample_number',
        'elapsed_seconds',
        'datetime',
        'system_time',
        'chronotick_time_fix1',          # V3: Fix 1 (system_time + offset + drift*delta)
        'chronotick_time_fix2',          # V3: Fix 2 (ntp_anchor + elapsed + drift*elapsed)
        'chronotick_offset_ms',
        'chronotick_drift_rate',         # V3: NEW - drift rate in s/s
        'chronotick_drift_uncertainty',  # EXPERIMENT-14: Drift uncertainty in s/s
        'chronotick_drift_source',       # EXPERIMENT-14: Source of drift (timesfm_predicted/ntp_calculated)
        'chronotick_prediction_time',    # V3: NEW - when prediction was made
        'chronotick_uncertainty_ms',
        'chronotick_confidence',
        'chronotick_source',
        'time_since_ntp_s',              # V3: NEW - seconds since last NTP anchor
        'system_clock_drift_rate',       # DRIFT EXP: System clock drift in s/s
        'ntp_time',
        'ntp_offset_ms',
        'ntp_uncertainty_ms',
        'ntp_mad_ms',
        'ntp_n_combined',
        'ntp_n_total',
        'ntp_n_rejected',
        'ntp_rejected_servers',
        'has_ntp'
    ])

    # Test loop
    test_start_time = time.time()
    test_duration_seconds = TEST_DURATION_MINUTES * 60
    sample_number = 0
    last_ntp_time = 0
    ntp_sample_count = 0

    # V3: NTP anchor tracking for Fix 2 (chrony-inspired approach)
    last_ntp_true_time = None      # True time from last NTP measurement
    last_ntp_system_time = None    # System time when last NTP was taken
    last_ntp_timestamp = 0         # Track when last NTP was taken

    # DRIFT EXP: System clock drift tracking
    last_ntp_offset_ms = None      # Last NTP offset in ms
    last_ntp_elapsed = None        # Elapsed time at last NTP
    system_clock_drift_rate = 0.0  # System clock drift rate in s/s

    while True:
        elapsed = time.time() - test_start_time

        # Check if test duration reached
        if elapsed >= test_duration_seconds:
            print(f"\n✓ Test completed after {elapsed/60:.1f} minutes")
            break

        sample_number += 1

        # Get system time
        system_time = time.time()

        # Get ChronoTick time (V3: Implement Fix 1 + Fix 2)
        try:
            correction = pipeline.get_real_clock_correction(system_time)

            # Extract fields
            chronotick_offset_ms = correction.offset_correction * 1000
            chronotick_drift_rate = correction.drift_rate
            chronotick_drift_uncertainty = getattr(correction, 'drift_uncertainty', 0.0)  # EXPERIMENT-14
            chronotick_drift_source = "timesfm_predicted"  # EXPERIMENT-14: Only TimesFM (no fallbacks)
            chronotick_prediction_time = correction.prediction_time
            chronotick_uncertainty_ms = correction.offset_uncertainty * 1000
            chronotick_confidence = correction.confidence
            chronotick_source = correction.source

            # V3: Calculate time_delta for Fix 1
            time_delta = system_time - chronotick_prediction_time

            # V3: FIX 1 - System-based with drift (still tied to system clock)
            chronotick_time_fix1 = (system_time +
                                   correction.offset_correction +
                                   correction.drift_rate * time_delta)

            # V3: FIX 2 - NTP-anchored time walking (chrony-inspired, independent!)
            if last_ntp_true_time is not None and last_ntp_system_time is not None:
                elapsed_since_ntp = system_time - last_ntp_system_time
                chronotick_time_fix2 = (last_ntp_true_time +
                                       elapsed_since_ntp +
                                       correction.drift_rate * elapsed_since_ntp)
                time_since_ntp_s = elapsed_since_ntp
            else:
                # Fallback to Fix 1 if no NTP anchor yet
                chronotick_time_fix2 = chronotick_time_fix1
                time_since_ntp_s = 0.0

        except Exception as e:
            # Log the exception for debugging
            if sample_number % 60 == 0:  # Log every minute to avoid spam
                print(f"  ⚠️  ChronoTick prediction error at sample {sample_number}: {type(e).__name__}: {e}")
            chronotick_time_fix1 = system_time
            chronotick_time_fix2 = system_time
            chronotick_offset_ms = 0.0
            chronotick_drift_rate = 0.0
            chronotick_drift_uncertainty = 0.0  # EXPERIMENT-14
            chronotick_drift_source = "error"  # EXPERIMENT-14
            chronotick_prediction_time = system_time
            chronotick_uncertainty_ms = 0.0
            chronotick_confidence = 0.0
            chronotick_source = "error"
            time_since_ntp_s = 0.0

        # Get Multi-Server NTP measurement every NTP_INTERVAL_SECONDS
        has_ntp = False
        ntp_time = ""
        ntp_offset_ms = ""
        ntp_uncertainty_ms = ""
        ntp_mad_ms = ""
        ntp_n_combined = ""
        ntp_n_total = ""
        ntp_n_rejected = ""
        ntp_rejected_servers = ""

        if elapsed - last_ntp_time >= NTP_INTERVAL_SECONDS:
            try:
                print(f"[{elapsed:6.1f}s] Querying {len(ntp_servers)} NTP servers in parallel...")
                ntp_result = ntp_client.get_averaged_measurement()

                if ntp_result is not None:
                    # Calculate NTP reference time
                    # NTP offset tells us: true_time = system_time + offset
                    ntp_time = system_time + (ntp_result['offset_ms'] / 1000.0)

                    ntp_offset_ms = ntp_result['offset_ms']
                    ntp_uncertainty_ms = ntp_result['uncertainty_ms']
                    ntp_mad_ms = ntp_result['mad_ms']
                    ntp_n_combined = ntp_result['n_combined']
                    ntp_n_total = ntp_result['n_total']
                    ntp_n_rejected = ntp_result['n_rejected']
                    ntp_rejected_servers = ', '.join(ntp_result['rejected_servers'])

                    has_ntp = True
                    last_ntp_time = elapsed
                    ntp_sample_count += 1

                    # V3: Update NTP anchor for Fix 2
                    last_ntp_true_time = ntp_time  # True time from this measurement
                    last_ntp_system_time = system_time  # System time at measurement
                    last_ntp_timestamp = elapsed

                    # DRIFT EXP: Calculate system clock drift rate
                    if last_ntp_offset_ms is not None and last_ntp_elapsed is not None:
                        time_diff = elapsed - last_ntp_elapsed  # seconds
                        offset_diff = ntp_offset_ms - last_ntp_offset_ms  # ms
                        # drift rate in s/s = (offset_diff in ms / time_diff in s) / 1000
                        system_clock_drift_rate = (offset_diff / time_diff) / 1000.0

                    # Update tracking variables for next calculation
                    last_ntp_offset_ms = ntp_offset_ms
                    last_ntp_elapsed = elapsed

                    # V3: Enhanced logging showing both fixes
                    print(f"[{elapsed:6.1f}s] 📡 NTP: offset={ntp_offset_ms:>8.2f}ms ± {ntp_uncertainty_ms:.2f}ms "
                          f"(combined {ntp_n_combined}/{ntp_n_total}, MAD={ntp_mad_ms:.2f}ms)")
                    print(f"           ChronoTick (Fix1): offset={chronotick_offset_ms:>8.2f}ms, "
                          f"drift={chronotick_drift_rate*1e6:+.3f}μs/s, source={chronotick_source}")

                    # Calculate offsets for both fixes
                    fix1_offset_ms = (chronotick_time_fix1 - ntp_time) * 1000
                    fix2_offset_ms = (chronotick_time_fix2 - ntp_time) * 1000
                    print(f"           Fix1 vs NTP: {fix1_offset_ms:+.3f}ms | Fix2 vs NTP: {fix2_offset_ms:+.3f}ms")

                    if ntp_n_rejected > 0:
                        print(f"           ⚠️  Rejected {ntp_n_rejected} outlier(s): {ntp_rejected_servers}")

            except Exception as e:
                # NTP failed, but continue test
                if ntp_sample_count == 0:
                    print(f"⚠️  Multi-server NTP query failed at {elapsed:.1f}s: {e}")

        # Log to CSV (V3: New columns for Fix 1 & Fix 2 analysis + system clock drift + EXPERIMENT-14 drift fields)
        csv_writer.writerow([
            sample_number,
            elapsed,
            datetime.fromtimestamp(system_time).isoformat(),
            system_time,
            chronotick_time_fix1,           # V3: NEW
            chronotick_time_fix2,           # V3: NEW
            chronotick_offset_ms,
            chronotick_drift_rate,          # V3: NEW
            chronotick_drift_uncertainty,   # EXPERIMENT-14: NEW
            chronotick_drift_source,        # EXPERIMENT-14: NEW
            chronotick_prediction_time,     # V3: NEW
            chronotick_uncertainty_ms,
            chronotick_confidence,
            chronotick_source,
            time_since_ntp_s,               # V3: NEW
            system_clock_drift_rate,        # DRIFT EXP: System clock drift in s/s
            ntp_time,
            ntp_offset_ms,
            ntp_uncertainty_ms,
            ntp_mad_ms,
            ntp_n_combined,
            ntp_n_total,
            ntp_n_rejected,
            ntp_rejected_servers,
            has_ntp
        ])

        # Periodic progress update (every 60 samples = 1 minute)
        if sample_number % 60 == 0:
            print(f"[{elapsed/60:6.1f}min] Progress: {sample_number} samples, {ntp_sample_count} NTP measurements")

        # Sleep until next sample
        time.sleep(SAMPLE_INTERVAL_SECONDS)

    # Close CSV
    csv_file.close()

    print()
    print("=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)
    print(f"  Total samples: {sample_number}")
    print(f"  NTP measurements: {ntp_sample_count}")
    print(f"  CSV saved to: {CSV_PATH}")
    print()


if __name__ == '__main__':
    main()
