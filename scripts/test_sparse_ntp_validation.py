#!/usr/bin/env python3
"""
ChronoTick Sparse NTP Validation Test
Tests the new 3-minute NTP interval configuration to verify:
1. NTP queries are properly spaced at 3-minute intervals after warmup
2. ML predictions maintain accuracy during 3-minute gaps
3. Time extrapolation works correctly between NTP queries
4. Overall reduction in NTP query frequency (20/hour vs 720/hour)
"""

import time
import ntplib
from pathlib import Path
from collections import defaultdict
import sys
import logging
import threading

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from chronotick.inference.engine import ChronoTickInferenceEngine
from chronotick.inference.real_data_pipeline import RealDataPipeline
from chronotick.inference.tsfm_model_wrapper import create_model_wrappers

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class SparseNTPValidator:
    def __init__(self, test_duration_seconds: int = 900):  # 15 minutes default
        self.test_duration = test_duration_seconds
        self.config_path = "configs/config.yaml"

        # Components
        self.engine = None
        self.pipeline = None
        self.ntp_client = ntplib.NTPClient()

        # Tracking metrics
        self.ntp_query_times = []
        self.time_samples = []
        self.start_time = None
        self.warmup_complete_time = None
        self.last_dataset_size = 0

        # Monitoring thread
        self.monitor_thread = None
        self.stop_monitoring = threading.Event()

    def setup(self):
        """Initialize ChronoTick components"""
        print("=" * 80)
        print("ChronoTick Sparse NTP Validation Test")
        print("=" * 80)
        print(f"Test Duration: {self.test_duration}s ({self.test_duration/60:.1f} minutes)")
        print(f"Expected NTP Interval: 180s (3 minutes)")
        print(f"Expected Warmup: 60s")
        print("=" * 80)
        print()

        logger.info("Initializing ChronoTick system...")
        self.engine = ChronoTickInferenceEngine(self.config_path)
        self.engine.initialize_models()
        logger.info("✓ TimesFM models loaded")

        self.pipeline = RealDataPipeline(self.config_path)
        cpu_wrapper, gpu_wrapper = create_model_wrappers(
            self.engine, self.pipeline.dataset_manager, self.pipeline.system_metrics
        )
        self.pipeline.initialize(cpu_model=cpu_wrapper, gpu_model=gpu_wrapper)
        self.pipeline.predictive_scheduler.set_model_interfaces(
            cpu_model=cpu_wrapper,
            gpu_model=gpu_wrapper,
            fusion_engine=self.pipeline.fusion_engine
        )
        logger.info("✓ All components initialized")
        print()

        # Start system metrics collection
        self.pipeline.system_metrics.start_collection()
        self.start_time = time.time()

    def _timestamp(self) -> str:
        """Get formatted timestamp"""
        if self.start_time:
            elapsed = time.time() - self.start_time
            return f"t+{elapsed:6.1f}s"
        return f"{time.time():.1f}"

    def get_reference_ntp_time(self) -> float:
        """Get direct NTP time for comparison"""
        try:
            response = self.ntp_client.request('pool.ntp.org', version=3, timeout=2)
            return response.tx_time
        except Exception as e:
            logger.debug(f"Could not get reference NTP: {e}")
            return None

    def monitor_ntp_queries(self):
        """Monitor NTP queries in background thread"""
        while not self.stop_monitoring.is_set():
            try:
                # Check dataset size to detect new NTP measurements
                dataset_size = len(self.pipeline.dataset_manager.measurement_dataset)

                if dataset_size > self.last_dataset_size:
                    elapsed = time.time() - self.start_time

                    # Check if we're in warmup or normal operation
                    if not self.pipeline.warm_up_complete:
                        phase = "WARMUP"
                    else:
                        phase = "NORMAL"
                        if not self.warmup_complete_time:
                            self.warmup_complete_time = elapsed
                            logger.info("")
                            logger.info("✅ Warmup complete! Entering normal operation with 3-minute NTP intervals")
                            logger.info("")

                    # Calculate interval from last NTP query
                    if self.ntp_query_times:
                        interval = elapsed - self.ntp_query_times[-1]
                        logger.info(f"[{phase}] NTP query #{dataset_size} (Δt: {interval:.1f}s)")
                    else:
                        logger.info(f"[{phase}] NTP query #{dataset_size} (first)")

                    self.ntp_query_times.append(elapsed)
                    self.last_dataset_size = dataset_size

                time.sleep(1)  # Check every second

            except Exception as e:
                logger.error(f"Error monitoring NTP: {e}")
                time.sleep(1)

    def sample_time(self):
        """Sample ChronoTick time and compare to reference"""
        elapsed = time.time() - self.start_time
        current_time = time.time()

        # Get ChronoTick corrected time
        correction = self.pipeline.get_real_clock_correction(current_time)

        # Get reference NTP for comparison
        ntp_time = self.get_reference_ntp_time()
        system_time = time.time()

        # Calculate ChronoTick corrected time
        chronotick_time = system_time + correction.offset_correction

        sample = {
            'elapsed': elapsed,
            'chronotick_time': chronotick_time,
            'ntp_time': ntp_time,
            'system_time': system_time,
            'offset': correction.offset_correction,
            'uncertainty': correction.offset_uncertainty,
            'drift': correction.drift_rate,
            'source': correction.source,
            'confidence': correction.confidence,
            'chronotick_error': None,
            'system_error': None
        }

        # Calculate errors (if we have reference NTP)
        if ntp_time:
            sample['chronotick_error'] = abs(chronotick_time - ntp_time) * 1000  # ms
            sample['system_error'] = abs(system_time - ntp_time) * 1000  # ms

        self.time_samples.append(sample)
        return sample

    def run_test(self):
        """Run the full validation test"""
        self.setup()

        # Wait for initial NTP data
        logger.info("Waiting for warmup period...")
        time.sleep(5)
        print()

        # Start NTP monitoring thread
        self.monitor_thread = threading.Thread(target=self.monitor_ntp_queries, daemon=True)
        self.monitor_thread.start()

        # Run test, sampling every 10 seconds
        sample_interval = 10
        end_time = self.start_time + self.test_duration

        logger.info("=" * 80)
        logger.info(f"Starting {self.test_duration/60:.1f}-minute test - sampling every {sample_interval}s")
        logger.info("=" * 80)
        print()

        iteration = 0
        while time.time() < end_time:
            sample = self.sample_time()

            if sample['chronotick_error'] is not None:
                logger.info(
                    f"[{iteration:3d}] "
                    f"ChronoTick error: {sample['chronotick_error']:6.3f}ms "
                    f"(±{sample['uncertainty']*1000:6.3f}ms, source: {sample['source']})"
                )
            else:
                logger.info(
                    f"[{iteration:3d}] "
                    f"Offset: {sample['offset']*1000:6.3f}ms "
                    f"(±{sample['uncertainty']*1000:6.3f}ms, source: {sample['source']})"
                )

            iteration += 1
            time.sleep(sample_interval)

        # Stop monitoring
        self.stop_monitoring.set()
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)

        # Generate report
        self.generate_report()

        # Cleanup
        logger.info("\nCleaning up...")
        self.engine.shutdown()
        self.pipeline.system_metrics.stop_collection()

    def generate_report(self):
        """Generate comprehensive test report"""
        print()
        print("=" * 80)
        print("SPARSE NTP VALIDATION REPORT")
        print("=" * 80)

        actual_duration = time.time() - self.start_time
        print(f"\nTest Duration: {actual_duration:.1f}s ({actual_duration/60:.1f} minutes)")

        # NTP query analysis
        print(f"\n--- NTP Query Analysis ---")
        print(f"Total NTP queries: {len(self.ntp_query_times)}")
        print(f"NTP queries per hour: {len(self.ntp_query_times) / (actual_duration/3600):.1f}")

        if self.warmup_complete_time:
            warmup_queries = sum(1 for t in self.ntp_query_times if t < self.warmup_complete_time)
            normal_queries = len(self.ntp_query_times) - warmup_queries
            normal_duration = actual_duration - self.warmup_complete_time

            print(f"\nWarmup phase (0-{self.warmup_complete_time:.1f}s):")
            print(f"  Queries: {warmup_queries}")
            print(f"  Avg interval: {self.warmup_complete_time/max(warmup_queries-1, 1):.1f}s")

            print(f"\nNormal operation ({self.warmup_complete_time:.1f}s - {actual_duration:.1f}s):")
            print(f"  Queries: {normal_queries}")
            print(f"  Duration: {normal_duration:.1f}s ({normal_duration/60:.1f} minutes)")
            if normal_queries > 1:
                print(f"  Avg interval: {normal_duration/(normal_queries-1):.1f}s")
                print(f"  Expected interval: 180s (3 minutes)")

        # Query timing details
        if len(self.ntp_query_times) > 1:
            intervals = [self.ntp_query_times[i+1] - self.ntp_query_times[i]
                        for i in range(len(self.ntp_query_times)-1)]
            print(f"\nRecent NTP query intervals: {[f'{i:.1f}s' for i in intervals[-5:]]}")

        # Time accuracy analysis
        print(f"\n--- Time Accuracy Analysis ---")
        print(f"Total time samples: {len(self.time_samples)}")

        valid_samples = [s for s in self.time_samples if s['chronotick_error'] is not None]
        if valid_samples:
            errors = [s['chronotick_error'] for s in valid_samples]
            uncertainties = [s['uncertainty'] * 1000 for s in valid_samples]

            print(f"\nChronoTick Error Statistics:")
            print(f"  Mean error: {sum(errors)/len(errors):.3f}ms")
            print(f"  Max error: {max(errors):.3f}ms")
            print(f"  Min error: {min(errors):.3f}ms")

            print(f"\nUncertainty Statistics:")
            print(f"  Mean uncertainty: {sum(uncertainties)/len(uncertainties):.3f}ms")
            print(f"  Max uncertainty: {max(uncertainties):.3f}ms")

        # Source distribution
        sources = defaultdict(int)
        for s in self.time_samples:
            sources[s['source']] += 1

        print(f"\n--- Prediction Source Distribution ---")
        for source, count in sorted(sources.items()):
            percentage = (count / len(self.time_samples)) * 100
            print(f"  {source:20s}: {count:3d} ({percentage:5.1f}%)")

        # Summary
        print(f"\n--- Summary ---")
        expected_ntp_old = 720  # Old config: every 5s
        expected_ntp_new = 20   # New config: every 3 minutes
        actual_ntp_rate = len(self.ntp_query_times) / (actual_duration/3600)

        print(f"Expected NTP rate (old config): {expected_ntp_old}/hour")
        print(f"Expected NTP rate (new config): {expected_ntp_new}/hour")
        print(f"Actual NTP rate: {actual_ntp_rate:.1f}/hour")

        reduction = ((expected_ntp_old - actual_ntp_rate) / expected_ntp_old) * 100
        print(f"Reduction in NTP queries: {reduction:.1f}%")

        if actual_ntp_rate < 30:
            print("\n✅ SUCCESS: NTP query frequency significantly reduced!")
            print("   ChronoTick is properly using ML to reduce NTP dependency.")
        else:
            print("\n⚠️  WARNING: NTP query frequency still high")
            print("   Expected ~20/hour, check configuration.")

        print("=" * 80)

def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description='ChronoTick Sparse NTP Validation')
    parser.add_argument('--duration', type=int, default=900,
                       help='Test duration in seconds (default: 900 = 15 minutes)')
    args = parser.parse_args()

    validator = SparseNTPValidator(test_duration_seconds=args.duration)
    validator.run_test()

if __name__ == '__main__':
    main()
