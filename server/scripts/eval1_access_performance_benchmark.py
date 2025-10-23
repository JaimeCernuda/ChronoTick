#!/usr/bin/env python3
"""
Evaluation 1: Access Performance Benchmark

Measures latency for different time access methods:
a) System clock (time.time())
b) NTP (single server direct query)
c) ChronoTick (IPC + correction calculation, scaled to 1, 2, 4, 8 concurrent clients)

Collects 50-100 measurements per method with statistical analysis.
"""

import sys
import time
import ntplib
import numpy as np
import multiprocessing as mp
from pathlib import Path
from typing import List, Dict, Tuple
import json

# Add server/src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from chronotick.inference.engine import ChronoTickInferenceEngine
from chronotick.inference.real_data_pipeline import RealDataPipeline
from chronotick.inference.tsfm_model_wrapper import create_model_wrappers


class AccessPerformanceBenchmark:
    def __init__(self):
        self.results = {}

    def benchmark_system_clock(self, num_iterations=100):
        """Benchmark system clock access (time.time())"""
        print("\n" + "="*80)
        print("Benchmarking System Clock Access")
        print("="*80)
        print(f"Iterations: {num_iterations}")

        latencies = []

        for i in range(num_iterations):
            start = time.perf_counter_ns()
            current_time = time.time()
            end = time.perf_counter_ns()

            latency_ms = (end - start) / 1e6  # Convert ns to ms
            latencies.append(latency_ms)

            if i % 20 == 0:
                print(f"  Progress: {i}/{num_iterations}")

        self.results['system_clock'] = {
            'method': 'System Clock (time.time())',
            'latencies': latencies,
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies),
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
        }

        print(f"\n✓ System Clock Results:")
        print(f"  Mean:   {self.results['system_clock']['mean']:.6f} ms")
        print(f"  Std:    {self.results['system_clock']['std']:.6f} ms")
        print(f"  Median: {self.results['system_clock']['p50']:.6f} ms")

    def benchmark_ntp_access(self, num_iterations=50, ntp_server="time.google.com"):
        """Benchmark NTP access (single server with full round-trip)"""
        print("\n" + "="*80)
        print("Benchmarking NTP Access (Single Server)")
        print("="*80)
        print(f"Iterations: {num_iterations}")
        print(f"NTP Server: {ntp_server}")
        print("Sleep interval: 2s between queries to avoid being banned")

        ntp_client = ntplib.NTPClient()
        latencies = []
        failed = 0

        for i in range(num_iterations):
            try:
                start = time.perf_counter_ns()
                response = ntp_client.request(ntp_server, version=3, timeout=2)
                end = time.perf_counter_ns()

                latency_ms = (end - start) / 1e6  # Convert ns to ms
                latencies.append(latency_ms)

                if i % 10 == 0:
                    print(f"  Progress: {i}/{num_iterations}, latency={latency_ms:.2f}ms")

                # Sleep to avoid being banned
                time.sleep(2)

            except Exception as e:
                print(f"  ⚠️  NTP query {i} failed: {e}")
                failed += 1
                time.sleep(2)

        if not latencies:
            print("  ✗ All NTP queries failed!")
            self.results['ntp'] = {
                'method': 'NTP (single server)',
                'latencies': [],
                'error': 'All queries failed',
            }
            return

        self.results['ntp'] = {
            'method': 'NTP (single server)',
            'latencies': latencies,
            'mean': np.mean(latencies),
            'std': np.std(latencies),
            'min': np.min(latencies),
            'max': np.max(latencies),
            'p50': np.percentile(latencies, 50),
            'p95': np.percentile(latencies, 95),
            'p99': np.percentile(latencies, 99),
            'failed_count': failed,
            'success_rate': (num_iterations - failed) / num_iterations * 100,
        }

        print(f"\n✓ NTP Results:")
        print(f"  Mean:    {self.results['ntp']['mean']:.2f} ms")
        print(f"  Std:     {self.results['ntp']['std']:.2f} ms")
        print(f"  Median:  {self.results['ntp']['p50']:.2f} ms")
        print(f"  Success: {self.results['ntp']['success_rate']:.1f}%")

    def benchmark_chronotick_access(self, num_iterations=100, config_path="configs/config_stable_clock.yaml"):
        """Benchmark ChronoTick access (IPC + correction calculation)"""
        print("\n" + "="*80)
        print("Benchmarking ChronoTick Access (Single Client)")
        print("="*80)
        print(f"Iterations: {num_iterations}")
        print(f"Config: {config_path}")

        # Initialize ChronoTick
        print("Initializing ChronoTick...")
        engine = ChronoTickInferenceEngine(config_path)
        engine.initialize_models()

        pipeline = RealDataPipeline(config_path)
        cpu_wrapper, gpu_wrapper = create_model_wrappers(
            engine, pipeline.dataset_manager, pipeline.system_metrics
        )
        pipeline.initialize(cpu_model=cpu_wrapper, gpu_model=gpu_wrapper)
        print("✓ ChronoTick initialized")

        # Warmup
        print("Warmup (30 seconds)...")
        warmup_start = time.time()
        while time.time() - warmup_start < 30:
            try:
                correction = pipeline.get_real_clock_correction(time.time())
            except:
                pass
            time.sleep(1)
        print("✓ Warmup complete")

        # Benchmark
        ipc_latencies = []
        correction_latencies = []
        total_latencies = []
        failed = 0

        for i in range(num_iterations):
            try:
                system_time = time.time()

                # Measure IPC call (get correction from daemon)
                ipc_start = time.perf_counter_ns()
                correction = pipeline.get_real_clock_correction(system_time)
                ipc_end = time.perf_counter_ns()

                ipc_latency_ms = (ipc_end - ipc_start) / 1e6

                # Measure correction calculation
                correction_start = time.perf_counter_ns()
                time_delta = system_time - correction.prediction_time
                chronotick_time = system_time + correction.offset_correction + (correction.drift_rate * time_delta)
                correction_end = time.perf_counter_ns()

                correction_latency_ms = (correction_end - correction_start) / 1e6
                total_latency_ms = ipc_latency_ms + correction_latency_ms

                ipc_latencies.append(ipc_latency_ms)
                correction_latencies.append(correction_latency_ms)
                total_latencies.append(total_latency_ms)

                if i % 20 == 0:
                    print(f"  Progress: {i}/{num_iterations}, IPC={ipc_latency_ms:.6f}ms, correction={correction_latency_ms:.6f}ms")

            except Exception as e:
                print(f"  ⚠️  ChronoTick query {i} failed: {e}")
                failed += 1

        if not total_latencies:
            print("  ✗ All ChronoTick queries failed!")
            return

        self.results['chronotick_1'] = {
            'method': 'ChronoTick (1 client)',
            'num_clients': 1,
            'ipc_latencies': ipc_latencies,
            'correction_latencies': correction_latencies,
            'total_latencies': total_latencies,
            'ipc_mean': np.mean(ipc_latencies),
            'ipc_std': np.std(ipc_latencies),
            'correction_mean': np.mean(correction_latencies),
            'correction_std': np.std(correction_latencies),
            'total_mean': np.mean(total_latencies),
            'total_std': np.std(total_latencies),
            'total_min': np.min(total_latencies),
            'total_max': np.max(total_latencies),
            'total_p50': np.percentile(total_latencies, 50),
            'total_p95': np.percentile(total_latencies, 95),
            'total_p99': np.percentile(total_latencies, 99),
            'failed_count': failed,
        }

        print(f"\n✓ ChronoTick Results (1 client):")
        print(f"  IPC mean:        {self.results['chronotick_1']['ipc_mean']:.6f} ms")
        print(f"  Correction mean: {self.results['chronotick_1']['correction_mean']:.6f} ms")
        print(f"  Total mean:      {self.results['chronotick_1']['total_mean']:.6f} ms")
        print(f"  Total median:    {self.results['chronotick_1']['total_p50']:.6f} ms")

        # Cleanup
        print("Cleaning up...")
        try:
            pipeline.stop()
        except:
            pass

    def benchmark_chronotick_concurrent(self, num_clients_list=[2, 4, 8],
                                       iterations_per_client=100,
                                       config_path="configs/config_stable_clock.yaml"):
        """Benchmark ChronoTick with concurrent clients"""
        print("\n" + "="*80)
        print("Benchmarking ChronoTick Concurrent Access")
        print("="*80)

        for num_clients in num_clients_list:
            print(f"\n--- Testing with {num_clients} concurrent clients ---")
            print(f"Iterations per client: {iterations_per_client}")

            # Initialize ChronoTick (shared across processes)
            print("Initializing ChronoTick...")
            engine = ChronoTickInferenceEngine(config_path)
            engine.initialize_models()

            pipeline = RealDataPipeline(config_path)
            cpu_wrapper, gpu_wrapper = create_model_wrappers(
                engine, pipeline.dataset_manager, pipeline.system_metrics
            )
            pipeline.initialize(cpu_model=cpu_wrapper, gpu_model=gpu_wrapper)
            print("✓ ChronoTick initialized")

            # Warmup
            print("Warmup (15 seconds)...")
            warmup_start = time.time()
            while time.time() - warmup_start < 15:
                try:
                    correction = pipeline.get_real_clock_correction(time.time())
                except:
                    pass
                time.sleep(1)
            print("✓ Warmup complete")

            # Run concurrent clients
            result_queue = mp.Queue()
            processes = []

            print(f"Launching {num_clients} concurrent processes...")
            for i in range(num_clients):
                p = mp.Process(target=self._worker_process,
                             args=(i, iterations_per_client, config_path, result_queue))
                processes.append(p)
                p.start()

            # Wait for all processes to complete
            for p in processes:
                p.join()

            # Collect results
            all_ipc_latencies = []
            all_correction_latencies = []
            all_total_latencies = []

            while not result_queue.empty():
                result = result_queue.get()
                all_ipc_latencies.extend(result['ipc_latencies'])
                all_correction_latencies.extend(result['correction_latencies'])
                all_total_latencies.extend(result['total_latencies'])

            if not all_total_latencies:
                print(f"  ✗ No successful measurements for {num_clients} clients!")
                continue

            key = f'chronotick_{num_clients}'
            self.results[key] = {
                'method': f'ChronoTick ({num_clients} clients)',
                'num_clients': num_clients,
                'ipc_latencies': all_ipc_latencies,
                'correction_latencies': all_correction_latencies,
                'total_latencies': all_total_latencies,
                'ipc_mean': np.mean(all_ipc_latencies),
                'ipc_std': np.std(all_ipc_latencies),
                'correction_mean': np.mean(all_correction_latencies),
                'correction_std': np.std(all_correction_latencies),
                'total_mean': np.mean(all_total_latencies),
                'total_std': np.std(all_total_latencies),
                'total_min': np.min(all_total_latencies),
                'total_max': np.max(all_total_latencies),
                'total_p50': np.percentile(all_total_latencies, 50),
                'total_p95': np.percentile(all_total_latencies, 95),
                'total_p99': np.percentile(all_total_latencies, 99),
            }

            print(f"\n✓ ChronoTick Results ({num_clients} clients):")
            print(f"  IPC mean:        {self.results[key]['ipc_mean']:.6f} ms")
            print(f"  Correction mean: {self.results[key]['correction_mean']:.6f} ms")
            print(f"  Total mean:      {self.results[key]['total_mean']:.6f} ms")
            print(f"  Total median:    {self.results[key]['total_p50']:.6f} ms")
            print(f"  Total samples:   {len(all_total_latencies)}")

            # Cleanup
            print("Cleaning up...")
            try:
                pipeline.stop()
            except:
                pass
            time.sleep(2)

    @staticmethod
    def _worker_process(worker_id, num_iterations, config_path, result_queue):
        """Worker process for concurrent benchmarking"""
        try:
            # Each worker initializes its own pipeline connection
            engine = ChronoTickInferenceEngine(config_path)
            engine.initialize_models()

            pipeline = RealDataPipeline(config_path)
            cpu_wrapper, gpu_wrapper = create_model_wrappers(
                engine, pipeline.dataset_manager, pipeline.system_metrics
            )
            pipeline.initialize(cpu_model=cpu_wrapper, gpu_model=gpu_wrapper)

            ipc_latencies = []
            correction_latencies = []
            total_latencies = []

            for i in range(num_iterations):
                try:
                    system_time = time.time()

                    # Measure IPC
                    ipc_start = time.perf_counter_ns()
                    correction = pipeline.get_real_clock_correction(system_time)
                    ipc_end = time.perf_counter_ns()

                    ipc_latency_ms = (ipc_end - ipc_start) / 1e6

                    # Measure correction
                    correction_start = time.perf_counter_ns()
                    time_delta = system_time - correction.prediction_time
                    chronotick_time = system_time + correction.offset_correction + (correction.drift_rate * time_delta)
                    correction_end = time.perf_counter_ns()

                    correction_latency_ms = (correction_end - correction_start) / 1e6
                    total_latency_ms = ipc_latency_ms + correction_latency_ms

                    ipc_latencies.append(ipc_latency_ms)
                    correction_latencies.append(correction_latency_ms)
                    total_latencies.append(total_latency_ms)

                except Exception as e:
                    pass

            result_queue.put({
                'worker_id': worker_id,
                'ipc_latencies': ipc_latencies,
                'correction_latencies': correction_latencies,
                'total_latencies': total_latencies,
            })

        except Exception as e:
            print(f"Worker {worker_id} failed: {e}")

    def save_results(self, output_path="eval1_access_performance_results.json"):
        """Save results to JSON"""
        print(f"\nSaving results to: {output_path}")

        # Convert numpy types to native Python types for JSON serialization
        json_results = {}
        for key, value in self.results.items():
            json_results[key] = {}
            for k, v in value.items():
                if isinstance(v, (list, np.ndarray)):
                    json_results[key][k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
                elif isinstance(v, (np.floating, np.integer)):
                    json_results[key][k] = float(v)
                else:
                    json_results[key][k] = v

        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)

        print(f"✓ Results saved!")

    def print_summary(self):
        """Print summary of all results"""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)

        print("\nLatency Comparison (mean ± std):")
        print("-" * 80)

        if 'system_clock' in self.results:
            r = self.results['system_clock']
            print(f"System Clock:       {r['mean']:>10.6f} ± {r['std']:>8.6f} ms")

        if 'ntp' in self.results and 'error' not in self.results['ntp']:
            r = self.results['ntp']
            print(f"NTP (single):       {r['mean']:>10.2f} ± {r['std']:>8.2f} ms")

        for key in sorted(self.results.keys()):
            if key.startswith('chronotick_'):
                r = self.results[key]
                clients = r['num_clients']
                print(f"ChronoTick ({clients} clients): {r['total_mean']:>10.6f} ± {r['total_std']:>8.6f} ms")
                print(f"  └─ IPC:          {r['ipc_mean']:>10.6f} ± {r['ipc_std']:>8.6f} ms")
                print(f"  └─ Correction:   {r['correction_mean']:>10.6f} ± {r['correction_std']:>8.6f} ms")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Eval 1: Access Performance Benchmark')
    parser.add_argument('--config', default='configs/config_stable_clock.yaml',
                       help='ChronoTick config file')
    parser.add_argument('--ntp-server', default='time.google.com',
                       help='NTP server for benchmarking')
    parser.add_argument('--system-iterations', type=int, default=100,
                       help='Iterations for system clock benchmark')
    parser.add_argument('--ntp-iterations', type=int, default=50,
                       help='Iterations for NTP benchmark')
    parser.add_argument('--chronotick-iterations', type=int, default=100,
                       help='Iterations for ChronoTick benchmark')
    parser.add_argument('--concurrent-clients', type=int, nargs='+', default=[2, 4, 8],
                       help='Number of concurrent clients to test')
    parser.add_argument('--output', default='eval1_access_performance_results.json',
                       help='Output JSON file')
    parser.add_argument('--skip-ntp', action='store_true',
                       help='Skip NTP benchmark (slow)')
    parser.add_argument('--skip-concurrent', action='store_true',
                       help='Skip concurrent benchmarks')

    args = parser.parse_args()

    benchmark = AccessPerformanceBenchmark()

    try:
        # Benchmark 1: System Clock
        benchmark.benchmark_system_clock(num_iterations=args.system_iterations)

        # Benchmark 2: NTP (optional, slow)
        if not args.skip_ntp:
            benchmark.benchmark_ntp_access(num_iterations=args.ntp_iterations,
                                          ntp_server=args.ntp_server)

        # Benchmark 3: ChronoTick single client
        benchmark.benchmark_chronotick_access(num_iterations=args.chronotick_iterations,
                                             config_path=args.config)

        # Benchmark 4: ChronoTick concurrent clients (optional)
        if not args.skip_concurrent:
            benchmark.benchmark_chronotick_concurrent(
                num_clients_list=args.concurrent_clients,
                iterations_per_client=args.chronotick_iterations,
                config_path=args.config
            )

        # Save and summarize
        benchmark.save_results(output_path=args.output)
        benchmark.print_summary()

        print("\n" + "="*80)
        print("✓ BENCHMARK COMPLETE!")
        print("="*80)

    except KeyboardInterrupt:
        print("\n\n⚠️  Benchmark interrupted by user!")
        benchmark.save_results(output_path=args.output)
        benchmark.print_summary()
    except Exception as e:
        print(f"\n\n✗ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
