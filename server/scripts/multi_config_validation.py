#!/usr/bin/env python3
"""
Sequential Multi-Configuration Validation Testing

Runs ChronoTick with different configurations sequentially, each for a specified
duration (default 1 hour), collecting separate datasets for comparison.

This allows us to showcase the value of different design decisions by comparing
accuracy across configurations.

Usage:
    python scripts/multi_config_validation.py --duration 3600 --machine homelab
    python scripts/multi_config_validation.py --duration 1800 --dry-run
"""

import subprocess
import time
import os
import sys
import signal
import argparse
from pathlib import Path
from datetime import datetime
import json

# Add server/src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Configuration test matrix
CONFIGURATIONS = [
    {
        "name": "01_single_short_only",
        "config_file": "configs/multitest/01_single_short_only.yaml",
        "description": "Phase 1: Single short-term model only, no correction"
    },
    {
        "name": "02_single_long_only",
        "config_file": "configs/multitest/02_single_long_only.yaml",
        "description": "Phase 1: Single long-term model only, no correction"
    },
    {
        "name": "03_dual_baseline",
        "config_file": "configs/multitest/03_dual_baseline.yaml",
        "description": "Phase 1: Dual model baseline, no correction"
    },
    {
        "name": "04_dual_linear",
        "config_file": "configs/multitest/04_dual_linear.yaml",
        "description": "Phase 2: Dual model + linear correction"
    },
    {
        "name": "05_dual_advanced",
        "config_file": "configs/multitest/05_dual_advanced.yaml",
        "description": "Phase 2: Dual model + advanced correction"
    },
    {
        "name": "06_production_baseline",
        "config_file": "configs/multitest/06_production_baseline.yaml",
        "description": "Phase 2: PRODUCTION - Dual + backtracking + windowed + smoothing"
    },
    {
        "name": "07_consecutive_drift",
        "config_file": "configs/multitest/07_consecutive_drift.yaml",
        "description": "Phase 3: Backtracking + consecutive drift (vs windowed)"
    },
    {
        "name": "08_no_smoothing",
        "config_file": "configs/multitest/08_no_smoothing.yaml",
        "description": "Phase 3: Backtracking + no baseline smoothing"
    },
]


class ConfigurationTest:
    def __init__(self, output_dir, machine, test_duration):
        self.output_dir = Path(output_dir)
        self.machine = machine
        self.test_duration = test_duration
        self.daemon_proc = None
        self.validation_proc = None

    def run_test(self, config):
        """Run a single configuration test"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        config_name = config["name"]

        print(f"\n{'='*80}")
        print(f"Starting test: {config_name}")
        print(f"Description: {config['description']}")
        print(f"Config file: {config['config_file']}")
        print(f"Duration: {self.test_duration}s ({self.test_duration/60:.1f} minutes)")
        print(f"Machine: {self.machine}")
        print(f"Timestamp: {timestamp}")
        print(f"{'='*80}\n")

        # Create output directory for this configuration
        output_subdir = self.output_dir / f"{timestamp}_{config_name}"
        output_subdir.mkdir(parents=True, exist_ok=True)

        try:
            # Start ChronoTick daemon
            self._start_daemon(config, output_subdir)

            # Wait for daemon to initialize
            print("Waiting for daemon to initialize...")
            time.sleep(15)

            # Check if daemon is running
            if self.daemon_proc and self.daemon_proc.poll() is not None:
                print(f"ERROR: Daemon failed to start!")
                return False

            # Start client validation
            self._start_validation(config, output_subdir)

            # Wait for test to complete
            print(f"Test running... ({self.test_duration}s remaining)")
            start_time = time.time()

            while time.time() - start_time < self.test_duration:
                # Check if validation process is still running
                if self.validation_proc.poll() is not None:
                    print("WARNING: Validation process terminated early!")
                    break

                # Progress update every 5 minutes
                elapsed = time.time() - start_time
                if int(elapsed) % 300 == 0 and elapsed > 0:
                    remaining = self.test_duration - elapsed
                    print(f"  Progress: {elapsed/60:.1f}m elapsed, {remaining/60:.1f}m remaining")

                time.sleep(10)

            print(f"✓ Test completed successfully!")

        except KeyboardInterrupt:
            print(f"\n⚠️  Test interrupted by user!")
            raise

        except Exception as e:
            print(f"ERROR: Test failed with exception: {e}")
            return False

        finally:
            # Always cleanup
            self._cleanup()

        # Save metadata
        self._save_metadata(config, output_subdir, timestamp)

        print(f"\n✓ Results saved to: {output_subdir}\n")
        return True

    def _start_daemon(self, config, output_subdir):
        """Start ChronoTick daemon"""
        daemon_log = output_subdir / "daemon.log"

        # Kill any existing daemon
        subprocess.run(
            [sys.executable, "-m", "chronotick.inference.daemon", "stop"],
            cwd=Path(__file__).parent.parent,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent / "src")},
            capture_output=True
        )
        time.sleep(2)

        daemon_cmd = [
            sys.executable, "-m", "chronotick.inference.daemon",
            "start",
            "--config", config["config_file"]
        ]

        print(f"Starting ChronoTick daemon: {' '.join(daemon_cmd)}")

        self.daemon_proc = subprocess.Popen(
            daemon_cmd,
            stdout=open(daemon_log, 'w'),
            stderr=subprocess.STDOUT,
            cwd=Path(__file__).parent.parent,
            env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent / "src")}
        )

    def _start_validation(self, config, output_subdir):
        """Start client validation"""
        validation_log = output_subdir / "validation.log"
        validation_csv = output_subdir / "data.csv"

        validation_cmd = [
            sys.executable, "scripts/client_driven_validation.py"
        ]

        print(f"Starting client validation...")

        self.validation_proc = subprocess.Popen(
            validation_cmd,
            stdout=open(validation_log, 'w'),
            stderr=subprocess.STDOUT,
            cwd=Path(__file__).parent.parent.parent
        )

    def _cleanup(self):
        """Stop daemon and validation processes"""
        print("Cleaning up processes...")

        # Terminate validation process
        if self.validation_proc:
            try:
                self.validation_proc.terminate()
                self.validation_proc.wait(timeout=10)
            except:
                self.validation_proc.kill()

        # Stop daemon
        if self.daemon_proc:
            try:
                subprocess.run(
                    [sys.executable, "-m", "chronotick.inference.daemon", "stop"],
                    cwd=Path(__file__).parent.parent,
                    env={**os.environ, "PYTHONPATH": str(Path(__file__).parent.parent / "src")},
                    timeout=15,
                    capture_output=True
                )
            except:
                pass

            try:
                self.daemon_proc.terminate()
                self.daemon_proc.wait(timeout=10)
            except:
                self.daemon_proc.kill()

        # Additional cleanup - kill any lingering processes
        subprocess.run(["pkill", "-f", "chronotick.inference.daemon"], capture_output=True)
        time.sleep(2)

    def _save_metadata(self, config, output_subdir, timestamp):
        """Save test metadata"""
        metadata = {
            "configuration": config["name"],
            "description": config["description"],
            "config_file": config["config_file"],
            "machine": self.machine,
            "start_time": timestamp,
            "duration_seconds": self.test_duration,
        }

        # Save as JSON
        with open(output_subdir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        # Save as text for easy reading
        with open(output_subdir / "metadata.txt", 'w') as f:
            f.write(f"Configuration: {config['name']}\n")
            f.write(f"Description: {config['description']}\n")
            f.write(f"Config file: {config['config_file']}\n")
            f.write(f"Machine: {self.machine}\n")
            f.write(f"Start time: {timestamp}\n")
            f.write(f"Duration: {self.test_duration}s ({self.test_duration/60:.1f} minutes)\n")


def main():
    parser = argparse.ArgumentParser(
        description="Sequential multi-configuration validation testing"
    )
    parser.add_argument(
        "--duration", "-d",
        type=int,
        default=3600,
        help="Duration per test in seconds (default: 3600 = 1 hour)"
    )
    parser.add_argument(
        "--output", "-o",
        default="results/multi_config_test",
        help="Output directory for results"
    )
    parser.add_argument(
        "--machine", "-m",
        default="unknown",
        help="Machine name for metadata"
    )
    parser.add_argument(
        "--start-from", "-s",
        type=int,
        default=0,
        help="Start from configuration N (0-indexed, for resuming)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print test plan without running"
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip confirmation prompt and start immediately"
    )

    args = parser.parse_args()

    # Print test plan
    print("="*80)
    print("ChronoTick Multi-Configuration Sequential Testing")
    print("="*80)
    print(f"Total configurations: {len(CONFIGURATIONS)}")
    print(f"Duration per test: {args.duration}s ({args.duration/60:.1f} minutes)")
    print(f"Total estimated time: {len(CONFIGURATIONS) * args.duration / 3600:.1f} hours")
    print(f"Output directory: {args.output}")
    print(f"Machine: {args.machine}")
    print(f"Start from: Configuration #{args.start_from}")
    print("="*80)
    print("\nTest Configurations:")
    for i, config in enumerate(CONFIGURATIONS):
        marker = "→ " if i >= args.start_from else "✓ "
        print(f"  {marker}{i+1}. {config['name']}")
        print(f"     {config['description']}")
    print("="*80)

    if args.dry_run:
        print("\nDry run complete. Use without --dry-run to execute.")
        return

    # Confirm start (skip if --yes flag is set)
    if not args.yes:
        response = input("\nStart testing? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run tests
    tester = ConfigurationTest(output_dir, args.machine, args.duration)

    failed_configs = []

    for i, config in enumerate(CONFIGURATIONS[args.start_from:], start=args.start_from):
        print(f"\n[{i+1}/{len(CONFIGURATIONS)}] Testing configuration: {config['name']}")

        try:
            success = tester.run_test(config)

            if not success:
                print(f"⚠️  WARNING: Test failed for {config['name']}")
                failed_configs.append(config['name'])

                response = input("Continue with next configuration? (y/n): ")
                if response.lower() != 'y':
                    break

            # Brief pause between tests
            if i < len(CONFIGURATIONS) - 1:
                print(f"\nPausing 30s before next test...")
                time.sleep(30)

        except KeyboardInterrupt:
            print("\n\n⚠️  Testing interrupted by user.")
            break

    # Final summary
    print("\n" + "="*80)
    print("Testing Complete!")
    print("="*80)
    print(f"Completed: {i+1}/{len(CONFIGURATIONS)} configurations")
    if failed_configs:
        print(f"Failed: {len(failed_configs)} configurations")
        for name in failed_configs:
            print(f"  - {name}")
    print(f"\nResults available in: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTesting interrupted by user.")
        sys.exit(1)
