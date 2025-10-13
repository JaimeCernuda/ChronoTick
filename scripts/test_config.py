#!/usr/bin/env python3
"""Test that config loads correctly"""
import yaml
from pathlib import Path

config_path = Path(__file__).parent.parent / "chronotick_inference" / "config.yaml"

print(f"Loading config from: {config_path}")
print(f"File exists: {config_path.exists()}")

with open(config_path) as f:
    config = yaml.safe_load(f)

print(f"\nTop-level keys: {list(config.keys())}")
print(f"\nHas clock_measurement: {'clock_measurement' in config}")
print(f"Has prediction_scheduling: {'prediction_scheduling' in config}")

if 'clock_measurement' in config:
    print("\n✓ clock_measurement config:")
    print(f"  NTP servers: {config['clock_measurement']['ntp']['servers']}")
    print(f"  Warmup duration: {config['clock_measurement']['timing']['warm_up']['duration_seconds']}s")
else:
    print("\n✗ clock_measurement NOT FOUND")

if 'prediction_scheduling' in config:
    print("\n✓ prediction_scheduling config:")
    print(f"  CPU interval: {config['prediction_scheduling']['cpu_model']['prediction_interval']}s")
else:
    print("\n✗ prediction_scheduling NOT FOUND")
