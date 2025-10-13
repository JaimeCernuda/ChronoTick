#!/usr/bin/env python3
"""
Basic test for ChronoTick evaluation collection system
"""

import os
import sys
import time
import socket
import struct
import json
from pathlib import Path


def test_ntp_client():
    """Test basic NTP functionality"""
    print("Testing NTP client...")

    servers = ['pool.ntp.org', 'time.google.com']

    for server in servers:
        try:
            t1 = time.time()

            # Create NTP request
            packet = [0] * 12
            packet[0] = 0x1B000000  # NTP v3 client
            ntp_packet = struct.pack("!12I", *packet)

            # Send request
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.settimeout(2.0)
            sock.sendto(ntp_packet, (server, 123))
            response, _ = sock.recvfrom(1024)
            t4 = time.time()
            sock.close()

            # Parse response
            unpacked = struct.unpack("!12I", response)
            stratum = (unpacked[0] >> 16) & 0xFF

            # Extract timestamps
            NTP_EPOCH_OFFSET = 2208988800
            t2 = unpacked[8] - NTP_EPOCH_OFFSET + (unpacked[9] / 2**32)
            t3 = unpacked[10] - NTP_EPOCH_OFFSET + (unpacked[11] / 2**32)

            # Calculate offset and delay
            offset = ((t2 - t1) + (t3 - t4)) / 2.0
            delay = (t4 - t1) - (t3 - t2)

            print(f"  {server}: offset={offset*1e6:.1f}μs, delay={delay*1000:.1f}ms, stratum={stratum}")

        except Exception as e:
            print(f"  {server}: FAILED - {e}")


def test_system_metrics():
    """Test system metrics collection"""
    print("Testing system metrics...")

    try:
        # CPU load
        with open('/proc/loadavg') as f:
            load = float(f.read().split()[0])
        print(f"  CPU load: {load}")

        # Memory usage
        with open('/proc/meminfo') as f:
            meminfo = f.read()
            mem_total = int([line for line in meminfo.split('\n') if 'MemTotal:' in line][0].split()[1])
            mem_free = int([line for line in meminfo.split('\n') if 'MemAvailable:' in line][0].split()[1])
            mem_usage = (mem_total - mem_free) / mem_total * 100
        print(f"  Memory usage: {mem_usage:.1f}%")

        # CPU temperature (if available)
        thermal_zones = list(Path('/sys/class/thermal').glob('thermal_zone*'))
        if thermal_zones:
            with open(thermal_zones[0] / 'temp') as f:
                temp = int(f.read().strip()) / 1000.0
            print(f"  CPU temperature: {temp:.1f}°C")
        else:
            print("  CPU temperature: Not available")

    except Exception as e:
        print(f"  System metrics failed: {e}")


def test_data_collection():
    """Test basic data collection and storage"""
    print("Testing data collection...")

    output_dir = Path('/tmp/chronotick_test')
    output_dir.mkdir(exist_ok=True)

    try:
        # Collect 5 measurements
        measurements = []

        for i in range(5):
            print(f"  Collecting measurement {i+1}/5...")

            measurement = {
                'timestamp': time.time(),
                'measurement_id': i,
                'test_data': f'test_{i}'
            }
            measurements.append(measurement)
            time.sleep(1)

        # Save to file
        output_file = output_dir / 'test_measurements.json'
        with open(output_file, 'w') as f:
            json.dump(measurements, f, indent=2)

        print(f"  Saved {len(measurements)} measurements to {output_file}")

        # Verify file
        with open(output_file) as f:
            loaded = json.load(f)

        print(f"  Verified: loaded {len(loaded)} measurements")

    except Exception as e:
        print(f"  Data collection failed: {e}")


def main():
    print("ChronoTick Evaluation 1 - Basic Collection Test")
    print("=" * 50)

    test_ntp_client()
    print()

    test_system_metrics()
    print()

    test_data_collection()
    print()

    print("Basic tests complete!")
    print("If all tests passed, the collection system should work correctly.")


if __name__ == "__main__":
    main()