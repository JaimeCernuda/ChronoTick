#!/usr/bin/env python3
"""
Simple ChronoTick Client Example

This example demonstrates the basic usage of ChronoTick's high-level client API.
Perfect for getting started or quick integration.

Run:
    python examples/simple_client_example.py

Requirements:
    - ChronoTick daemon must be running: chronotick-daemon
"""

from chronotick_shm import ChronoTickClient, get_current_time


def main():
    print("=" * 70)
    print("ChronoTick Simple Client Example")
    print("=" * 70)

    # ========================================================================
    # Example 1: Check if daemon is running
    # ========================================================================
    print("\n1. Checking ChronoTick daemon status...")
    print("-" * 70)

    client = ChronoTickClient()

    if not client.is_daemon_ready():
        print("❌ ChronoTick daemon is not running or still warming up.")
        print("\nStart the daemon with:")
        print("  chronotick-daemon")
        print("\nOr with custom config:")
        print("  chronotick-daemon --config /path/to/config.yaml")
        return

    print("✅ ChronoTick daemon is running and ready!")

    # ========================================================================
    # Example 2: Get current corrected time
    # ========================================================================
    print("\n2. Getting current corrected time...")
    print("-" * 70)

    time_info = client.get_time()

    print(f"Corrected Time:    {time_info.corrected_timestamp:.6f}")
    print(f"System Time:       {time_info.system_timestamp:.6f}")
    print(f"Time Difference:   {(time_info.corrected_timestamp - time_info.system_timestamp) * 1000:+.3f}ms")
    print(f"Offset Correction: {time_info.offset_correction * 1000:+.3f}ms")
    print(f"Drift Rate:        {time_info.drift_rate * 1e6:+.3f}μs/s")
    print(f"Uncertainty:       ±{time_info.uncertainty_seconds * 1000:.3f}ms")
    print(f"Confidence:        {time_info.confidence:.1%}")
    print(f"Data Source:       {time_info.source}")

    # ========================================================================
    # Example 3: Project uncertainty into the future
    # ========================================================================
    print("\n3. Projecting uncertainty into the future...")
    print("-" * 70)

    # What will uncertainty be in 5 minutes?
    future_time = client.get_future_time(300)  # 300 seconds = 5 minutes

    print(f"Current uncertainty:  ±{time_info.uncertainty_seconds * 1000:.3f}ms")
    print(f"Future uncertainty:   ±{future_time.uncertainty_seconds * 1000:.3f}ms (in 5 minutes)")

    uncertainty_increase = (future_time.uncertainty_seconds - time_info.uncertainty_seconds) * 1000
    print(f"Uncertainty increase: +{uncertainty_increase:.3f}ms over 5 minutes")

    # ========================================================================
    # Example 4: Get daemon information
    # ========================================================================
    print("\n4. Getting daemon information...")
    print("-" * 70)

    daemon_info = client.get_daemon_info()

    print(f"Daemon Status:      {daemon_info['status']}")
    print(f"Uptime:             {daemon_info['daemon_uptime']:.1f} seconds")
    print(f"NTP Measurements:   {daemon_info['measurement_count']}")
    print(f"Total Corrections:  {daemon_info['total_corrections']}")
    print(f"Last NTP Sync:      {daemon_info['seconds_since_ntp']:.1f}s ago")
    print(f"NTP Ready:          {'✓' if daemon_info['ntp_ready'] else '✗'}")
    print(f"Models Ready:       {'✓' if daemon_info['models_ready'] else '✗'}")
    print(f"Avg Latency:        {daemon_info['average_latency_ms']:.3f}ms")

    # ========================================================================
    # Example 5: Using convenience function
    # ========================================================================
    print("\n5. Using convenience function...")
    print("-" * 70)

    # For one-off time requests, you can use the convenience function
    quick_time = get_current_time()
    print(f"Quick time access:  {quick_time.corrected_timestamp:.6f}")
    print(f"Uncertainty:        ±{quick_time.uncertainty_seconds * 1000:.3f}ms")

    # ========================================================================
    # Example 6: Context manager usage
    # ========================================================================
    print("\n6. Using context manager (auto cleanup)...")
    print("-" * 70)

    with ChronoTickClient() as client2:
        time_info2 = client2.get_time()
        print(f"Context manager time: {time_info2.corrected_timestamp:.6f}")
        print(f"Confidence:           {time_info2.confidence:.1%}")
    # Client automatically cleaned up here

    print("\n" + "=" * 70)
    print("✅ Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as e:
        print("\n" + "=" * 70)
        print("❌ Error")
        print("=" * 70)
        print(f"\n{e}\n")
        print("Make sure the ChronoTick daemon is running:")
        print("  chronotick-daemon")
    except KeyboardInterrupt:
        print("\n\n⏹  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
