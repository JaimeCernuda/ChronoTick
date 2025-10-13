#!/usr/bin/env python3
"""
CLIENT INTEGRATION TEST - End-to-End Validation
Tests the full ChronoTick stack from the client perspective using the public API.

This test simulates a real client:
1. Starts ChronoTick using chronotick.start()
2. Waits for warmup to complete
3. Calls chronotick.time() to get corrected timestamps
4. Validates ML predictions are being used
5. Verifies no fallbacks occurred
"""

import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Minimal logging - only warnings and errors
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)-8s | %(name)-30s | %(message)s'
)

import chronotick

print("=" * 80)
print("CLIENT INTEGRATION TEST - Full Stack Validation")
print("Testing: chronotick.start() → chronotick.time() with ML predictions")
print("=" * 80)
print()

# Get configuration
config_path = str(Path(__file__).parent.parent / "chronotick_inference" / "config.yaml")

# Step 1: Start ChronoTick
print("Step 1: Starting ChronoTick daemon...")
print(f"Config: {config_path}")
print()

success = chronotick.start(config_path=config_path)

if not success:
    print("❌ FAILED: Could not start ChronoTick daemon")
    sys.exit(1)

print("✓ ChronoTick daemon started\n")

# Step 2: Check initial status
print("Step 2: Checking daemon status...")
status = chronotick.status()
print(f"  Started: {status['started']}")
print(f"  Daemon running: {status.get('daemon_running', 'unknown')}")
print(f"  Daemon PID: {status.get('daemon_pid', 'unknown')}")
print()

# Step 3: Wait for warmup
print("Step 3: Waiting for warmup to complete...")
print("  ChronoTick daemon has a 60s warmup period to collect NTP data")
print("  We'll check status every 10 seconds and query once warmup is complete")
print()

warmup_duration = 60
warmup_buffer = 25  # Extra time for ML models to load and scheduler to start

for i in range(0, warmup_duration + warmup_buffer, 10):
    time.sleep(10)
    status = chronotick.status()
    print(f"  [{i+10}/{warmup_duration+warmup_buffer}s] Status: calls={status['total_calls']}, "
          f"success_rate={status['success_rate']:.1%}")

print()
print("✓ Warmup period complete\n")

# Step 4: Main test - Get corrected time from client API
print("=" * 80)
print("Step 4: MAIN TEST - Calling chronotick.time() API")
print("Testing 20 samples over 100 seconds (5s intervals)")
print("=" * 80)
print()

test_duration = 100  # seconds
interval = 5
iterations = test_duration // interval

results = []
errors = []

for i in range(iterations):
    try:
        # Get detailed time with all metadata
        corrected = chronotick.time_detailed()

        result = {
            'iteration': i + 1,
            'corrected_time': corrected.timestamp,
            'raw_time': corrected.raw_timestamp,
            'offset_ms': corrected.offset_correction * 1000,
            'uncertainty_ms': corrected.uncertainty * 1000 if corrected.uncertainty else None,
            'confidence': corrected.confidence
        }
        results.append(result)

        offset_ms = corrected.offset_correction * 1000
        uncertainty_str = f"±{corrected.uncertainty*1000:.3f}ms" if corrected.uncertainty else "N/A"
        confidence_str = f"{corrected.confidence:.2f}" if corrected.confidence else "N/A"

        print(f"[{i+1:2d}/{iterations}] "
              f"offset={offset_ms:>8.3f}ms "
              f"uncertainty={uncertainty_str:>12s} "
              f"confidence={confidence_str:>4s}")

        if i < iterations - 1:
            time.sleep(interval)

    except Exception as e:
        errors.append(str(e))
        print(f"[{i+1:2d}/{iterations}] ERROR: {e}")
        break

print()
print("=" * 80)
print("TEST RESULTS")
print("=" * 80)
print()

# Get final status
final_status = chronotick.status()

print("Client API Statistics:")
print(f"  Total API calls: {final_status['total_calls']}")
print(f"  Successful calls: {final_status['successful_calls']}")
print(f"  Fallback calls: {final_status['fallback_calls']}")
print(f"  Success rate: {final_status['success_rate']:.1%}")
print()

print("Test Results:")
print(f"  Test samples collected: {len(results)}")
print(f"  Errors encountered: {len(errors)}")
if results:
    avg_offset = sum(r['offset_ms'] for r in results) / len(results)
    print(f"  Average offset correction: {avg_offset:.3f}ms")

    valid_uncertainty = [r for r in results if r['uncertainty_ms'] is not None]
    if valid_uncertainty:
        avg_uncertainty = sum(r['uncertainty_ms'] for r in valid_uncertainty) / len(valid_uncertainty)
        print(f"  Average uncertainty: ±{avg_uncertainty:.3f}ms")

    valid_confidence = [r for r in results if r['confidence'] is not None]
    if valid_confidence:
        avg_confidence = sum(r['confidence'] for r in valid_confidence) / len(valid_confidence)
        print(f"  Average confidence: {avg_confidence:.2f}")
print()

# Check daemon statistics if available
if 'daemon_requests' in final_status:
    print("Daemon Statistics:")
    print(f"  Total daemon requests: {final_status['daemon_requests']}")
    print(f"  Daemon success rate: {final_status.get('daemon_success_rate', 0):.1%}")
    print(f"  Daemon memory usage: {final_status.get('daemon_memory_mb', 0):.1f}MB")
    print(f"  Daemon uptime: {final_status.get('daemon_uptime', 0):.1f}s")
    if 'avg_inference_time_ms' in final_status and final_status['avg_inference_time_ms']:
        print(f"  Avg inference time: {final_status['avg_inference_time_ms']:.3f}ms")
    print()

# Final verdict
print("Final Verdict:")
if errors:
    print("❌ TEST FAILED - Errors occurred:")
    for err in errors:
        print(f"   {err}")
elif final_status['fallback_calls'] > 0:
    fallback_pct = (final_status['fallback_calls'] / final_status['total_calls']) * 100
    print(f"⚠️  PARTIAL SUCCESS: {final_status['fallback_calls']} fallback calls ({fallback_pct:.1f}%)")
    print("   Some requests used fallback instead of ML predictions")
elif final_status['success_rate'] == 1.0 and len(results) == iterations:
    print("✅ SUCCESS: Complete end-to-end validation passed!")
    print("   ✓ ChronoTick daemon started successfully")
    print("   ✓ Client API working correctly")
    print("   ✓ All time requests successful")
    print("   ✓ No fallbacks - ML predictions in use")
    print("   ✓ Full stack integration verified")
else:
    print(f"⚠️  PARTIAL SUCCESS: {len(results)}/{iterations} samples collected")
    print(f"   Success rate: {final_status['success_rate']:.1%}")

print()

# Cleanup
print("Stopping ChronoTick daemon...")
chronotick.stop()
print("✓ Test complete")
