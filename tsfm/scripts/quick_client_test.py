#!/usr/bin/env python3
"""
QUICK CLIENT TEST - Short End-to-End Validation
Tests full stack with detailed call tracing to verify no fake data.

Duration: ~90 seconds (60s warmup + buffer + 10s test)
Logging: Comprehensive call chain tracing
"""

import sys
import time
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# DETAILED LOGGING - trace all calls
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)-35s | %(funcName)-25s | %(message)s',
    datefmt='%H:%M:%S'
)

# Enable debug for critical modules to trace call chain
logging.getLogger('chronotick_inference.real_data_pipeline').setLevel(logging.DEBUG)
logging.getLogger('chronotick_inference.predictive_scheduler').setLevel(logging.DEBUG)
logging.getLogger('chronotick_inference.tsfm_model_wrapper').setLevel(logging.DEBUG)
logging.getLogger('ChronoTick').setLevel(logging.DEBUG)

import chronotick

print("=" * 80)
print("QUICK CLIENT TEST - Full Stack with Call Tracing")
print("Duration: ~90 seconds")
print("=" * 80)
print()

# Configuration - using config_complete.yaml with all required sections
config_path = str(Path(__file__).parent.parent / "chronotick_inference" / "config_complete.yaml")

# Step 1: Start ChronoTick
print("Step 1: Starting ChronoTick...")
print(f"Config: {config_path}")
print()

success = chronotick.start(config_path=config_path, auto_config=False)  # Disable auto-config to use our complete config

if not success:
    print("❌ FAILED: Could not start ChronoTick")
    sys.exit(1)

print("✓ ChronoTick started\n")

# Step 2: Wait for warmup (shorter buffer for quick test)
print("Step 2: Waiting for warmup (60s) + buffer (15s)...")
print()

warmup_time = 75  # 60s warmup + 15s buffer for scheduler

for i in range(0, warmup_time, 15):
    time.sleep(15)
    status = chronotick.status()
    print(f"  [{i+15:2d}/{warmup_time}s] Waiting... (calls={status['total_calls']})")

print()
print("✓ Warmup complete\n")

# Step 3: Test with detailed call tracing
print("=" * 80)
print("Step 3: MAIN TEST - Get corrected time (10 samples)")
print("Detailed logging shows full call chain:")
print("  chronotick.time_detailed() → daemon → pipeline → scheduler → ML model")
print("=" * 80)
print()

results = []

for i in range(10):
    print(f"\n{'='*60}")
    print(f"Sample {i+1}/10 - Tracing full call chain:")
    print(f"{'='*60}")

    try:
        # Call the client API - watch logs for full call trace
        corrected = chronotick.time_detailed()

        result = {
            'sample': i + 1,
            'offset_ms': corrected.offset_correction * 1000 if corrected.offset_correction else 0,
            'uncertainty_ms': corrected.uncertainty * 1000 if corrected.uncertainty else None,
            'confidence': corrected.confidence
        }
        results.append(result)

        print(f"\n✓ Sample {i+1} result:")
        print(f"    Offset: {result['offset_ms']:.3f}ms")
        if result['uncertainty_ms']:
            print(f"    Uncertainty: ±{result['uncertainty_ms']:.3f}ms")
        if result['confidence']:
            print(f"    Confidence: {result['confidence']:.2f}")

        if i < 9:
            time.sleep(0.5)  # Small delay between samples

    except Exception as e:
        print(f"\n❌ Sample {i+1} FAILED: {e}")
        import traceback
        traceback.print_exc()
        break

print()
print("=" * 80)
print("RESULTS")
print("=" * 80)
print()

# Final status
status = chronotick.status()

print("Summary:")
print(f"  Samples collected: {len(results)}/10")
print(f"  Total API calls: {status['total_calls']}")
print(f"  Successful calls: {status['successful_calls']}")
print(f"  Fallback calls: {status['fallback_calls']}")
print(f"  Success rate: {status['success_rate']:.1%}")
print()

if results:
    avg_offset = sum(r['offset_ms'] for r in results) / len(results)
    print(f"  Average offset: {avg_offset:.3f}ms")

    valid_conf = [r for r in results if r['confidence'] is not None]
    if valid_conf:
        avg_conf = sum(r['confidence'] for r in valid_conf) / len(valid_conf)
        print(f"  Average confidence: {avg_conf:.2f}")

print()

# Verdict
if len(results) == 10 and status['fallback_calls'] == 0:
    print("✅ SUCCESS: All 10 samples from ML pipeline!")
    print("   Check logs above to verify full call chain:")
    print("   - Client API (chronotick.time_detailed)")
    print("   - Daemon communication")
    print("   - Pipeline (get_real_clock_correction)")
    print("   - Scheduler (get_correction_at_time)")
    print("   - ML Model (predict_with_uncertainty)")
elif len(results) == 10:
    print(f"⚠️  PARTIAL: {status['fallback_calls']} fallback calls")
else:
    print(f"❌ FAILED: Only {len(results)}/10 samples collected")

print()

# Cleanup
print("Stopping ChronoTick...")
chronotick.stop()
print("✓ Test complete")
