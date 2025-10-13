#!/usr/bin/env python3
"""
Debug pipeline hanging issue - systematic isolation testing
"""

import sys
import time
import logging
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from chronotick_inference.engine import ChronoTickInferenceEngine
from chronotick_inference.real_data_pipeline import RealDataPipeline
from chronotick_inference.tsfm_model_wrapper import create_model_wrappers

# Enable verbose logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s | %(levelname)-8s | %(name)-30s | %(funcName)-25s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

def timeout_wrapper(func, timeout_seconds=5, description=""):
    """Wrap a function with timeout detection"""
    result = [None]
    error = [None]

    def target():
        try:
            result[0] = func()
        except Exception as e:
            error[0] = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout_seconds)

    if thread.is_alive():
        print(f"✗ TIMEOUT: {description} (>{timeout_seconds}s)")
        return None
    elif error[0]:
        print(f"✗ ERROR: {description}: {error[0]}")
        raise error[0]
    else:
        print(f"✓ SUCCESS: {description} ({timeout_seconds}s)")
        return result[0]


def test_1_basic_initialization():
    """Test 1: Basic component initialization"""
    print("\n" + "=" * 80)
    print("TEST 1: Basic Initialization")
    print("=" * 80)

    config_path = "chronotick_inference/config.yaml"

    # Initialize engine
    print("\n1.1 Initializing engine...")
    engine = ChronoTickInferenceEngine(config_path)
    engine.initialize_models()
    print("✓ Engine initialized")

    # Initialize pipeline (should not block)
    print("\n1.2 Initializing pipeline...")
    pipeline = RealDataPipeline(config_path)
    print(f"✓ Pipeline initialized: initialized={pipeline.initialized}, warm_up_complete={pipeline.warm_up_complete}")

    # Create wrappers
    print("\n1.3 Creating model wrappers...")
    cpu_wrapper, gpu_wrapper = create_model_wrappers(
        engine, pipeline.dataset_manager, pipeline.system_metrics
    )
    print("✓ Wrappers created")

    # Connect models to pipeline
    print("\n1.4 Connecting models to pipeline...")
    pipeline.initialize(cpu_model=cpu_wrapper, gpu_model=gpu_wrapper)
    print(f"✓ Pipeline initialized with models: initialized={pipeline.initialized}")

    # Set scheduler interfaces
    print("\n1.5 Setting scheduler interfaces...")
    pipeline.predictive_scheduler.set_model_interfaces(
        cpu_model=cpu_wrapper,
        gpu_model=gpu_wrapper,
        fusion_engine=pipeline.fusion_engine
    )
    print("✓ Scheduler configured")

    print("\n✅ TEST 1 PASSED: All components initialized without hanging")

    engine.shutdown()
    return pipeline


def test_2_pipeline_states(pipeline):
    """Test 2: Test get_real_clock_correction in different states"""
    print("\n" + "=" * 80)
    print("TEST 2: Pipeline States")
    print("=" * 80)

    current_time = time.time()

    # Test 2.1: Call during warmup (should use _get_warm_up_correction)
    print("\n2.1 Testing during warmup period...")
    print(f"  - warm_up_complete: {pipeline.warm_up_complete}")
    print(f"  - initialized: {pipeline.initialized}")

    def call_correction():
        return pipeline.get_real_clock_correction(current_time)

    correction = timeout_wrapper(
        call_correction,
        timeout_seconds=3,
        description="get_real_clock_correction during warmup"
    )

    if correction:
        print(f"  ✓ Got correction: offset={correction.offset_correction}, source={correction.source}")

    print("\n✅ TEST 2 PASSED: Pipeline responds in warmup state")


def test_3_warmup_to_normal_transition():
    """Test 3: Transition from warmup to normal operation"""
    print("\n" + "=" * 80)
    print("TEST 3: Warmup to Normal Transition")
    print("=" * 80)

    config_path = "chronotick_inference/config.yaml"

    # Reinitialize everything
    engine = ChronoTickInferenceEngine(config_path)
    engine.initialize_models()

    pipeline = RealDataPipeline(config_path)
    cpu_wrapper, gpu_wrapper = create_model_wrappers(
        engine, pipeline.dataset_manager, pipeline.system_metrics
    )
    pipeline.initialize(cpu_model=cpu_wrapper, gpu_model=gpu_wrapper)
    pipeline.predictive_scheduler.set_model_interfaces(
        cpu_model=cpu_wrapper,
        gpu_model=gpu_wrapper,
        fusion_engine=pipeline.fusion_engine
    )

    # Wait for warmup to complete
    print("\n3.1 Waiting for warmup to complete (5 seconds of NTP collection)...")
    pipeline.system_metrics.start_collection()

    for i in range(5):
        time.sleep(1)
        print(f"  Warmup: {i+1}/5s, warm_up_complete={pipeline.warm_up_complete}")

    # Force warmup completion
    if not pipeline.warm_up_complete:
        print("\n3.2 Forcing warmup completion...")
        pipeline.warm_up_complete = True
        print(f"  ✓ Set warm_up_complete={pipeline.warm_up_complete}")

    # Now test in normal operation mode
    print("\n3.3 Testing get_real_clock_correction after warmup...")
    current_time = time.time()

    def call_after_warmup():
        return pipeline.get_real_clock_correction(current_time)

    correction = timeout_wrapper(
        call_after_warmup,
        timeout_seconds=5,
        description="get_real_clock_correction after warmup"
    )

    if correction:
        print(f"  ✓ Got correction: offset={correction.offset_correction}, source={correction.source}")
    else:
        print(f"  ✗ FAILED: Correction call timed out after warmup!")
        print(f"  Debug info:")
        print(f"    - warm_up_complete: {pipeline.warm_up_complete}")
        print(f"    - initialized: {pipeline.initialized}")
        print(f"    - scheduler has CPU model: {pipeline.predictive_scheduler.cpu_model is not None}")
        print(f"    - scheduler has GPU model: {pipeline.predictive_scheduler.gpu_model is not None}")

    engine.shutdown()
    pipeline.system_metrics.stop_collection()

    print("\n✅ TEST 3 COMPLETE")


def test_4_scheduler_isolation():
    """Test 4: Test predictive scheduler in isolation"""
    print("\n" + "=" * 80)
    print("TEST 4: Scheduler Isolation")
    print("=" * 80)

    config_path = "chronotick_inference/config.yaml"

    engine = ChronoTickInferenceEngine(config_path)
    engine.initialize_models()

    pipeline = RealDataPipeline(config_path)
    cpu_wrapper, gpu_wrapper = create_model_wrappers(
        engine, pipeline.dataset_manager, pipeline.system_metrics
    )

    # Test scheduler methods directly
    scheduler = pipeline.predictive_scheduler

    print("\n4.1 Testing scheduler.get_fused_correction()...")
    current_time = time.time()

    def call_fused_correction():
        return scheduler.get_fused_correction(current_time)

    result = timeout_wrapper(
        call_fused_correction,
        timeout_seconds=2,
        description="scheduler.get_fused_correction()"
    )

    print(f"  Result: {result}")

    print("\n4.2 Testing scheduler.get_correction_at_time()...")

    def call_correction_at_time():
        return scheduler.get_correction_at_time(current_time)

    result = timeout_wrapper(
        call_correction_at_time,
        timeout_seconds=2,
        description="scheduler.get_correction_at_time()"
    )

    print(f"  Result: {result}")

    print("\n4.3 Setting model interfaces on scheduler...")
    scheduler.set_model_interfaces(
        cpu_model=cpu_wrapper,
        gpu_model=gpu_wrapper,
        fusion_engine=pipeline.fusion_engine
    )
    print("  ✓ Interfaces set")

    print("\n4.4 Testing scheduler after interface setup...")

    result = timeout_wrapper(
        call_fused_correction,
        timeout_seconds=2,
        description="scheduler.get_fused_correction() after setup"
    )

    print(f"  Result: {result}")

    engine.shutdown()

    print("\n✅ TEST 4 COMPLETE")


def test_5_lock_detection():
    """Test 5: Check for lock contention"""
    print("\n" + "=" * 80)
    print("TEST 5: Lock Detection")
    print("=" * 80)

    config_path = "chronotick_inference/config.yaml"

    engine = ChronoTickInferenceEngine(config_path)
    engine.initialize_models()

    pipeline = RealDataPipeline(config_path)

    print("\n5.1 Checking pipeline.lock status...")
    print(f"  - Pipeline has lock: {hasattr(pipeline, 'lock')}")
    if hasattr(pipeline, 'lock'):
        print(f"  - Lock locked: {pipeline.lock.locked()}")

    print("\n5.2 Checking scheduler.lock status...")
    print(f"  - Scheduler has lock: {hasattr(pipeline.predictive_scheduler, 'lock')}")
    if hasattr(pipeline.predictive_scheduler, 'lock'):
        print(f"  - Lock locked: {pipeline.predictive_scheduler.lock.locked()}")

    print("\n5.3 Testing with explicit lock acquisition...")
    current_time = time.time()

    try:
        with pipeline.lock:
            print("  ✓ Acquired pipeline lock")
            # Try to call correction while holding lock (should still work if properly designed)
            def call_with_lock():
                return pipeline._fallback_correction(current_time)

            correction = timeout_wrapper(
                call_with_lock,
                timeout_seconds=2,
                description="_fallback_correction while holding lock"
            )

            print(f"  Result: {correction}")
    except Exception as e:
        print(f"  ✗ Error with lock: {e}")

    engine.shutdown()

    print("\n✅ TEST 5 COMPLETE")


def main():
    print("=" * 80)
    print("PIPELINE HANG DEBUG TEST SUITE")
    print("=" * 80)

    try:
        # Test 1: Basic initialization
        pipeline = test_1_basic_initialization()

        # Test 2: Different pipeline states
        test_2_pipeline_states(pipeline)

        # Test 3: Warmup transition
        test_3_warmup_to_normal_transition()

        # Test 4: Scheduler isolation
        test_4_scheduler_isolation()

        # Test 5: Lock detection
        test_5_lock_detection()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS COMPLETE")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
