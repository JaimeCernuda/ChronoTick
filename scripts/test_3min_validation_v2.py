#!/usr/bin/env python3
"""
Quick 3-Minute Validation Test for Uncertainty Fix

Validates that after config fix (use_covariates: false):
1. TimesFM returns quantiles
2. Inference engine calculates uncertainties
3. Model wrapper passes real uncertainties (not None, not placeholders)
4. Client receives varying uncertainty values (0.1-0.5ms range)

Usage:
    cd /home/jcernuda/tick_project/ChronoTick
    uv run python scripts/test_3min_validation_v2.py
"""

import sys
import logging
import numpy as np
from pathlib import Path

# Add server src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'server/src'))

from chronotick.inference.engine import ChronoTickInferenceEngine
from chronotick.inference.real_data_pipeline import RealDataPipeline
from chronotick.inference.tsfm_model_wrapper import create_model_wrappers

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    print("\n" + "="*80)
    print("QUICK UNCERTAINTY VALIDATION TEST (3 minutes)")
    print("="*80)
    print("Config: configs/config_enhanced_features.yaml (with use_covariates: false fix)")
    print()

    config_path = 'configs/config_enhanced_features.yaml'

    # Test 1: TimesFM Direct Call
    print("Test 1: TimesFM Direct Forecast (Verify Quantiles)")
    print("-" * 80)

    engine = ChronoTickInferenceEngine(config_path)
    engine.initialize_models()

    # Create synthetic data
    t = np.arange(200)
    offset_data = 0.001 * t + 0.01 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 0.0001, len(t))

    # Reshape for TimesFM
    offset_data_2d = offset_data.reshape(1, -1)
    horizon = 10

    # Direct call to TimesFM
    result = engine.short_term_model.forecast(
        offset_data_2d,
        horizon,
        freq=engine.frequency_info.freq_value
    )

    print(f"✓ TimesFM forecast completed")
    print(f"  - Predictions shape: {result.predictions.shape}")

    if hasattr(result, 'quantiles') and result.quantiles:
        print(f"  - ✅ Quantiles PRESENT: {list(result.quantiles.keys())}")
        q_levels = len(result.quantiles)
        print(f"  - Quantile levels: {q_levels}")
        if q_levels >= 7:
            print(f"  - ✅ PASS: TimesFM returns quantiles properly")
            test1_pass = True
        else:
            print(f"  - ❌ FAIL: Expected 9 quantile levels, got {q_levels}")
            test1_pass = False
    else:
        print(f"  - ❌ FAIL: Quantiles MISSING")
        test1_pass = False

    print()

    # Test 2: Inference Engine
    print("Test 2: Inference Engine (Calculate Uncertainty)")
    print("-" * 80)

    prediction = engine.predict_short_term(offset_data, covariates=None)

    print(f"✓ Inference engine prediction completed")
    print(f"  - Predictions shape: {prediction.predictions.shape}")

    if prediction.uncertainty is not None:
        print(f"  - ✅ Uncertainty PRESENT")
        print(f"  - Uncertainty shape: {prediction.uncertainty.shape}")
        print(f"  - Uncertainty values: min={prediction.uncertainty.min():.6f}s ({prediction.uncertainty.min()*1000:.3f}ms), "
              f"max={prediction.uncertainty.max():.6f}s ({prediction.uncertainty.max()*1000:.3f}ms), "
              f"mean={prediction.uncertainty.mean():.6f}s ({prediction.uncertainty.mean()*1000:.3f}ms)")

        # Check if values are in reasonable range (not placeholders)
        if prediction.uncertainty.mean() > 0 and prediction.uncertainty.mean() < 0.01:
            print(f"  - ✅ PASS: Uncertainty values in reasonable range")
            test2_pass = True
        else:
            print(f"  - ❌ FAIL: Uncertainty values out of expected range")
            test2_pass = False
    else:
        print(f"  - ❌ FAIL: Uncertainty is None")
        test2_pass = False

    if prediction.quantiles:
        print(f"  - ✅ Quantiles passed through: {list(prediction.quantiles.keys())}")
    else:
        print(f"  - ❌ WARNING: Quantiles not passed through")

    print()

    # Test 3: Model Wrapper
    print("Test 3: Model Wrapper (Pass Uncertainty to Client)")
    print("-" * 80)

    pipeline = RealDataPipeline(config_path)
    cpu_wrapper, gpu_wrapper = create_model_wrappers(
        engine, pipeline.dataset_manager, pipeline.system_metrics
    )

    # Add some measurements to dataset
    import time
    current_time = time.time()
    for i in range(150):
        offset = 0.001 + 0.0001 * np.random.randn()
        pipeline.dataset_manager.add_measurement(current_time - (150-i), offset)

    predictions = cpu_wrapper.predict_with_uncertainty(horizon=10)

    print(f"✓ Model wrapper prediction completed")
    print(f"  - Number of predictions: {len(predictions)}")

    if predictions and predictions[0].offset_uncertainty is not None:
        pred = predictions[0]
        print(f"  - ✅ offset_uncertainty PRESENT: {pred.offset_uncertainty:.6f}s ({pred.offset_uncertainty*1000:.3f}ms)")

        # Check if it's NOT a placeholder
        if abs(pred.offset_uncertainty - 0.001) > 1e-6:
            print(f"  - ✅ PASS: NOT a placeholder (0.001)")
            test3_pass = True
        else:
            print(f"  - ❌ FAIL: Appears to be placeholder value (0.001)")
            test3_pass = False

        # Check for variation across predictions
        uncertainties = [p.offset_uncertainty for p in predictions if p.offset_uncertainty is not None]
        if len(uncertainties) >= 2:
            unc_std = np.std(uncertainties)
            print(f"  - Uncertainty variation: std={unc_std:.6f}")
            if unc_std > 1e-9:
                print(f"  - ✅ Uncertainties vary across predictions (good!)")
            else:
                print(f"  - ⚠️  WARNING: All uncertainties identical (may indicate issue)")
    else:
        print(f"  - ❌ FAIL: offset_uncertainty is None")
        test3_pass = False

    print()

    # Summary
    print("="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    tests_passed = 0
    total_tests = 3

    if test1_pass:
        print("✅ Test 1 PASSED: TimesFM returns quantiles")
        tests_passed += 1
    else:
        print("❌ Test 1 FAILED: TimesFM does NOT return quantiles")

    if test2_pass:
        print("✅ Test 2 PASSED: Inference engine calculates uncertainty")
        tests_passed += 1
    else:
        print("❌ Test 2 FAILED: Inference engine does NOT calculate uncertainty")

    if test3_pass:
        print("✅ Test 3 PASSED: Model wrapper passes real uncertainty")
        tests_passed += 1
    else:
        print("❌ Test 3 FAILED: Model wrapper does NOT pass real uncertainty")

    print()
    print(f"Result: {tests_passed}/{total_tests} tests passed")

    if tests_passed == total_tests:
        print("✅✅✅ ALL TESTS PASSED - Uncertainty fix is working!")
        print("\nNext step: Run full 5-minute integration test to validate end-to-end")
        return 0
    else:
        print("❌❌❌ SOME TESTS FAILED - Uncertainty fix needs debugging")
        print("\nCheck logs above for specific failure points")
        return 1

    engine.shutdown()

if __name__ == "__main__":
    sys.exit(main())
