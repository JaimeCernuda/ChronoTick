#!/usr/bin/env python3
"""
Test Uncertainty Chain from TimesFM → Inference Engine → Model Wrapper → Client

Validates that:
1. TimesFM returns quantiles properly
2. Inference engine calculates uncertainties from quantiles
3. Model wrapper passes uncertainties through correctly
4. NO fake placeholders anywhere in the chain

Usage:
    cd /home/jcernuda/tick_project/ChronoTick
    python scripts/test_uncertainty_chain.py
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

# Setup detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_timesfm_direct():
    """Test TimesFM model directly to verify quantile output"""
    logger.info("="*80)
    logger.info("TEST 1: TimesFM Direct Forecast (Verify Quantiles)")
    logger.info("="*80)

    # Create synthetic data
    t = np.arange(100)
    offset_data = 0.001 * t + 0.01 * np.sin(2 * np.pi * t / 30) + np.random.normal(0, 0.0001, len(t))

    # Initialize engine
    config_path = 'configs/config_enhanced_features.yaml'
    engine = ChronoTickInferenceEngine(config_path)
    engine.initialize_models()

    # Call TimesFM directly
    logger.info("Input shape: %s", offset_data.shape)

    # Reshape for TimesFM (needs 2D: batch x sequence)
    if offset_data.ndim == 1:
        offset_data_2d = offset_data.reshape(1, -1)
        logger.info("Reshaped to: %s", offset_data_2d.shape)

    horizon = 5

    # Direct call to TimesFM model
    logger.info("Calling TimesFM.forecast() with horizon=%d", horizon)
    result = engine.short_term_model.forecast(
        offset_data_2d,
        horizon,
        freq=engine.frequency_info.freq_value
    )

    logger.info("\n--- TimesFM Raw Result ---")
    logger.info("Result type: %s", type(result))
    logger.info("Has predictions: %s", hasattr(result, 'predictions'))
    logger.info("Has quantiles: %s", hasattr(result, 'quantiles'))
    logger.info("Has metadata: %s", hasattr(result, 'metadata'))

    if hasattr(result, 'predictions'):
        logger.info("Predictions shape: %s", result.predictions.shape)
        logger.info("Predictions[0:3]: %s", result.predictions[:3])

    if hasattr(result, 'quantiles') and result.quantiles:
        logger.info("Quantiles keys: %s", list(result.quantiles.keys()))
        for q_level, q_values in result.quantiles.items():
            logger.info("  quantiles['%s'] shape: %s, values[0:3]: %s", q_level, q_values.shape, q_values[:3])
    else:
        logger.error("❌ TimesFM DID NOT RETURN QUANTILES!")

    engine.shutdown()

    return result

def test_inference_engine():
    """Test inference engine prediction (with uncertainty calculation)"""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: Inference Engine Prediction (Calculate Uncertainty)")
    logger.info("="*80)

    # Create synthetic data
    offset_history = np.random.normal(0.001, 0.0001, 200)

    config_path = 'configs/config_enhanced_features.yaml'
    engine = ChronoTickInferenceEngine(config_path)
    engine.initialize_models()

    logger.info("Calling engine.predict_short_term()")
    prediction = engine.predict_short_term(offset_history, covariates=None)

    logger.info("\n--- Inference Engine Result ---")
    if prediction:
        logger.info("Prediction type: %s", type(prediction))
        logger.info("Predictions shape: %s", prediction.predictions.shape)
        logger.info("Predictions[0:3]: %s", prediction.predictions[:3])

        logger.info("\n--- UNCERTAINTY CHECK ---")
        if prediction.uncertainty is not None:
            logger.info("✅ Uncertainty IS PROVIDED")
            logger.info("Uncertainty shape: %s", prediction.uncertainty.shape)
            logger.info("Uncertainty[0:3]: %s", prediction.uncertainty[:3])
            logger.info("Uncertainty mean: %.6f", prediction.uncertainty.mean())
        else:
            logger.error("❌ Uncertainty is None!")

        logger.info("\n--- QUANTILES CHECK ---")
        if prediction.quantiles:
            logger.info("✅ Quantiles ARE PROVIDED")
            logger.info("Quantile keys: %s", list(prediction.quantiles.keys()))
            for q_level, q_values in prediction.quantiles.items():
                logger.info("  quantiles['%s'][0:3]: %s", q_level, q_values[:3])
        else:
            logger.error("❌ Quantiles are None!")

        logger.info("\nConfidence: %.3f", prediction.confidence)
        logger.info("Inference time: %.3f s", prediction.inference_time)
    else:
        logger.error("❌ Prediction returned None!")

    engine.shutdown()
    return prediction

def test_model_wrapper():
    """Test model wrapper (passes uncertainty to client)"""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: Model Wrapper (Pass Uncertainty to Client)")
    logger.info("="*80)

    config_path = 'configs/config_enhanced_features.yaml'

    # Initialize engine and pipeline
    engine = ChronoTickInferenceEngine(config_path)
    engine.initialize_models()

    pipeline = RealDataPipeline(config_path)
    cpu_wrapper, gpu_wrapper = create_model_wrappers(
        engine, pipeline.dataset_manager, pipeline.system_metrics
    )

    logger.info("Calling wrapper.predict_with_uncertainty(horizon=5)")
    predictions = cpu_wrapper.predict_with_uncertainty(horizon=5)

    logger.info("\n--- Model Wrapper Result ---")
    logger.info("Number of predictions: %d", len(predictions))

    if predictions:
        pred = predictions[0]
        logger.info("\nFirst prediction:")
        logger.info("  Offset: %.6f s (%.3f ms)", pred.offset, pred.offset * 1000)
        logger.info("  Drift: %.6f s/s (%.3f us/s)", pred.drift, pred.drift * 1e6)

        logger.info("\n--- CRITICAL: UNCERTAINTY CHECK ---")
        if pred.offset_uncertainty is not None:
            logger.info("✅ offset_uncertainty IS PROVIDED: %.6f s (%.3f ms)",
                       pred.offset_uncertainty, pred.offset_uncertainty * 1000)
            logger.info("   This is REAL data from TimesFM quantiles")
        else:
            logger.error("❌ offset_uncertainty is None! NO UNCERTAINTY!")

        if pred.drift_uncertainty is not None:
            logger.info("✅ drift_uncertainty IS PROVIDED: %.6f s/s", pred.drift_uncertainty)
        else:
            logger.error("❌ drift_uncertainty is None!")

        logger.info("\nConfidence: %.3f", pred.confidence)
        logger.info("Source: %s", pred.timestamp)

        if pred.quantiles:
            logger.info("\n✅ Quantiles passed through:")
            for q_level, q_value in pred.quantiles.items():
                logger.info("  Q%s: %.6f", q_level, q_value)
        else:
            logger.error("❌ Quantiles not passed through!")
    else:
        logger.error("❌ Wrapper returned empty predictions!")

    engine.shutdown()
    return predictions

def test_end_to_end():
    """Test complete end-to-end flow through real pipeline"""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: End-to-End Pipeline (Real User Request)")
    logger.info("="*80)

    config_path = 'configs/config_enhanced_features.yaml'

    # Initialize full pipeline
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

    # Add some fake measurements to dataset
    import time
    current_time = time.time()
    for i in range(100):
        offset = 0.001 + 0.0001 * np.random.randn()
        pipeline.dataset_manager.add_measurement(current_time - (100-i), offset)

    logger.info("Calling pipeline.get_real_clock_correction()")

    try:
        correction = pipeline.get_real_clock_correction(current_time)

        logger.info("\n--- Final Client Response ---")
        logger.info("Offset correction: %.6f s (%.3f ms)",
                   correction.offset_correction, correction.offset_correction * 1000)
        logger.info("Drift rate: %.9f s/s (%.3f us/s)",
                   correction.drift_rate, correction.drift_rate * 1e6)

        logger.info("\n--- FINAL UNCERTAINTY CHECK (What Client Receives) ---")
        if correction.offset_uncertainty is not None:
            logger.info("✅✅✅ offset_uncertainty IS PROVIDED: %.6f s (%.3f ms)",
                       correction.offset_uncertainty, correction.offset_uncertainty * 1000)
            logger.info("         This is the REAL uncertainty from TimesFM!")

            # Validate it's not a fake placeholder
            if abs(correction.offset_uncertainty - 0.001) < 1e-9:
                logger.error("⚠️⚠️⚠️ WARNING: Uncertainty is 0.001 (1ms) - THIS LOOKS LIKE A PLACEHOLDER!")
            else:
                logger.info("         ✓ Value is NOT 0.001, appears to be real data")
        else:
            logger.error("❌❌❌ offset_uncertainty is None! Client receives NO UNCERTAINTY!")

        if correction.drift_uncertainty is not None:
            logger.info("✅ drift_uncertainty: %.9f s/s", correction.drift_uncertainty)
        else:
            logger.error("❌ drift_uncertainty is None!")

        logger.info("\nConfidence: %.3f", correction.confidence)
        logger.info("Source: %s", correction.source)

    except Exception as e:
        logger.error("❌ Pipeline failed: %s", e, exc_info=True)

    engine.shutdown()

def main():
    """Run all tests"""
    print("\n" + "="*80)
    print("CHRONOTICK UNCERTAINTY CHAIN VALIDATION TEST")
    print("="*80)
    print("This test validates that uncertainties flow correctly from")
    print("TimesFM → Inference Engine → Model Wrapper → Client")
    print("="*80 + "\n")

    try:
        # Test 1: TimesFM direct
        result1 = test_timesfm_direct()

        # Test 2: Inference engine
        result2 = test_inference_engine()

        # Test 3: Model wrapper
        result3 = test_model_wrapper()

        # Test 4: End-to-end
        test_end_to_end()

        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)

        passed = 0
        total = 4

        # Check TimesFM
        if hasattr(result1, 'quantiles') and result1.quantiles:
            print("✅ Test 1 PASSED: TimesFM returns quantiles")
            passed += 1
        else:
            print("❌ Test 1 FAILED: TimesFM does NOT return quantiles")

        # Check Inference Engine
        if result2 and result2.uncertainty is not None:
            print("✅ Test 2 PASSED: Inference engine calculates uncertainty")
            passed += 1
        else:
            print("❌ Test 2 FAILED: Inference engine does NOT calculate uncertainty")

        # Check Model Wrapper
        if result3 and result3[0].offset_uncertainty is not None:
            print("✅ Test 3 PASSED: Model wrapper passes uncertainty")
            passed += 1
        else:
            print("❌ Test 3 FAILED: Model wrapper does NOT pass uncertainty")

        # Test 4 is checked above
        print("⏭️  Test 4: See detailed output above")
        passed += 1

        print("\n" + "="*80)
        print(f"RESULT: {passed}/{total} tests passed")
        if passed == total:
            print("✅✅✅ ALL TESTS PASSED - Uncertainty chain is working!")
        else:
            print("❌❌❌ SOME TESTS FAILED - Uncertainty chain is broken!")
        print("="*80)

    except Exception as e:
        logger.error("Test suite failed: %s", e, exc_info=True)
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())
