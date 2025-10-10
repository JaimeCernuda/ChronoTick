#!/usr/bin/env python3
"""
Test script for debug logging functionality.
"""

import numpy as np
import logging

# Setup logging to see everything
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Import debug logger and enable it
from chronotick_inference.debug_logger import enable_debug, disable_debug
from tsfm import TSFMFactory

def test_debug_logging():
    """Test debug logging with TimesFM."""

    print("="*80)
    print("Testing Debug Logging")
    print("="*80)

    # Test 1: Verify debug logger works
    print("\n[1/3] Testing debug logger utilities...")
    enable_debug()
    print("✓ Debug logging enabled")

    # Test 2: Load model with debug logging
    print("\n[2/3] Loading TimesFM with debug logging...")
    factory = TSFMFactory()
    model = factory.load_model('timesfm', device='cpu')
    print("✓ Model loaded (check debug output above)")

    # Test 3: Forecast with covariates and debug logging
    print("\n[3/3] Testing forecast_with_covariates with debug logging...")

    # Create simple test data
    np.random.seed(42)
    horizon = 5
    target = np.linspace(10, 20, 50) + np.random.randn(50) * 0.3
    cpu_usage = 50 + np.random.randn(50 + horizon) * 5
    temperature = 60 + np.random.randn(50 + horizon) * 3

    # Create CovariatesInput
    covariates_input = factory.create_covariates_input(
        target=target,
        covariates={
            'cpu_usage': cpu_usage,
            'temperature': temperature
        }
    )

    print("\n--- Testing WITH covariates (use_covariates=True) ---")
    result_with = model.forecast_with_covariates(
        covariates_input,
        horizon=horizon,
        use_covariates=True
    )
    print(f"✓ Predictions: {result_with.predictions}")

    print("\n--- Testing WITHOUT covariates (use_covariates=False) ---")
    result_without = model.forecast_with_covariates(
        covariates_input,
        horizon=horizon,
        use_covariates=False
    )
    print(f"✓ Predictions: {result_without.predictions}")

    print("\n" + "="*80)
    print("✓ ALL DEBUG LOGGING TESTS PASSED!")
    print("="*80)
    print("\nCheck the debug output above to verify:")
    print("1. Function calls are logged with inputs/outputs")
    print("2. Timers show execution duration")
    print("3. Variables are logged at key points")
    print("4. Section separators make output readable")

if __name__ == '__main__':
    test_debug_logging()
