#!/usr/bin/env python3
"""Test TSFMModelWrapper API fixes in isolation"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from chronotick.inference.engine import ChronoTickInferenceEngine
from chronotick.inference.real_data_pipeline import RealDataPipeline
from chronotick.inference.tsfm_model_wrapper import create_model_wrappers

print("=" * 60)
print("Testing TSFMModelWrapper API fixes")
print("=" * 60)

config_path = "configs/config.yaml"

# Initialize engine
print("\n1. Initializing engine...")
engine = ChronoTickInferenceEngine(config_path)
engine.initialize_models()
print("✓ Engine initialized")

# Initialize pipeline (for dataset manager and metrics)
print("\n2. Initializing pipeline...")
pipeline = RealDataPipeline(config_path)
print("✓ Pipeline initialized")

# Create wrappers
print("\n3. Creating wrappers...")
cpu_wrapper, gpu_wrapper = create_model_wrappers(
    engine, pipeline.dataset_manager, pipeline.system_metrics
)
print(f"✓ CPU wrapper: {cpu_wrapper.model_type}")
print(f"✓ GPU wrapper: {gpu_wrapper.model_type}")

# Test wrapper can be called (will use fallback since no real data yet)
print("\n4. Testing wrapper prediction (will use fallback)...")
try:
    predictions = cpu_wrapper.predict_with_uncertainty(horizon=5)
    print(f"✓ Got {len(predictions)} predictions")

    # Check the prediction has correct fields
    pred = predictions[0]
    print(f"\nPrediction fields:")
    print(f"  - offset: {pred.offset}")
    print(f"  - drift: {pred.drift}")
    print(f"  - offset_uncertainty: {pred.offset_uncertainty}")
    print(f"  - drift_uncertainty: {pred.drift_uncertainty}")
    print(f"  - confidence: {pred.confidence}")
    print(f"  - timestamp: {pred.timestamp}")

    print("\n✅ All API fields present and accessible!")

except Exception as e:
    print(f"✗ Wrapper prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Cleanup
engine.shutdown()
print("\n✓ Test complete!")
